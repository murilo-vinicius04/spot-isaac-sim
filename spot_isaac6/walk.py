# Spot + spot_legcol60 walking in PhysX (Isaac Sim 6.0). Structure from IsaacRobotics applications/ (SpotLocoPolicy +
# SpotRunner: joints mapped by NAME, control from a per-physics-step callback), rewritten on the Isaac 6 APIs
# (SimulationManager + physics tensor view). The policy contract is in legcol60.py / policies/spot_legcol60/PORT_CONTRACT.md.
#   /isaac-sim/python.sh spot_isaac6/walk.py [--stage stages/spot_floor.usda] [--vx 0.5] [--steps 500] [--stop-after S]
#       [--track /World/link1 ...] [--act explicit|implicit] [--no-video]
# Any stage made by make_spot_stage.py works (open floor, or Spot added on top of your own scene).
# Writes <out>/trace.npz in crawl-lab's eval trace schema (score it with validation/plot.py), <out>/RESULT.json and,
# with video on, <out>/frames/*.png (one per rendered update, camera following the robot).
import argparse, json, os, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--stage", default=os.path.join(HERE, "stages", "spot_floor.usda"))
ap.add_argument("--out", default=None, help="default: runs/<stage name>, so the floor and chain runs do not overwrite each other")
ap.add_argument("--onnx", default=os.path.join(HERE, "policies", "spot_legcol60", "spot_legcol60.onnx"))
ap.add_argument("--act", choices=["explicit", "implicit"], default="explicit", help="must match how the stage was built")
ap.add_argument("--vx", type=float, default=0.5); ap.add_argument("--vy", type=float, default=0.0); ap.add_argument("--wz", type=float, default=0.0)
ap.add_argument("--height", type=float, default=0.56, help="0.56 = what the Newton reference eval ran (C.3.2); deploy value 0.58")
ap.add_argument("--steps", type=int, default=500, help="control steps at 50 Hz (Newton reference: 500)")
ap.add_argument("--lag", type=int, default=2, help="last_actions lag in policy queries (C.0 trap 2: 2 is what trained)")
ap.add_argument("--no-arm-fix", action="store_true", help="feed raw arm angles (C.0 trap 1) - for A/B only")
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--track", nargs="*", default=[], help="rigid-body prims whose displacement to report (e.g. objects Spot must not disturb)")
ap.add_argument("--stop-after", type=float, default=None, help="switch the command to (0,0,0) at this time (s). The policy never\n                saw a mid-episode command change in training (C.3.1), so stopping is itself under test")
a = ap.parse_args()
if a.out is None: a.out = os.path.join(HERE, "runs", os.path.splitext(os.path.basename(a.stage))[0])
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "multi_gpu": False})
import numpy as np, omni.usd, omni.timeline, warp as wp, sys
sys.path.insert(0, HERE)
from legcol60 import DEPLOY, REF, STOW, pd_torque, rot, Legcol60Policy
from pxr import UsdGeom, Gf
from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent

def wait_for_pngs(d, n_min=0, quiet_s=2.0, max_s=60.0):
    """Replicator writes PNGs on background threads: wait until the count stops changing (and reaches n_min)."""
    import time as _t
    t0, last, since = _t.time(), -1, _t.time()
    while _t.time() - t0 < max_s:
        n = len([f for f in os.listdir(d) if f.endswith(".png")])
        if n != last: last, since = n, _t.time()
        elif n >= n_min and _t.time() - since >= quiet_s: break
        _t.sleep(0.2)
    return last
# ---- stage, camera, video ----------------------------------------------------------------------------------------------
ctx = omni.usd.get_context(); print("[walk] open_stage ->", ctx.open_stage(a.stage), flush=True)
stage = ctx.get_stage()
for _ in range(20): app.update()
os.makedirs(a.out, exist_ok=True)
cam = UsdGeom.Camera.Define(stage, "/World/follow_cam"); cam.CreateFocalLengthAttr(18.0); cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(11.787); cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0)); cam_op = cam.AddTransformOp()
def aim(base):
    tgt = np.array([base[0], base[1], 0.35]); eye = tgt + np.array([1.3, -2.6, 0.9]); f = tgt - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f); M = np.eye(4); M[:3, 0] = r; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    cam_op.Set(Gf.Matrix4d(*M.T.flatten().tolist()))
aim(np.array(UsdGeom.Xformable(stage.GetPrimAtPath("/World/spot")).ComputeLocalToWorldTransform(0).ExtractTranslation()))
if not a.no_video:
    import omni.replicator.core as rep, carb
    # writers attached to a render product write on every update only with capture-on-play enabled. Isaac's default is off;
    # a GUI session can persist it on, which silently made video work on one machine and not on a fresh container.
    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", True)
    rp = rep.create.render_product("/World/follow_cam", (1280, 720)); fdir = os.path.join(a.out, "frames"); os.makedirs(fdir, exist_ok=True)
    for f_ in os.listdir(fdir): os.remove(os.path.join(fdir, f_))
    wr = rep.WriterRegistry.get("BasicWriter"); wr.initialize(output_dir=fdir, rgb=True); wr.attach([rp])   # writes every update

# ---- controller: runs before EVERY 5 ms physics step ---------------------------------------------------------------------
pol = Legcol60Policy(a.onnx, (a.vx, a.vy, a.wz), a.height, lag=a.lag, arm_fix=not a.no_arm_fix)
S = {"art": None, "n": 0, "k": 0, "target": np.concatenate([REF, STOW]), "trace": {}, "fell": None, "t_policy": 0.0, "done": False}
def state():
    art = S["art"]
    # .copy(): on CPU the view's .numpy() aliases one reused buffer, so a stored slice would change under us
    root = art.get_root_transforms().numpy()[0].copy(); vel = art.get_root_velocities().numpy()[0].copy()
    q = art.get_dof_positions().numpy()[0][S["d2a"]]; qd = art.get_dof_velocities().numpy()[0][S["d2a"]]
    L = art.get_link_transforms().numpy()[0].copy(); LV = art.get_link_velocities().numpy()[0].copy()
    R = rot(root[3:7]); return root, vel, R, q.astype(np.float32), qd.astype(np.float32), L, LV
def sample(root, vel, R, q, qd, L, LV, tau):
    fp = L[S["feet"], :3]; fv = LV[S["feet"], :3]
    s = dict(root_pos_w=root[:3], root_quat_w=root[3:7], lin_vel_b=R.T @ vel[:3], ang_vel_b=R.T @ vel[3:6], grav_b=R.T @ np.array([0, 0, -1.0]),
             joint_pos=q, joint_vel=qd, torque=tau, feet_pos_w=fp, feet_vel_w=fv, feet_height=fp[:, 2], feet_pos_b=(fp - root[:3]) @ R,
             cmd=pol.cmd)
    upright = -s["grav_b"][2]; low = L[S["term_bodies"], 2].min()                       # crawl-lab terminations (config.yaml)
    fell = bool(upright < 0.9 or root[2] < 0.3 or low < 0.045)
    s["done"] = np.float32(fell); s["term"] = np.float32(fell)
    for key, v in s.items(): S["trace"].setdefault(key, []).append(np.array(v, np.float32)[None])   # [1 env, ...]
    return fell, upright, low
def on_pre_step(dt, context):
    if S["done"]: return
    if S["art"] is None:
        view = SimulationManager.get_physics_simulation_view()
        if view is None: return
        art = view.create_articulation_view("/World/spot/Geometry/body"); meta = art.shared_metatype
        names, links = list(meta.dof_names), list(meta.link_names)
        S.update(art=art, dofs=names, links=links, d2a=np.array([names.index(j) for j in DEPLOY]),
                 feet=np.array([links.index(f"{f}_foot") for f in ("fl", "fr", "hl", "hr")]),
                 term_bodies=np.array([i for i, n in enumerate(links) if "leg" in n or n == "body"]),
                 idx=wp.array([0], dtype=wp.uint32, device=art.get_dof_positions().device), t0=SimulationManager.get_simulation_time())
        st_ = art.get_dof_stiffnesses().numpy()[0]; fr = art.get_dof_friction_properties().numpy()[0]; arm_ = art.get_dof_armatures().numpy()[0]
        S["check"] = {d: {"stiffness": float(st_[names.index(d)]), "static_friction": float(fr[names.index(d), 0]),
                          "dynamic_friction": float(fr[names.index(d), 1]), "armature": float(arm_[names.index(d)])} for d in DEPLOY}
        c0.update({L: np.array(get_physx_interface().get_rigidbody_transformation(L)["position"]) for L in TRACK})
        print("[walk] articulation dofs", names, "| device", art.get_dof_positions().device, flush=True)
    root, vel, R, q, qd, L, LV = state()
    if S["n"] % 4 == 0:                                        # control-step boundary
        if S["n"] > 0:                                         # close control step k-1: gait update, then the trace sample
            pol.advance_gait(float(np.linalg.norm((R.T @ vel[:3])[:2])))
            fell, up, low = sample(root, vel, R, q, qd, L, LV, S["tau"]); S["k"] += 1
            if fell and S["fell"] is None: S["fell"] = {"step": S["k"], "t": S["k"] * 0.02, "upright": float(up), "base_z": float(root[2]), "lowest_body_z": float(low)}
            if fell or S["k"] >= a.steps: S["done"] = True; return
            if a.stop_after is not None and S["k"] * 0.02 >= a.stop_after and pol.cmd.any():
                pol.cmd = np.zeros(3, np.float32); S["stop_step"] = S["k"]; print("[walk] stop command at t=%.2f" % (S["k"] * 0.02), flush=True)
        t_ = time.perf_counter()
        legs = pol.query(R.T @ vel[3:6], R.T @ np.array([0, 0, -1.0]), q, qd); S["t_policy"] += time.perf_counter() - t_
        S["target"] = np.concatenate([legs, STOW]).astype(np.float32)
        if a.act == "implicit":
            tgt = np.zeros(len(S["dofs"]), np.float32); tgt[S["d2a"]] = S["target"]
            S["art"].set_dof_position_targets(wp.array(tgt[None], dtype=wp.float32, device=S["idx"].device), S["idx"])
    if a.act == "explicit":
        tau = pd_torque(S["target"], q, qd)
        ta = np.zeros(len(S["dofs"]), np.float32); ta[S["d2a"]] = tau
        S["art"].set_dof_actuation_forces(wp.array(ta[None], dtype=wp.float32, device=S["idx"].device), S["idx"])
    else:
        tau = np.zeros(19, np.float32)                         # implicit drive torque is not exposed; recorded as 0
    S["tau"] = tau; S["n"] += 1
SimulationManager.setup_simulation(dt=1.0 / 200.0)
cb = SimulationManager.register_callback(on_pre_step, event=SimulationEvent.PHYSICS_PRE_STEP)
TRACK = [p_ for p_ in a.track if stage.GetPrimAtPath(p_).IsValid()]
if len(TRACK) != len(a.track): print("[walk] --track: not found:", sorted(set(a.track) - set(TRACK)), flush=True)
from omni.physx import get_physx_interface
tl = omni.timeline.get_timeline_interface(); tl.play()
c0 = {}                                                # filled on the first physics step (not valid before)
wall0 = time.time(); updates = 0
while not S["done"] and updates < 20 * a.steps:
    app.update(); updates += 1
    if S["art"] is not None and S["trace"]:
        aim(S["trace"]["root_pos_w"][-1][0])
tl.pause()
c1 = {L: np.array(get_physx_interface().get_rigidbody_transformation(L)["position"]) for L in TRACK}
if not a.no_video:                                             # frames are written asynchronously: wait for all of them
    wr.detach(); print("[walk] video frames written:", wait_for_pngs(fdir, n_min=1), flush=True)
tr = {k: np.stack(v) for k, v in S["trace"].items()}
np.savez(os.path.join(a.out, "trace.npz"), **tr)
rz = tr["root_pos_w"][:, 0]; vb = tr["lin_vel_b"][:, 0]; ok = slice(25, None)
res = {"act": a.act, "cmd": [a.vx, a.vy, a.wz], "height_cmd": a.height, "lag": a.lag, "arm_fix": not a.no_arm_fix,
       "control_steps": int(S["k"]), "sim_seconds": round(S["k"] * 0.02, 2), "fell": S["fell"],
       "mean_vx_after_0.5s": round(float(vb[ok, 0].mean()), 4) if len(vb) > 25 else None,
       "base_height_mean_after_0.5s": round(float(rz[ok, 2].mean()), 4) if len(rz) > 25 else None,
       "distance_m": round(float(np.linalg.norm(rz[-1, :2] - rz[0, :2])), 3), "physics_steps": S["n"], "render_updates": updates,
       "wall_s": round(time.time() - wall0, 1), "policy_ms_per_query": round(1000 * S["t_policy"] / max(S["k"], 1), 3),
       "joint_props_read_back": S.get("check")}
if TRACK: res["tracked_moved_mm"] = {L: round(float(np.linalg.norm(c1[L] - c0[L]) * 1000), 2) for L in TRACK}
if a.stop_after is not None and "stop_step" in S:
    j = S["stop_step"]; after = vb[j:, :2]; pos = rz[:, :2]
    moving = np.flatnonzero(np.linalg.norm(after, axis=1) > 0.05)
    res["stop"] = {"t_cmd": round(j * 0.02, 2), "speed_at_cmd": round(float(np.linalg.norm(vb[j, :2])), 3),
                   "t_to_below_0.05_mps": round(float((moving[-1] + 1) * 0.02), 2) if len(moving) else 0.0,
                   "distance_after_cmd_m": round(float(np.linalg.norm(pos[-1] - pos[j])), 3),
                   "speed_last_1s_mean": round(float(np.linalg.norm(vb[-50:, :2], axis=1).mean()), 4),
                   "base_xy_end": [round(float(v), 3) for v in pos[-1]]}
json.dump(res, open(os.path.join(a.out, "RESULT.json"), "w"), indent=1)
print("[walk] RESULT " + json.dumps({k: v for k, v in res.items() if k != "joint_props_read_back"}), flush=True)
print("[walk] props fl_hx/fl_kn/arm_f1x:", {d: res["joint_props_read_back"][d] for d in ("fl_hx", "fl_kn", "arm_f1x")} if res["joint_props_read_back"] else None, flush=True)
SimulationManager.deregister_callback(cb) if hasattr(SimulationManager, "deregister_callback") else None
tl.stop()
# Kit sometimes hangs in app.close() after a headless run (seen 2026-09-19: an hour at 330% CPU after RESULT was
# written). Everything is on disk by now, so leave hard instead.
import sys; sys.stdout.flush(); os._exit(0)
