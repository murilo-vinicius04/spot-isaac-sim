# Drive Spot (spot_legcol60) with the keyboard in the Isaac Sim 6.0 GUI. Hold keys to move; release to stop.
#   /isaac-sim/python.sh spot_isaac6/teleop.py [--stage stages/spot_warehouse.usda]
#   keys (Isaac window focused):  W/S or Up/Down  forward/back     A/D or Left/Right  sideways
#                                 Q/E  turn left/right              R  reset Spot to its start pose
# Commands are capped to what the policy trained on (|vx|<=0.55, |vy|<=0.45, |wz|<=0.55; see PORT_CONTRACT.md 3.1).
# The policy only ever saw ONE command per episode, so changing it while walking is outside training: stopping on
# command was tested (it stops within ~0.6 s), turning and strafing were not.
# --headless --script "W:3,Q:2,-:2,R:1" runs a key sequence without a window (key:seconds, "-" = nothing held), for tests.
import argparse, json, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--stage", default=os.path.join(HERE, "stages", "spot_warehouse.usda"))
ap.add_argument("--onnx", default=os.path.join(HERE, "policies", "spot_legcol60", "spot_legcol60.onnx"))
ap.add_argument("--height", type=float, default=0.56)
ap.add_argument("--vx", type=float, default=0.5, help="forward/back speed while the key is held (m/s)")
ap.add_argument("--vy", type=float, default=0.4, help="sideways speed (m/s)")
ap.add_argument("--wz", type=float, default=0.5, help="turn rate (rad/s)")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--script", default=None, help='headless key sequence, e.g. "W:3,Q:2,-:2" (key:seconds)')
ap.add_argument("--out", default=os.path.join(HERE, "runs", "teleop"), help="with --script: where RESULT.json and the log go")
a = ap.parse_args(); a.stage = os.path.abspath(a.stage); a.out = os.path.abspath(a.out)
from isaacsim import SimulationApp
app = SimulationApp({"headless": a.headless, "renderer": "RayTracedLighting", "width": 1600, "height": 900, "multi_gpu": False})
import sys, numpy as np, carb, omni.usd, omni.timeline, warp as wp
from pxr import UsdGeom, UsdPhysics, Gf
from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent
sys.path.insert(0, HERE)
from legcol60 import DEPLOY, REF, STOW, pd_torque, rot, Legcol60Policy

ctx = omni.usd.get_context(); print("[teleop] open_stage ->", ctx.open_stage(a.stage), flush=True)
stage = ctx.get_stage()
for _ in range(20): app.update()
_root = stage.GetPrimAtPath("/World/spot/Geometry/body")
if not (_root.IsValid() and _root.HasAPI(UsdPhysics.ArticulationRootAPI)):
    print("[teleop] ERROR: no Spot articulation at /World/spot/Geometry/body in", a.stage, flush=True); os._exit(2)

# chase camera behind Spot, used as the viewport camera
cam = UsdGeom.Camera.Define(stage, "/World/follow_cam"); cam.CreateFocalLengthAttr(18.0); cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(11.787); cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 200.0)); cam_op = cam.AddTransformOp()
def aim(base, yaw):
    back = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    tgt = np.array([base[0], base[1], 0.35]); eye = tgt - 2.6 * back + np.array([0, 0, 1.1]); f = tgt - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f); M = np.eye(4); M[:3, 0] = r; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    cam_op.Set(Gf.Matrix4d(*M.T.flatten().tolist()))
p0 = np.array(UsdGeom.Xformable(stage.GetPrimAtPath("/World/spot")).ComputeLocalToWorldTransform(0).ExtractTranslation()); aim(p0, 0.0)
if not a.headless:
    for _ in range(10): app.update()
    try:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().camera_path = "/World/follow_cam"
    except Exception as e:
        print("[teleop] could not set the viewport camera:", e, flush=True)

# ---- keyboard: held keys -> velocity command -----------------------------------------------------------------------
KEYS = {"W": (1, 0, 0), "UP": (1, 0, 0), "S": (-1, 0, 0), "DOWN": (-1, 0, 0), "A": (0, 1, 0), "LEFT": (0, 1, 0),
        "D": (0, -1, 0), "RIGHT": (0, -1, 0), "Q": (0, 0, 1), "E": (0, 0, -1)}
held, reset_req, last_cmd = set(), [False], [""]
def command():
    v = np.zeros(3)
    for k in held: v += KEYS[k]
    return (np.clip(v, -1, 1) * [a.vx, a.vy, a.wz]).astype(np.float32)
if not a.headless:
    import omni.appwindow
    def on_key(ev, *args):
        # every key press also sends a CHAR event whose .input is the typed str, so check the type before .name
        if ev.type not in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_RELEASE): return True
        name = ev.input.name
        if ev.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name in KEYS: held.add(name)
            elif name == "R": reset_req[0] = True
        else:
            held.discard(name)
        c = "[teleop] cmd vx %.2f vy %.2f wz %.2f" % tuple(command())
        if c != last_cmd[0]: print(c, flush=True); last_cmd[0] = c
        return True
    _input = carb.input.acquire_input_interface()
    _kb_sub = _input.subscribe_to_keyboard_events(omni.appwindow.get_default_app_window().get_keyboard(), on_key)
    print("[teleop] keys: W/S forward/back, A/D sideways, Q/E turn, R reset (click the Isaac window first)", flush=True)

# ---- controller: runs before every 5 ms physics step (same law as walk.py) ------------------------------------------
pol = Legcol60Policy(a.onnx, (0.0, 0.0, 0.0), a.height)
S = {"art": None, "n": 0, "k": 0, "target": np.concatenate([REF, STOW]).astype(np.float32), "fell": False, "log": []}
def reset_policy():
    pol.h[:] = 0; pol.hist = [REF.copy()] * pol.lag; pol.phase = 0.0; pol.fh = np.full(4, 0.036, np.float32)
def on_pre_step(dt, context):
    if S["art"] is None:
        view = SimulationManager.get_physics_simulation_view()
        if view is None: return
        art = view.create_articulation_view("/World/spot/Geometry/body"); names = list(art.shared_metatype.dof_names)
        links = list(art.shared_metatype.link_names)
        S.update(art=art, dofs=names, d2a=np.array([names.index(j) for j in DEPLOY]),
                 term=np.array([i for i, n in enumerate(links) if "leg" in n or n == "body"]),
                 idx=wp.array([0], dtype=wp.uint32, device=art.get_dof_positions().device),
                 root0=art.get_root_transforms().numpy().copy(), q0=art.get_dof_positions().numpy().copy())
    art, dev = S["art"], S["idx"].device
    if reset_req[0]:                                           # teleport back to the start pose, standing still
        art.set_root_transforms(wp.array(S["root0"], dtype=wp.float32, device=dev), S["idx"])
        art.set_root_velocities(wp.zeros((1, 6), dtype=wp.float32, device=dev), S["idx"])
        art.set_dof_positions(wp.array(S["q0"], dtype=wp.float32, device=dev), S["idx"])
        art.set_dof_velocities(wp.zeros(S["q0"].shape, dtype=wp.float32, device=dev), S["idx"])
        reset_policy(); S["n"] = 0; S["fell"] = False; reset_req[0] = False; print("[teleop] reset", flush=True)
        return
    root = art.get_root_transforms().numpy()[0].copy(); vel = art.get_root_velocities().numpy()[0].copy()
    q = art.get_dof_positions().numpy()[0][S["d2a"]].astype(np.float32); qd = art.get_dof_velocities().numpy()[0][S["d2a"]].astype(np.float32)
    R = rot(root[3:7])
    if S["n"] % 4 == 0:                                        # 50 Hz control step
        if S["n"] > 0:
            pol.advance_gait(float(np.linalg.norm((R.T @ vel[:3])[:2]))); S["k"] += 1
            low = art.get_link_transforms().numpy()[0][S["term"], 2].min()
            fell = bool(-(R.T @ [0, 0, -1.0])[2] < 0.9 or root[2] < 0.3 or low < 0.045)
            if fell and not S["fell"]: print("[teleop] Spot fell - press R to reset", flush=True)
            S["fell"] = fell
            if a.script: S["log"].append([S["k"] * 0.02, *pol.cmd.tolist(), *(R.T @ vel[:3])[:2].tolist(), (R.T @ vel[3:6])[2], root[0], root[1], root[2], float(fell)])
        pol.cmd = command()
        legs = pol.query(R.T @ vel[3:6], R.T @ np.array([0, 0, -1.0]), q, qd)
        S["target"] = np.concatenate([legs, STOW]).astype(np.float32)
    tau = pd_torque(S["target"], q, qd); ta = np.zeros(len(S["dofs"]), np.float32); ta[S["d2a"]] = tau
    art.set_dof_actuation_forces(wp.array(ta[None], dtype=wp.float32, device=dev), S["idx"])
    S["n"] += 1
SimulationManager.setup_simulation(dt=1.0 / 200.0)
cb = SimulationManager.register_callback(on_pre_step, event=SimulationEvent.PHYSICS_PRE_STEP)
tl = omni.timeline.get_timeline_interface(); tl.play()

# headless key script: [(key or None, seconds)], "R" = reset
plan = []
if a.script:
    for item in a.script.split(","):
        k, sec = item.split(":"); plan.append((None if k == "-" else k.upper(), float(sec)))
t_end, cur = 0.0, None
while app.is_running():
    if a.script:
        t = S["k"] * 0.02
        if t >= t_end:
            if cur and cur in KEYS: held.discard(cur)
            if not plan: break
            cur, sec = plan.pop(0); t_end = t + sec
            if cur == "R": reset_req[0] = True
            elif cur in KEYS: held.add(cur)
            print("[teleop] t=%.2f key %s for %.1f s" % (t, cur or "-", sec), flush=True)
    app.update()
    if S["art"] is not None:
        r = S["art"].get_root_transforms().numpy()[0]; x, y, z, w = r[3:7]
        aim(r[:3], np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
if a.script:
    os.makedirs(a.out, exist_ok=True)
    cols = ["t", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "x", "y", "z", "fell"]
    L = np.array(S["log"]); np.save(os.path.join(a.out, "log.npy"), L)
    json.dump({"columns": cols, "script": a.script, "steps": len(L), "falls": int((np.diff(np.r_[0, L[:, -1]]) > 0).sum()) if len(L) else 0},
              open(os.path.join(a.out, "RESULT.json"), "w"), indent=1)
    print("[teleop] wrote", a.out, flush=True)
tl.stop(); sys.stdout.flush(); os._exit(0)
