# Spot (crawl-lab leg-collision asset, the robot spot_legcol60 was trained on) in a PhysX stage: on an open floor
# (--open-floor), or added on top of your own scene (--base scene.usda, sublayered; its /World and physicsScene are reused).
# Motor model per docs/LEGCOL60_PORT_CONTRACT.md section 5 (crawl-lab FusedDelayedPDActuator, what TRAINED):
#   --act explicit (default): every drive has stiffness = damping = 0; the runner computes the PD torque itself on every
#       5 ms physics step (knee angle/speed envelope included) and applies it as a joint effort. Drive targets are still
#       authored so the pose reads right in the GUI, but with zero gains they exert nothing.
#   --act implicit: PhysX drives with the training gains (step-1 behaviour). USD angular drives are per DEGREE:
#       stiffness = kp*pi/180, damping = kd*pi/180, targets in degrees.
#   legs kp 60 kd 1.5, hip clamp 45 N m; arm kp (120,120,120,100,100,100,16) kd (2,..,2,0.32), armature 0.01 x6, 0.001 f1x.
#   Joint friction as trained (MuJoCo frictionloss = absolute Coulomb torque): 0.008 N m hx/hy, 0.18 N m kn, as PhysX
#   physxJointAxis:angular:{static,dynamic}FrictionEffort (absolute effort, N m for revolute joints).
# Foot/ground friction: the asset has no physics material (PhysX default 0.5). Training eval ran at an effective mu 1.0
#   (MuJoCo combines a pair with max; robot 0.75, ground 1.0). Here every robot collider gets mu 1.0 with combine "max".
# Physics 200 Hz (training: 0.005 s). Leg self-collision on, filtered exactly as trained (see below).
#   python make_spot_stage.py --open-floor stages/spot_floor.usda
#   python make_spot_stage.py --base my_scene.usda my_scene_with_spot.usda --spot-xy X Y --yaw DEG [--act explicit|implicit]
import os, math, argparse, json
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux, Sdf, Gf
ap = argparse.ArgumentParser(); ap.add_argument("out")
ap.add_argument("--base", default=None, help="scene stage to add Spot to (Z-up, metres, with /World and /World/physicsScene)")
ap.add_argument("--open-floor", action="store_true", help="a 40 m floor with 1 m stripes, Spot at the origin facing +x")
HERE = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--spot-usd", default=os.path.join(HERE, "assets", "spot_legcol", "spot_with_arm", "spot_with_arm.usda"),
                help="referenced by a path relative to the output stage, so the stage opens wherever the repo is mounted")
ap.add_argument("--spot-xy", type=float, nargs=2, default=None); ap.add_argument("--yaw", type=float, default=None)
ap.add_argument("--z", type=float, default=0.65, help="initial base height (SPOT_DEFAULT_POS z)")
ap.add_argument("--act", choices=["explicit", "implicit"], default="explicit")
ap.add_argument("--robot-mu", type=float, default=1.0)
ap.add_argument("--no-cameras", action="store_true", help="skip the 6 Spot cameras (cameras.py)")
a = ap.parse_args(); D2R = math.pi / 180
assert bool(a.open_floor) != bool(a.base), "give exactly one of --open-floor or --base <scene.usda>"
LEG = {"hx": (60.0, 1.5, 45.0), "hy": (60.0, 1.5, 45.0), "kn": (60.0, 1.5, 80.0)}   # implicit only: knee 80 = table plateau
ARM = {"arm_sh0": (120, 2, 90.9), "arm_sh1": (120, 2, 181.8), "arm_el0": (120, 2, 90.9), "arm_el1": (100, 2, 30.3),
       "arm_wr0": (100, 2, 30.3), "arm_wr1": (100, 2, 30.3), "arm_f1x": (16, 0.32, 15.32)}
ARM_ARMATURE = {j: (0.001 if j == "arm_f1x" else 0.01) for j in ARM}                  # crawl-lab actuator_specs.py:30
LEG_FRICTION = {"hx": 0.008, "hy": 0.008, "kn": 0.18}                                  # N m, env.py:437-447
STAND = {"fl_hx": 0.12, "fl_hy": 0.5, "fl_kn": -1.0, "fr_hx": -0.12, "fr_hy": 0.5, "fr_kn": -1.0,
         "hl_hx": 0.12, "hl_hy": 0.5, "hl_kn": -1.0, "hr_hx": -0.12, "hr_hy": 0.5, "hr_kn": -1.0}
STOW = {"arm_sh0": 0.0, "arm_sh1": -3.1415, "arm_el0": 3.1415, "arm_el1": 1.5655, "arm_wr0": 0.0, "arm_wr1": -1.5655, "arm_f1x": 0.0}
if os.path.exists(a.out): os.remove(a.out)
st = Usd.Stage.CreateNew(a.out)
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
if a.base:
    st.GetRootLayer().subLayerPaths.append(os.path.relpath(os.path.abspath(a.base), os.path.dirname(os.path.abspath(a.out))))
    st.SetDefaultPrim(st.GetPrimAtPath("/World"))
    _b = Usd.Stage.Open(os.path.abspath(a.base))                  # use the base scene's own physics scene, wherever it lives
    _scenes = [q.GetPath() for q in _b.Traverse() if q.IsA(UsdPhysics.Scene)]
    sc = st.OverridePrim(_scenes[0] if _scenes else "/World/physicsScene")
    if not _scenes: UsdPhysics.Scene.Define(st, "/World/physicsScene")
    x, y = a.spot_xy if a.spot_xy else (0.0, 0.0); yaw = 0.0 if a.yaw is None else a.yaw
else:
    world = UsdGeom.Xform.Define(st, "/World"); st.SetDefaultPrim(world.GetPrim())
    sc = UsdPhysics.Scene.Define(st, "/World/physicsScene").GetPrim()
    UsdPhysics.Scene(sc).CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); UsdPhysics.Scene(sc).CreateGravityMagnitudeAttr(9.81)
    sc.AddAppliedSchema("PhysxSceneAPI"); sc.CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("TGS")
    gm = UsdShade.Material.Define(st, "/World/looks/floor_phys"); gp = gm.GetPrim(); UsdPhysics.MaterialAPI.Apply(gp)
    UsdPhysics.MaterialAPI(gp).CreateStaticFrictionAttr(1.0); UsdPhysics.MaterialAPI(gp).CreateDynamicFrictionAttr(1.0)
    UsdPhysics.MaterialAPI(gp).CreateRestitutionAttr(0.0)
    def look(path, rgb):
        m = UsdShade.Material.Define(st, path); s = UsdShade.Shader.Define(st, path + "/s"); s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb)); s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface"); return m
    grey, dark = look("/World/looks/floor", (0.42, 0.42, 0.43)), look("/World/looks/stripe", (0.22, 0.22, 0.24))
    g = UsdGeom.Cube.Define(st, "/World/ground"); g.CreateSizeAttr(1.0)
    UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05)); UsdGeom.Xformable(g).AddScaleOp().Set(Gf.Vec3f(40, 40, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim()); b = UsdShade.MaterialBindingAPI.Apply(g.GetPrim()); b.Bind(grey)
    b.Bind(gm, UsdShade.Tokens.weakerThanDescendants, "physics")
    for i in range(-10, 21):                                                      # 1 m stripes across x: speed is readable on video
        s = UsdGeom.Cube.Define(st, f"/World/stripes/x{i + 10:02d}"); s.CreateSizeAttr(1.0)
        UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(i, 0, 0.0005)); UsdGeom.Xformable(s).AddScaleOp().Set(Gf.Vec3f(0.03, 20, 0.001))
        UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(dark)
    dome = UsdLux.DomeLight.Define(st, "/World/dome"); dome.CreateIntensityAttr(1200.0); dome.CreateColorAttr(Gf.Vec3f(0.55, 0.56, 0.58))
    key = UsdLux.DistantLight.Define(st, "/World/key"); key.CreateIntensityAttr(1800.0); key.CreateAngleAttr(1.5)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-52.0, 0.0, 34.0))
    x, y = a.spot_xy if a.spot_xy else (0.0, 0.0); yaw = 0.0 if a.yaw is None else a.yaw
sc.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(200)
spot = UsdGeom.Xform.Define(st, "/World/spot")
spot.GetPrim().GetReferences().AddReference(os.path.relpath(os.path.abspath(a.spot_usd), os.path.dirname(os.path.abspath(a.out))))
xf = UsdGeom.Xformable(spot); xf.AddTranslateOp().Set(Gf.Vec3d(x, y, a.z)); xf.AddRotateZOp().Set(yaw)
body = st.OverridePrim("/World/spot/Geometry/body")
body.AddAppliedSchema("PhysxArticulationAPI")
for n, t, v in (("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool, True),
                ("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int, 8),
                ("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 1)):
    body.CreateAttribute(n, t).Set(v)
# Self-collision exactly as trained: faster's Newton importer filters EVERY robot shape pair, then legcol un-filters
# leg-vs-OTHER-leg pairs (assets/make_selfcol_asset.py). PhysX is the opposite (all pairs collide unless filtered), so
# every pair except "leg link of leg A vs leg link of leg B" is filtered here. Body and collider lists read from the asset.
try:
    _src = Usd.Stage.Open(a.spot_usd)
    ALL = [str(q.GetPath()).replace("/spot_with_arm/", "/World/spot/") for q in _src.Traverse() if q.HasAPI(UsdPhysics.RigidBodyAPI)]
    COLS = [str(q.GetPath()).replace("/spot_with_arm/", "/World/spot/") for q in _src.Traverse() if q.HasAPI(UsdPhysics.CollisionAPI)]
except Exception as e:
    raise SystemExit("cannot read the robot asset for its body list: %s" % e)
def leg_of(path):
    for L in ("fl", "fr", "hl", "hr"):
        if f"/{L}_hip" in path: return L
    return None
nf = 0
for b in ALL:
    others = [o for o in ALL if o != b and not (leg_of(b) and leg_of(o) and leg_of(b) != leg_of(o))]
    pr = st.OverridePrim(b); pr.AddAppliedSchema("PhysicsFilteredPairsAPI")
    pr.CreateRelationship("physics:filteredPairs").SetTargets([Sdf.Path(o) for o in others]); nf += len(others)
print("self-collision: %d bodies, %d filtered pair entries; only leg-vs-other-leg pairs collide" % (len(ALL), nf))
# robot contact material: mu 1.0, combine max (MuJoCo pair rule), no restitution
rm = UsdShade.Material.Define(st, "/World/looks/spot_phys"); rp = rm.GetPrim(); UsdPhysics.MaterialAPI.Apply(rp)
UsdPhysics.MaterialAPI(rp).CreateStaticFrictionAttr(a.robot_mu); UsdPhysics.MaterialAPI(rp).CreateDynamicFrictionAttr(a.robot_mu)
UsdPhysics.MaterialAPI(rp).CreateRestitutionAttr(0.0); rp.AddAppliedSchema("PhysxMaterialAPI")
rp.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
rp.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
for c in COLS:
    pr = st.OverridePrim(c); pr.AddAppliedSchema("MaterialBindingAPI")
    pr.CreateRelationship("material:binding:physics").SetTargets([rp.GetPath()])
print("robot material mu %.2f (combine max) on %d colliders" % (a.robot_mu, len(COLS)))
if not a.no_cameras:                                   # hand + 5 body cameras, as children of their links (cameras.py)
    import sys; sys.path.insert(0, HERE); from cameras import author_spot_cameras
    links = {q.rsplit("/", 1)[-1]: q for q in ALL}
    print("cameras:", ", ".join(c.rsplit("/", 1)[-1] for c in author_spot_cameras(st, links)))
def drive(j, kp, kd, fmax, target, armature=0.0, friction=0.0):
    p = st.OverridePrim(f"/World/spot/Physics/{j}")
    p.AddAppliedSchema("PhysicsDriveAPI:angular"); p.AddAppliedSchema("PhysicsJointStateAPI:angular"); p.AddAppliedSchema("PhysxJointAxisAPI:angular")
    exp = a.act == "explicit"
    p.CreateAttribute("drive:angular:physics:type", Sdf.ValueTypeNames.Token).Set("force")
    p.CreateAttribute("drive:angular:physics:stiffness", Sdf.ValueTypeNames.Float).Set(0.0 if exp else kp * D2R)
    p.CreateAttribute("drive:angular:physics:damping", Sdf.ValueTypeNames.Float).Set(0.0 if exp else kd * D2R)
    p.CreateAttribute("drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float).Set(1e9 if exp else fmax)   # explicit: runner clamps
    p.CreateAttribute("drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float).Set(target / D2R)
    p.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(target / D2R)   # start in the pose
    p.CreateAttribute("state:angular:physics:velocity", Sdf.ValueTypeNames.Float).Set(0.0)
    p.CreateAttribute("physxJointAxis:angular:armature", Sdf.ValueTypeNames.Float).Set(armature)
    p.CreateAttribute("physxJointAxis:angular:staticFrictionEffort", Sdf.ValueTypeNames.Float).Set(friction)
    p.CreateAttribute("physxJointAxis:angular:dynamicFrictionEffort", Sdf.ValueTypeNames.Float).Set(friction)
for j, q in STAND.items(): kp, kd, fm = LEG[j[-2:]]; drive(j, kp, kd, fm, q, 0.0, LEG_FRICTION[j[-2:]])
for j, q in STOW.items(): kp, kd, fm = ARM[j]; drive(j, kp, kd, fm, q, ARM_ARMATURE[j])
st.GetRootLayer().Save()
print("wrote %s | act %s | spot at (%.3f, %.3f, %.2f) yaw %.0f%s" % (a.out, a.act, x, y, a.z, yaw, " | on " + a.base if a.base else " | open floor"))
