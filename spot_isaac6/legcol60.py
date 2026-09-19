# spot_legcol60 (crawl-lab, Newton/MuJoCo + PufferLib MinGRU) as it must be driven in PhysX: the training-side conventions the
# exported ONNX leaves out. Contract with file:line sources: policies/spot_legcol60/PORT_CONTRACT.md (cited as C.<n>).
# numpy + onnxruntime only, so it imports (and can be tested) outside Isaac Sim.
import numpy as np, onnxruntime as ort

# ---- contract constants ------------------------------------------------------------------------------------------------
DEPLOY = ["fl_hx", "fl_hy", "fl_kn", "fr_hx", "fr_hy", "fr_kn", "hl_hx", "hl_hy", "hl_kn", "hr_hx", "hr_hy", "hr_kn",
          "arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"]                       # C.2
REF = np.array([0.12, 0.5, -1.0, -0.12, 0.5, -1.0, 0.12, 0.5, -1.0, -0.12, 0.5, -1.0], np.float32)              # C.4
STOW = np.array([0.0, -3.1415, 3.1415, 1.5655, 0.0, -1.5655, 0.0], np.float32)                                   # C.4
CARRY = np.array([0.0, -0.9, 1.8, 0.0, -0.9, 0.0, -1.54], np.float32)                                            # C.0 trap 1
LO = np.tile(np.array([-0.785398, -0.898845, -2.7929], np.float32), 4); HI = np.tile(np.array([0.785398, 2.295108, -0.2471], np.float32), 4)
KP = np.array([60.0] * 12 + [120, 120, 120, 100, 100, 100, 16], np.float32)                                      # C.5.3
KD = np.array([1.5] * 12 + [2, 2, 2, 2, 2, 2, 0.32], np.float32)
EFF = np.array([45.0, 45.0, 1e30] * 4 + [90.9, 181.8, 90.9, 30.3, 30.3, 30.3, 15.32], np.float32)
KNEE = np.array([2, 5, 8, 11])
KNEE_TABLE = np.array([37.165077, 39.435162, 41.690054, 43.928996, 46.151304, 48.356134, 50.542751, 52.710331, 54.858078,
    56.985128, 59.090595, 61.173609, 63.233231, 65.268557, 67.278557, 69.262310, 71.218735, 73.146824, 75.045502, 76.913641,
    78.750154] + [80.0] * 62 + [78.494694, 76.139643, 73.717049, 71.228605, 68.676006, 66.061146, 63.385900, 60.652325,
    57.862421, 55.018473, 52.122648, 49.177254, 46.184715, 43.147428, 40.067954, 36.948864, 33.792821, 30.602500])   # C.5.6
KNEE_ANG = np.linspace(-2.7929, -0.2471, 101)
assert len(KNEE_TABLE) == 101
def interp_ext(x, xs, ys):                     # crawl-lab actuators_native.py:29-42: clamped index, UNclamped lerp
    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    return ys[i] + (x - xs[i]) * (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
def pd_torque(target, q, qd):                  # C.5.2, per physics substep, deploy order
    tau = KP * (target - q) - KD * qd
    tau = np.minimum(np.maximum(tau, -EFF), EFF)
    qk, qdk, tk = q[KNEE], qd[KNEE], tau[KNEE]
    L = interp_ext(qk, KNEE_ANG, KNEE_TABLE); tk = np.minimum(np.maximum(tk, -L), L)
    hi = interp_ext(qdk, np.array([-30.0, 0.0, 14.0]), np.array([96.9972, 96.9972, 0.0]))
    lo = interp_ext(qdk, np.array([-15.0, 0.0, 30.0]), np.array([0.0, -96.9972, -96.9972]))
    tau[KNEE] = np.minimum(np.maximum(tk, lo), hi)
    return tau
OFFS = np.array([0.0, 0.5, 0.75, 0.25])                                                                          # C.3.4
def foot_heights(phase, cmd, v_xy):
    th = (phase - OFFS) % 1.0; sw = th < 0.25
    fh = 0.036 + 0.14 * np.where(sw, np.sin(np.pi * th / 0.25), 0.0)
    if np.linalg.norm(cmd) < 0.05 and v_xy < 0.05: fh[:] = 0.036
    return fh.astype(np.float32)
def rot(qxyzw):
    x, y, z, w = qxyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

class Legcol60Policy:
    """ONNX MinGRU policy with the training-side conventions the export leaves out (C.0 traps 1-3)."""
    def __init__(self, path, cmd, height, lag=2, arm_fix=True):
        self.lag, self.arm_fix = lag, arm_fix
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.cmd = np.asarray(cmd, np.float32); self.height = np.float32(height)
        self.h = np.zeros((2, 1, 256), np.float32); self.hist = [REF.copy()] * lag    # outputs of past queries
        self.phase = 0.0; self.fh = np.full(4, 0.036, np.float32)                        # k = 0 value (C.3.4)
    def query(self, w_b, g_b, q19, qd19):
        qin = q19.copy()
        if self.arm_fix: qin[12:] = q19[12:] - STOW + CARRY
        f = {"base_angular_velocity": w_b[None].astype(np.float32), "projected_gravity": g_b[None].astype(np.float32),
             "velocity_commands": self.cmd[None], "joint_positions": qin[None].astype(np.float32), "joint_velocities": qd19[None].astype(np.float32),
             "last_actions": self.hist[-self.lag][None], "height_commands": np.array([[self.height]], np.float32),
             "base_orientation_commands": np.zeros((1, 2), np.float32), "foot_height_commands": self.fh[None], "hidden_state": self.h}
        out, self.h = self.s.run(None, f); out = out[0]
        self.hist = (self.hist + [out.copy()])[-max(self.lag, 1):]       # unclamped output goes back as last_actions (C.9)
        return np.clip(out, LO, HI)                                    # soft limits = USD limits (C.4)
    def advance_gait(self, v_xy):                                       # after the 4 substeps (C.3.4 ordering)
        self.phase = (self.phase + 1.1 * 0.02) % 1.0; self.fh = foot_heights(self.phase, self.cmd, v_xy)

