# spot_legcol60 -> Isaac Sim 6.0 / PhysX port contract

Read-only audit, 2026-09-19. Every claim below is cited to code. Where I measured something myself
I say so. Where I could not verify something I say that too, in section 10.

Abbreviations for paths:

- `CL` = `/home/nexus/spot-teleop/isaac-sim_ws/crawl-lab`
- `FA` = `/home/nexus/spot-teleop/isaac-sim_ws/faster/faster/loco/crawl`
- `IL` = `CL/IsaacLab/source`
- `RUN` = `CL/runs/spot_legcol60`

**Which code is authoritative.** The policy was trained by crawl-lab (`scripts/train.py` ->
`spot_locomanipulation/task.py:17-31` -> `puffer_env.py:74-127` -> `env.py`), not by faster's
`SpotEnv`. The task config the env received is the run's `config.yaml` `task:` block
(`task.py:24`, `task_cfg = cfg.task`). The actuator is crawl-lab's `FusedDelayedPDActuator`
(`actuators_native.py:285-380`), selected by `puffer_env.py:99-104` (`FASTER_ACT` unset) and
`ACT_KIND` defaulting to `"fused"` (`actuators_native.py:194`). `FA/actuators.py` (the Warp kernel)
is **not** what ran in training. It is the reference the fused actuator was built to match, and it
differs in two places (see 5.7). **Where crawl-lab and faster disagree, crawl-lab wins.**

---

## 0. The traps (read these first)

1. **Arm offsets inside the ONNX are wrong for this policy. You have to compensate outside the
   graph.** The graph subtracts `joint_offsets` from `joint_positions`. For the 7 arm joints those
   offsets are the "carry" pose `(0, -0.9, 1.8, 0, -0.9, 0, -1.54)` (`puffer_onnx.py:33-42`, which
   reads `SPOT_DEFAULT_JOINT_POS`). Training, however, subtracted the **stow** pose
   `(0, -3.1415, 3.1415, 1.5655, 0, -1.5655, 0)` (`env.py:97-101`, `mdp/observations.py:43-44`).
   The in-repo adapter hides this because it adds and subtracts the same metadata offsets
   (`puffer_onnx.py:313-321, 345-352`). If you feed true absolute arm angles, the policy sees an
   arm offset of up to 2.24 rad that it never saw in training. **Feed**
   `joint_positions[12:19] = q_arm - stow + carry`. With the arm exactly at stow, that is the
   carry vector itself.
2. **`last_actions` lags by two policy steps, not one.** In `_pre_physics_step`,
   `prev_actions = actions` runs, then `actions = new` (`env.py:670-672`), and the observation
   returns `prev_actions` (`mdp/observations.py:51-52`). faster does the same (`FA/env.py:1006-1007`,
   `832`). So the observation built after applying `a_k` holds `a_{k-1}`. The policy query `k`
   therefore receives `last_actions = output of query k-2`, set to the reference pose for `k < 2`.
   The export's own parity test feeds `expected[0]` back immediately (`puffer_onnx.py:412`). That
   is a lag of one and **does not match training**. `policy.yaml` does not specify the lag.
3. **The standing rule depends on the robot's measured speed, not only on the command.**
   `standing_uses_velocity` defaults to `True` (`mdp/commands.py:41, 170-171`) and is absent from
   both `config.yaml` and `policy.yaml`. Standing requires `||(vx,vy,wz)_cmd|| < 0.05` **and**
   `||v_body,xy|| < 0.05`.
4. **The actuator is explicit PD at 200 Hz, with an angle- and speed-dependent knee envelope and
   no constant knee torque cap.** Every PhysX drive must be zeroed (IsaacLab zeroes the solver
   gains for explicit actuators: `IL/isaaclab_newton/.../articulation.py:4081-4083`). Torque must be
   recomputed on every 5 ms physics step from fresh q/qd.
5. **Joint friction is MuJoCo `frictionloss`, in N·m** (0.008 hx/hy, 0.18 kn). It is not PhysX's
   unitless `jointFriction` coefficient (`env.py:437-447`).
6. **Leg self-collision relies on a Newton-only attribute.** The asset sets
   `newton:selfCollisionEnabled = 1` and authors `physics:filteredPairs` (trunk vs all 16 leg
   links) **without applying `PhysicsFilteredPairsAPI`**. PhysX will probably ignore both unless
   you re-author them (section 7.4).

---

## 1. ONNX interface

File: `RUN/exported/spot_legcol60.onnx`. I loaded it myself with the system `python3`
(onnx 1.21.0, onnxruntime 1.23.2). Details: opset 18, IR 8, producer pytorch 2.11.0. The
`IsaacLab/.venv` python is a dangling symlink, and the 3dgrut and petrobras-chains venvs have no
onnx.

### 1.1 Graph I/O (printed from the file)

| name | dtype | shape |
|---|---|---|
| `base_angular_velocity` | float32 | [B, 3] |
| `projected_gravity` | float32 | [B, 3] |
| `velocity_commands` | float32 | [B, 3] |
| `joint_positions` | float32 | [B, 19] **absolute rad, but see trap 1 for the arm** |
| `joint_velocities` | float32 | [B, 19] |
| `last_actions` | float32 | [B, 12] **absolute joint targets, rad** |
| `height_commands` | float32 | [B, 1] |
| `base_orientation_commands` | float32 | [B, 2] |
| `foot_height_commands` | float32 | [B, 4] |
| `hidden_state` | float32 | [2, B, 256] |
| **out** `actions_output` | float32 | [B, 12] **absolute leg joint targets, rad** |
| **out** `next_hidden_state` | float32 | [2, B, 256] |

Input and output names come from `puffer_onnx.py:125-135` and `86-88`. The dynamic axes are at
`489-496`.

### 1.2 What the graph does

Source: `puffer_onnx.py:90-122`. I confirmed it by listing the graph's Sub/Div/Concat/Gemm nodes.

```
obs66 = concat[ base_angular_velocity, projected_gravity, velocity_commands,
                joint_positions - joint_offsets,                 # 19
                joint_velocities,                                # 19
                (last_actions - joint_offsets[:12]) / 0.2,       # 12
                height_commands, base_orientation_commands, foot_height_commands ]
h      = Linear(66->256)(obs66)                                  # encoder, no activation, no normaliser
h, h'  = MinGRU 2 layers x 256 (h_prev = hidden_state)
raw    = Linear(256->12)(h)                                      # Gaussian mean
actions_output = raw * 0.2 + joint_offsets[:12]
```

- Initializer `joint_offsets` (19) is
  `[0.12,0.5,-1.0,-0.12,0.5,-1.0,0.12,0.5,-1.0,-0.12,0.5,-1.0, 0.0,-0.9,1.8,0.0,-0.9,0.0,-1.54]`.
  `action_scales` is `0.2 x 12`. The metadata `joint_reference_positions` holds the same values.
- **No observation normalisation is in the graph, and none is needed outside it.** Config has
  `lib.obs_norm: false`. `PufferActor` builds the encoder without a norm
  (`common/policies.py:33-35, 171`). The only arithmetic before the encoder is the two offset
  subtractions and the `/0.2`.
- **No output clipping in the graph.** Training clamps the target to the joint soft limits
  outside the policy (section 4).
- Recurrent state: 2 layers x 256, float32. Feed `next_hidden_state` back as `hidden_state` on the
  next query. Reset it to **zeros** at episode start (metadata `hidden_state_reset=zeros`,
  `policy.yaml recurrent_state.reset: zeros`). In eval the reset is `masked_fill(done, 0)`
  (`policies.py:198-202`). MinGRU cell for reference (Warp twin, `puffer_onnx.py:154-180`):
  `cand = x>=0 ? x+0.5 : sigmoid(x)`, `g = sigmoid(gate)`, `h' = h + g*(cand-h)`,
  `out = sigmoid(proj)*h' + (1-sigmoid(proj))*x`.
- Export parity against torch: `max_parity_error = 1.907e-06` (metadata).
- Deterministic deployment uses the Gaussian **mean** (`policies.py:196`). The graph outputs the
  mean.

---

## 2. The 66-d observation

The order is fixed by `crawl_env_cfg.py:254-272` (`PolicyObsCfg`) and `mdp/observations.py:31-68`.
`robot.py:112-122` (`mirror_obs`) cross-checks the slice indices. All values are float32 SI units:
rad, rad/s, m, and gravity as a unit vector.

| idx | dim | term | definition (training) | source |
|---|---|---|---|---|
| 0-2 | 3 | base angular velocity | `root_com_ang_vel_w` rotated into the **trunk (`body`) link frame** (`quat_apply_inverse(root_link_quat_w, ·)`), rad/s | `observations.py:31-32`; `IL/isaaclab/.../base_articulation_data.py` `root_ang_vel_b -> root_com_ang_vel_b`; newton data `articulation_data.py:1283-1301` |
| 3-5 | 3 | projected gravity | world gravity **normalised to unit length**, rotated into the trunk link frame. Upright = `(0,0,-1)` | `observations.py:35-36`; `IL/isaaclab_newton/.../assets/kernels.py:460-474` |
| 6-8 | 3 | velocity command | `(vx, vy, wz)` in the body frame: m/s, m/s, rad/s | `observations.py:39-40`; `mdp/commands.py:86` |
| 9-27 | 19 | joint positions, relative | `q - default`. Legs: default = `(±0.12, 0.5, -1.0)`. **Arm: default = STOW** (`env.py:97-101`) | `observations.py:43-44` |
| 28-46 | 19 | joint velocities | `qd`, absolute, rad/s | `observations.py:47-48` |
| 47-58 | 12 | last action | **raw** policy output (`prev_actions`, see trap 2); 0 at episode start | `observations.py:51-52`; `env.py:670-672, 945-946` |
| 59 | 1 | height command | m | `observations.py:55-56` |
| 60-61 | 2 | orientation command | (roll, pitch) rad | `observations.py:59-60` |
| 62-65 | 4 | foot height commands | FL, FR, HL, HR, m | `observations.py:63-64`; `commands.py:159-176` |

**Joint order: 19 joints, legs then arm. The arm joints are included.** This is
`DEPLOY_JOINT_NAMES` (`robot.py:26-41`) and matches the ONNX metadata `joint_names`:

```
0 fl_hx  1 fl_hy  2 fl_kn   3 fr_hx  4 fr_hy  5 fr_kn
6 hl_hx  7 hl_hy  8 hl_kn   9 hr_hx 10 hr_hy 11 hr_kn
12 arm_sh0 13 arm_sh1 14 arm_el0 15 arm_el1 16 arm_wr0 17 arm_wr1 18 arm_f1x
```

The observation is gathered through the permutation `j_d2l` (`env.py:89-96`). **Your PhysX
articulation will not be in this order.** Newton's DFS import of this USD is arm-first; I checked
by importing it with Newton (section 7.1). PhysX's DOF order is its own. Build the name map
explicitly.

**Mapping to the ONNX inputs**, with `stow = (0,-3.1415,3.1415,1.5655,0,-1.5655,0)` and
`carry = (0,-0.9,1.8,0,-0.9,0,-1.54)`:

- `joint_positions[0:12] = q_legs` (absolute, rad; the graph subtracts the leg reference, which
  equals the training default).
- `joint_positions[12:19] = q_arm - stow + carry` (trap 1).
- `joint_velocities = qd` (19, same order).
- `last_actions = 0.2 * raw_{k-2} + ref_legs`, which is just `actions_output` from **two queries
  ago**. For queries 0 and 1 use `ref_legs = (0.12,0.5,-1.0,-0.12,0.5,-1.0,0.12,0.5,-1.0,-0.12,0.5,-1.0)`.
  This is `policy.yaml last_actions_reset: reference_joint_positions`, i.e. raw 0.

**Noise: OFF at deployment.** In training (`observation_noise: true`) the noise was uniform ±1
times a per-element scale times a per-env gain drawn from U(0.8, 1.2). Scales: ang vel 0.2,
gravity 0.05, joint pos 0.05, joint vel 0.5; commands and actions got none (`env.py:259-265,
819-827`). Inference turns it off via `play_mode` (`env_cfg.py:113-116`, `puffer_env.py:94-95`).
No clipping is applied to any observation term.

**When the observation is sampled.** It is built after the 4 physics substeps of the control
step, after the command/gait update, and after resets (`IL/isaaclab/.../envs/direct_rl_env.py:408`
pre-physics, `425-440` decimation loop, `445-447` dones and rewards (commands step inside
`_get_rewards`), `491` observations).

---

## 3. Commands and gait clock

Source: `CL/spot_locomanipulation/tasks/crawl/mdp/commands.py` (`CrawlCommands`), driven by
`commands_native.py`. The config.yaml values override `_DEFAULTS` (`commands.py:25-46, 60`).

### 3.1 Velocity command (obs 6-8)

- Layout `(vx, vy, wz)` in the body frame (`commands.py:86`).
- One draw per episode, held constant for the whole episode. Mid-episode resampling is disabled
  by pinning `resampling_time_range = 1e6` (`commands_native.py:99-102, 139`; `commands.py:125-149`).
  **The policy has never seen a command change mid-episode.**
- Training ranges come from the LP bins in config.yaml `curriculum.lp_acrl.bins`, with a random
  sign where marked. The envelope is |vx| ≤ 0.55 m/s, |vy| ≤ 0.45 m/s, |wz| ≤ 0.55 rad/s. The
  `stance` bin is exactly (0,0,0). Combined bins exist (forward_lateral, forward_turn,
  lateral_turn, ...). The bins never combine vx and vy with wz all three nonzero at once.
- The reference eval used a fixed (0.5, 0, 0) (`scripts/eval.py:392`).

### 3.2 Height command (obs 59)

- Training: U(0.52, 0.60) per episode (config `commands.height_range`, `commands.py:147`).
- Deployment value 0.58 (config `deployment_height_command`, `policy.yaml`).
- Note: `set_fixed()` without a height uses the **mean of the range, 0.56**
  (`commands.py:107`). The reference trace therefore ran at 0.56, not 0.58. Both are in
  distribution. Measured base z in that trace was 0.5385 m (mean over steps 100-499).

### 3.3 Orientation command (obs 60-61)

- Config sets roll and pitch to `[0.0, 0.0]`, so the policy only ever saw **(0, 0)**
  (`commands.py:142-145`). `policy.yaml deployment_orientation_command = [0,0]`. Feed zeros.
  `docs/POSTURE_AUTHORITY.md:11` confirms this checkpoint has no attitude authority.

### 3.4 Foot-height commands and the gait clock (obs 62-65)

Parameters from config.yaml: pattern `static_crawl`, `gait_frequency 1.1` Hz,
`swing_fraction 0.25`, `phase_offsets [0.0, 0.5, 0.75, 0.25]` (order FL, FR, HL, HR,
`commands.py:84`), `foot_height_max 0.14`, `foot_radius 0.036`,
`standing_velocity_threshold 0.05`. `kappa 0.02` only shapes the reward's contact probability,
not the observation. The pattern string only gates validation (`commands.py:68-69`); the math is
the same for any pattern.

Per control step (`commands.py:178-180, 159-176`):

```
phase  = (phase + 1.1 * 0.02) % 1.0                  # +0.022 per 20 ms step, float32
theta_i = (phase - offset_i) % 1.0                   # offsets FL 0.0, FR 0.5, HL 0.75, HR 0.25
swing_i = theta_i < 0.25
fh_i    = 0.036 + 0.14 * (sin(pi * theta_i / 0.25) if swing_i else 0)
standing = (||(vx,vy,wz)_cmd||_2 < 0.05) and (||v_body_xy|| < 0.05)   # standing_uses_velocity=True
if standing: fh_i = 0.036 for all four        # the phase still advances while standing
```

- `||(vx,vy,wz)||` is the 3-vector norm, mixing m/s and rad/s (`commands.py:166-169`).
- `v_body_xy` is `root_lin_vel_b[:, :2]`: the **COM** linear velocity expressed in the trunk frame
  (`env.py:854-856`; `root_lin_vel_b -> root_com_lin_vel_b`). It is sampled post-physics, just
  before the gait update (`env.py:854-859`).
- Disturbance masking (`commands_native.py:121-123`) is inactive at deployment.
- **Ordering within a step:** physics substeps, then body speed read, then phase += 0.022, then
  fh recomputed, then observation. The observation after step k therefore uses the phase that was
  just advanced.
- **Initial phase:** training uses U[0,1) per episode (`commands.py:149`). Eval with
  `set_fixed()` skips resampling (`commands.py:126`), so the phase starts at 0.0.
- **First observation of an episode:** `resample()` does not call `_update_gait()`, so
  observation 0 carries the foot heights from the last update. In eval that is the construction
  value: phase 0 and a zero command, i.e. standing, so `[0.036]*4`. From observation 1 on, the
  values follow the formula. For a clean deployment start, `[0.036]*4` at k=0 followed by the
  formula reproduces eval exactly.
- The swing peak is 0.176 m (0.036 + 0.14) at `theta = 0.125`. Each foot swings for 25% of a
  0.909 s cycle.

---

## 4. Action -> joint target

- **The 12 outputs drive the legs only**, in the order
  `fl_hx fl_hy fl_kn fr_hx fr_hy fr_kn hl_hx hl_hy hl_kn hr_hx hr_hy hr_kn` (`policy.yaml`,
  `actions_native.py:44-51` with `preserve_order=True`).
- Training: `target = default + 0.2 * raw`, then **clamp to the soft joint limits**
  (`actions_native.py:44-51`; `IL/isaaclab/.../mdp/actions/joint_actions.py:170-179, 195-196`).
  `soft_joint_pos_limit_factor = 1.0` (`IL/isaaclab/.../articulation_cfg.py:62`), so the soft
  limits equal the USD limits:
  - hx: [-0.785398, 0.785398]
  - hy: [-0.898845, 2.295108]
  - kn: [-2.792900, -0.247100] rad (converted from the USD degrees, `Physics.usda:29-66`)
- `absolute_joint_positions` means `actions_output` **already is** `0.2*raw + reference`
  (`puffer_onnx.py:122`). The runner must only **clamp it to the limits above**, which the graph
  does not do.
- `reference_joint_positions = (0.12,0.5,-1.0, -0.12,0.5,-1.0, 0.12,0.5,-1.0, -0.12,0.5,-1.0)`.
  These equal config `posture` (hx 0.12 with sign +,-,+,- for FL,FR,HL,HR; hy 0.5; kn -1.0) and
  `SPOT_DEFAULT_JOINT_POS` (`robot.py:50-57`). crawl-lab takes the leg default from
  `init_state.joint_pos` (`env_cfg.py:76-79`), not from `posture`. Here they are equal, so there
  is no conflict.
- Scale 0.2 rad per unit on every leg joint (config `action_scales`). The raw action is not
  clipped anywhere; it is nominally in [-1, 1] but unbounded (`puffer_onnx.py:454-455`).
- Update rate: the target is computed once per control step (`env.py:673-678`) and re-written
  unchanged on each of the 4 substeps (`env.py:705-713`).
- `last_actions` reset value: raw 0, i.e. the reference pose in absolute units (`env.py:945-946`).

**Arm during this policy.** It holds the stow target
`(0, -3.1415, 3.1415, 1.5655, 0, -1.5655, 0)` (`mdp/events.py:27-30`, `env.py:98-105, 711-713`),
driven by the **same explicit PD actuator**: kp `(120,120,120,100,100,100,16)`, kd
`(2,2,2,2,2,2,0.32)`, effort limits `(90.9,181.8,90.9,30.3,30.3,30.3,15.32)` N·m, armature
`(0.01 x6, 0.001)` (`actuator_specs.py:21-30`; `actuators_native.py:206-213`).
`arm_stiffness_scale` and `arm_damping_scale` are 1 in the config. During training the arm also
tracked reachability poses (config `arm_disturbance.enabled: true`, `env.py:188-193`,
`arm_disturbance.py`), so a moving arm is in distribution. The parked arm (`pose_id -1` = stow)
is the eval case.

Measured in the trace: sh1 sags to -3.175, which is 0.034 rad past its -3.14159 limit (MuJoCo
soft limit), and f1x sags to about -0.05. PhysX hard limits will park the arm a few hundredths of
a radian differently.

---

## 5. Actuator model (as trained)

Implementation: `actuators_native.py:285-380` (`FusedDelayedPDActuator`, one group over all 19
joints), built by `actuators_native.py:194-216`. IsaacLab's own implicit actuators in
`env_cfg.py:38-51` (stiffness 0 and damping 0) are **replaced** by this group
(`puffer_env.py:99-104`). Being an explicit `ActuatorBase` (`is_implicit_model=False`,
`IL/isaaclab/.../actuator_base.py:40`), it makes IsaacLab write **stiffness = 0 and damping = 0 to
the solver** for all 19 joints (`IL/isaaclab_newton/.../articulation.py:4081-4083`) and apply the
computed torque as a joint effort (`articulation.py:4146-4193`). **The solver-side (implicit)
gains are zero.**

### 5.1 Rate

The torque is computed **every physics substep, 200 Hz**, from the q/qd of that substep. Newton
does not take the "handles decimation" fast path because `use_newton_actuators` is unset
(`IL/isaaclab_newton/.../articulation.py:3846`; `newton_manager.py:3106-3117`). The standard loop
runs `_apply_action`, then `write_data_to_sim` (which calls `_apply_actuator_model` ->
`compute`), then `sim.step`, then `scene.update(physics_dt)`, 4 times per control step
(`direct_rl_env.py:425-440`). The target is constant within the control step.

### 5.2 Per-joint torque, per substep (`actuators_native.py:360-380`)

```
delayed = target                 # delay_steps hip/knee/arm = 0 and jitter 0 at inference -> no delay
tau = kp * (delayed - q) - kd * qd          # no velocity target, no feed-forward
tau = clamp(tau, -effort_limit, +effort_limit)
if joint is a knee:
    L   = interp_ext(q,  knee_angle, knee_tmax)          # angle limit (table below)
    tau = clamp(tau, -L, +L)
    hi  = interp_ext(qd, [-30, 0, 14],  [96.9972, 96.9972, 0])
    lo  = interp_ext(qd, [-15, 0, 30],  [0, -96.9972, -96.9972])
    tau = clamp(tau, lo, hi)
```

- Use `clamp(x, lo, hi) = min(max(x, lo), hi)`. That is torch's semantics even when lo > hi, in
  which case the result is `hi`.
- `interp_ext` is **not** `np.interp`. The index lookup is clamped to the table, but the lerp is
  not, so the function **extrapolates linearly past the ends along the end segment**
  (`actuators_native.py:29-42`, identical to `FA/actuators.py:103-113`). Consequences:
  - Speed upper bound: 96.9972 for qd ≤ 0 (flat below -30 as well); `96.9972*(1 - qd/14)` for
    qd > 0, which **goes negative** above 14 rad/s.
  - Speed lower bound: -96.9972 for qd ≥ 0; `-96.9972*(1 + qd/15)` for qd < 0, which **goes
    positive** below -15 rad/s.
  - Angle limit: past q = -0.2471 it extrapolates with slope -125.3 N·m/rad; below -2.7929 with
    slope +89.2 N·m/rad.
- Knees are only clamped by effort at `BIG = 1e30` (`actuator_specs.py:97`,
  `actuators_native.py:201-202`). **There is no 115 N·m or 80 N·m constant cap on the knee.** The
  USD's `urdf:limit:effort = 115` is not used.

### 5.3 Gains and limits for this run (config.yaml wins; the code defaults are identical)

| group | joints | kp | kd | effort clamp (N·m) | armature | joint friction (N·m) |
|---|---|---|---|---|---|---|
| hip | *_hx, *_hy (8) | 60.0 | 1.5 | ±45.0 (`actuator_specs.py:18`) | 0.0 | 0.008 |
| knee | *_kn (4) | 60.0 | 1.5 | ±1e30, then the angle and speed envelope | 0.0 | 0.18 |
| arm | sh0, sh1, el0, el1, wr0, wr1, f1x | 120, 120, 120, 100, 100, 100, 16 | 2, 2, 2, 2, 2, 2, 0.32 | 90.9, 181.8, 90.9, 30.3, 30.3, 30.3, 15.32 | 0.01 x6, 0.001 | 0 (USD authors none) |

- Leg gains come from `leg_gains()` with `gain_mode: explicit` (`actuator_specs.py:125-140`).
  `natural_frequency` is unused.
- Leg armature is 0.0 from config. It is not written anywhere for the legs, so it stays at the
  USD value, which is none (0).
- Arm armature is set in the actuator cfg (`actuators_native.py:213`). **crawl-lab uses 0.001 for
  f1x** (`actuator_specs.py:30`); faster uses 0.01 (`FA/actuators.py:19`). crawl-lab wins.
- Effort limit written to the solver: `effort_limit_sim = 1e9` for explicit actuators
  (`actuator_base.py:105, 173-174`). The sim does no clamping of its own.

### 5.4 Joint friction: how it is applied

The friction is not in the torque law. `env.py:437-447` writes 0.008 / 0.008 / 0.18 into
`model.joint_friction` for the 12 leg DOFs. Then `notify_model_changed(JOINT_DOF_PROPERTIES)`
pushes it to MuJoCo as **`dof_frictionloss`** (`env.py:448-459`), which is the same mechanism as
faster (`FA/env.py:209-216`). MuJoCo frictionloss is a soft-constraint dry (Coulomb) friction: it
resists motion up to a magnitude of `frictionloss` N·m, acting as both static and dynamic friction.
The arm's friction is whatever the USD authored, which is nothing, so 0.

**PhysX equivalent:** use an absolute-effort friction, not a coefficient. See section 10 for the
unverified PhysX API question.

### 5.5 Delay and randomisation

`delay_steps` is `{hip 0, knee 0, arm 0}`. The ring buffer holds M+1 = 4 physics-step snapshots
(`actuator_specs.py:33`). During training, `disturb=True` added a per-(env, group) jitter
U{0..3} **physics** steps and kp/kd scales U(0.995, 1.005) / U(0.99, 1.01) per joint per episode
(`actuators_native.py:153-156, 338-358`). At inference (`disturb=False`) the delay is 0 and the
gains are nominal. **Deploy with no delay and nominal gains.**

### 5.6 Knee angle -> max |torque| table (`actuator_specs.py:36-88`, identical to `FA/actuators.py:26-78`, verified by diff)

Columns: `[angle_rad, transmission_ratio (unused), max_torque_Nm]`. The grid is uniform with a
step of 0.025458 rad, 101 rows. The torque is flat at 80 for q in [-2.258282, -0.705344]. At the
nominal knee angle (-1.0) the limit is 80 N·m. At -0.5 it is 60.47; at -0.3 it is 37.19.

```
[-2.792900, -24.776718, 37.165077], [-2.767442, -26.290108, 39.435162],
[-2.741984, -27.793369, 41.690054], [-2.716526, -29.285997, 43.928996],
[-2.691068, -30.767536, 46.151304], [-2.665610, -32.237423, 48.356134],
[-2.640152, -33.695168, 50.542751], [-2.614694, -35.140221, 52.710331],
[-2.589236, -36.572052, 54.858078], [-2.563778, -37.990086, 56.985128],
[-2.538320, -39.393730, 59.090595], [-2.512862, -40.782406, 61.173609],
[-2.487404, -42.155487, 63.233231], [-2.461946, -43.512371, 65.268557],
[-2.436488, -44.852371, 67.278557], [-2.411030, -46.174873, 69.262310],
[-2.385572, -47.479156, 71.218735], [-2.360114, -48.764549, 73.146824],
[-2.334656, -50.030334, 75.045502], [-2.309198, -51.275761, 76.913641],
[-2.283740, -52.500103, 78.750154], [-2.258282, -53.702587, 80.000000],
[-2.232824, -54.882442, 80.000000], [-2.207366, -56.038860, 80.000000],
[-2.181908, -57.171028, 80.000000], [-2.156450, -58.278133, 80.000000],
[-2.130992, -59.359314, 80.000000], [-2.105534, -60.413738, 80.000000],
[-2.080076, -61.440529, 80.000000], [-2.054618, -62.438812, 80.000000],
[-2.029160, -63.407692, 80.000000], [-2.003702, -64.346268, 80.000000],
[-1.978244, -65.253670, 80.000000], [-1.952786, -66.128944, 80.000000],
[-1.927328, -66.971176, 80.000000], [-1.901870, -67.779457, 80.000000],
[-1.876412, -68.552864, 80.000000], [-1.850954, -69.290451, 80.000000],
[-1.825496, -69.991325, 80.000000], [-1.800038, -70.654541, 80.000000],
[-1.774580, -71.279190, 80.000000], [-1.749122, -71.864319, 80.000000],
[-1.723664, -72.409088, 80.000000], [-1.698206, -72.912567, 80.000000],
[-1.672748, -73.373871, 80.000000], [-1.647290, -73.792130, 80.000000],
[-1.621832, -74.166512, 80.000000], [-1.596374, -74.496147, 80.000000],
[-1.570916, -74.780251, 80.000000], [-1.545458, -75.017998, 80.000000],
[-1.520000, -75.208656, 80.000000], [-1.494542, -75.351448, 80.000000],
[-1.469084, -75.445686, 80.000000], [-1.443626, -75.490677, 80.000000],
[-1.418168, -75.485771, 80.000000], [-1.392710, -75.430344, 80.000000],
[-1.367252, -75.323830, 80.000000], [-1.341794, -75.165688, 80.000000],
[-1.316336, -74.955406, 80.000000], [-1.290878, -74.692551, 80.000000],
[-1.265420, -74.376694, 80.000000], [-1.239962, -74.007477, 80.000000],
[-1.214504, -73.584579, 80.000000], [-1.189046, -73.107742, 80.000000],
[-1.163588, -72.576752, 80.000000], [-1.138130, -71.991455, 80.000000],
[-1.112672, -71.351707, 80.000000], [-1.087214, -70.657486, 80.000000],
[-1.061756, -69.908813, 80.000000], [-1.036298, -69.105721, 80.000000],
[-1.010840, -68.248337, 80.000000], [-0.985382, -67.336861, 80.000000],
[-0.959924, -66.371513, 80.000000], [-0.934466, -65.352615, 80.000000],
[-0.909008, -64.280533, 80.000000], [-0.883550, -63.155693, 80.000000],
[-0.858092, -61.978588, 80.000000], [-0.832634, -60.749775, 80.000000],
[-0.807176, -59.469845, 80.000000], [-0.781718, -58.139503, 80.000000],
[-0.756260, -56.759487, 80.000000], [-0.730802, -55.330616, 80.000000],
[-0.705344, -53.853729, 80.000000], [-0.679886, -52.329796, 78.494694],
[-0.654428, -50.759762, 76.139643], [-0.628970, -49.144699, 73.717049],
[-0.603512, -47.485737, 71.228605], [-0.578054, -45.784004, 68.676006],
[-0.552596, -44.040764, 66.061146], [-0.527138, -42.257267, 63.385900],
[-0.501680, -40.434883, 60.652325], [-0.476222, -38.574947, 57.862421],
[-0.450764, -36.678982, 55.018473], [-0.425306, -34.748432, 52.122648],
[-0.399848, -32.784836, 49.177254], [-0.374390, -30.789810, 46.184715],
[-0.348932, -28.764952, 43.147428], [-0.323474, -26.711969, 40.067954],
[-0.298016, -24.632576, 36.948864], [-0.272558, -22.528547, 33.792821],
[-0.247100, -20.401667, 30.602500],
```

Torque-speed tables (`actuator_specs.py:91-94`):
`POS_TORQUE_SPEED_LIMIT = [[-30, 96.9972], [0, 96.9972], [14, 0]]` and
`NEG_TORQUE_SPEED_LIMIT = [[-15, 0], [0, -96.9972], [30, -96.9972]]`, with velocity in rad/s.

### 5.7 Where `FA/actuators.py` (the Warp kernel) differs from what trained

| item | FA kernel | crawl-lab fused (trained) |
|---|---|---|
| arm no-load speed derate | yes: `ARM_VELOCITY_LIMIT = (2.5,2.5,2.5,3,3,3,3)` rad/s, drive `e*clamp(1-qd/v,0,1)`, brake `e*clamp(1+qd/v,0,1)` (`FA/actuators.py:21, 214-216`) | **none** (`actuators_native.py:367-375`) |
| f1x armature | 0.01 | **0.001** |
| ring buffer at reset | seeded with the target | zeros (irrelevant at delay 0) |

Hips, knees, table math, and effort limits are otherwise the same. `tools/actuator_equiv.py`
asserts that the leg torques are bit-exact.

**Torque telemetry does not exist.** `env._last_torque` is initialised to zeros and never written
(`env.py:165`; grep confirms no writer). The trace's `torque` array is all zeros (I measured
`absmax = 0.0`).

---

## 6. Timing

| quantity | value | source |
|---|---|---|
| control dt | 0.02 s (50 Hz) | `robot.py:72`; `policy.yaml control.period_s` |
| physics dt | 0.005 s (200 Hz) | `env_cfg.py:154` (`STEP_DT / SUBSTEPS`) |
| decimation | 4 | `env_cfg.py:90`; `robot.py:75`; config `substeps: 4` agrees |
| Newton substeps per physics step | 1 | `env_cfg.py:190` |
| actuator update | every physics step | section 5.1 |
| gait phase increment | 0.022 per control step | section 3.4 |
| episode length | U(10, 20) s in training | config `episode_length_s` |

The solver settings, for context only since you are replacing MuJoCo-Warp: `solver="newton"`,
`cone="pyramidal"`, `integrator="implicitfast"`, `iterations=20`, `ls_iterations=50`,
`njmax=320`, `nconmax=100`, and `use_mujoco_contacts=False` with Newton's collision pipeline
(`env_cfg.py:157-181`), shape margin 0 and gap None (`env_cfg.py:195`). crawl-lab hardcodes
these; config.yaml's `substeps`, `solver_iterations` and `solver_njmax` have the same values and
are not read by this env.

---

## 7. Robot asset

USD: `CL/assets/usd/spot_legcol/spot_with_arm/spot_with_arm.usda`, with payload `Payload/Physics.usda`,
`Geometry.usda` and others. It is selected by `SPOT_USD_DIR=spot_legcol` (`env_cfg.py:31-33`) and
built by `CL/assets/make_selfcol_asset.py` from `assets/usd/spot`. The asset was produced by
urdf-usd-converter v0.3.3 with `fix_root_link=False` (`env_cfg.py:62-65`). Total mass is 39.07 kg.
I summed it from the USD MassAPI with pxr.

### 7.1 Joints

- 19 revolute and 5 fixed (`*_ank` x4, `arm_jaw`). Axes and limits are in `Physics.usda:17-386`,
  with limits authored in degrees.
- Axes: hx=X, hy=Y, kn=Y; sh0=Z, sh1=Y, el0=Y, el1=X, wr0=Y, wr1=X, f1x=Y.
- Limits in rad:

  | joint | limits |
  |---|---|
  | hx | ±0.785398 |
  | hy | [-0.898845, 2.295108] |
  | kn | [-2.7929, -0.2471] |
  | sh0 | [-2.61799, 3.14159] |
  | sh1 | [-3.14159, 0.523599] |
  | el0 | [0, 3.14159] |
  | el1 | ±2.79253 |
  | wr0 | ±1.8326 |
  | wr1 | [-2.87989, 2.87979] |
  | f1x | [-1.5708, 0] |

- `newton:velocityLimit = 5729.578` deg/s (100 rad/s) on every joint.
- There is no PhysicsDriveAPI, no joint friction and no armature in the USD.
- The USD's `Physics` scope lists legs first. **Newton's DFS import order is arm-first.** I ran
  Newton's `ModelBuilder.add_usd` on this file:
  `['joint_1'(free), arm_sh0..arm_f1x, arm_jaw, fl_hx, fl_hy, fl_kn, fl_ank, fr_..., hl_..., hr_...]`.
- IsaacLab's public order is `DEPLOY_JOINT_NAMES`, because `ArticulationCfg.joint_ordering` is
  honoured in this IsaacLab checkout: `articulation.py:3732` calls
  `_resolve_and_install_ordering_maps`, added upstream 2026-07-15, checkout 2026-08-19. The
  comment at `env.py:85-88` calling it "dead" is stale.
- Joint-limit softness in training: `limit_ke = 1e3`, `limit_kd = 1e1` (`env.py:428-436`). These
  are MuJoCo soft limits, and the arm really does exceed its limits by ~0.03 rad at stow. PhysX
  limits are hard.

### 7.2 Initial state

- `SPOT_DEFAULT_POS = (0, 0, 0.65)` (`robot.py:23`); config `posture.spawn_z` is also 0.65.
- Identity orientation, zero velocities. The nominal reset is `SpawnSampler` with `noise=0` in
  eval (`env.py:107-113`; `mdp/events.py:277-354, 535-561`).
- Joint pos (19, deploy order): legs `(0.12,0.5,-1.0, -0.12,0.5,-1.0, 0.12,0.5,-1.0, -0.12,0.5,-1.0)`;
  **arm at STOW** `(0,-3.1415,3.1415,1.5655,0,-1.5655,0)` (`mdp/events.py:346-354`).
- `SPOT_DEFAULT_JOINT_POS` (`robot.py:50-57`) has the same legs but the carry arm
  `(0,-0.9,1.8,0,-0.9,0,-1.54)`. That value is only the `init_state` placeholder, which reset
  immediately overwrites with stow, plus the ONNX arm offsets (trap 1).
- Training spawn (disturb=True): 20% of resets are nominal. The rest get xy ±0.5, z +[0, 0.10],
  roll/pitch ±0.3, full yaw, leg ±0.10 rad / ±0.5 rad/s, arm ±0.05, and a command-aligned
  initial velocity (`mdp/events.py:42-46, 298-354, 546-554`).
- Trace step 0 (after one control step) shows base z 0.6478.

### 7.3 Bodies and colliders (from pxr on the composed stage)

- Trunk `body`: 16.708 kg, COM (0, 0, -0.00496), articulation root. Colliders: `body_collision`
  (convex hull, 6752 pts), `box` (cube at (0.4, 0, -0.022), half-extents 0.0225/0.09/0.09, size 1
  times scale (0.045, 0.18, 0.18)), and `box_1` (cube at (-0.4, 0, 0), scale (0.065, 0.16, 0.145)).
- hip: 1.1369 kg, no collider.
- uleg: 2.2562 kg, convex hull (4414 pts).
- lleg: **0.33 kg** (USD value), convex hull (1905 pts).
- `*_foot`: separate body, mass 1e-6 kg, I = 1e-9, fixed to lleg at (0, 0, -0.3365). Collider:
  **sphere r = 0.036** at the foot origin.
- Arm links carry convex-hull meshes (sh1_1, el1_main, wr1_1, fingers, jaw).
- **The trunk/leg colliders are the raw URDF convex hulls, not faster's primitive capsules.**
  Config `contact.geometry: primitive` is not implemented by the port; it only warns
  (`env.py:272-324`).
- **Masses are the USD's.** `_apply_inertial_parity` is referenced in a comment (`env.py:450-454`)
  but no longer exists; grep finds no definition. The lleg therefore stays at 0.33 kg, not
  faster's 0.5254.

### 7.4 Collision filtering and self-collision

- `body` has `newton:selfCollisionEnabled = 1` and
  `rel physics:filteredPairs = [16 leg prims: *_hip, *_uleg, *_lleg, *_foot]`
  (`Physics.usda:423-445`).
- **`PhysicsFilteredPairsAPI` is not in the prim's applied schemas** (pxr's `GetAppliedSchemas`
  returns only ArticulationRoot, RigidBody, Mass, Collision, MeshCollision, MaterialBinding, and
  GeomModel). Newton's importer reads the relationship anyway (`make_selfcol_asset.py:21-24`).
- Intent: leg vs leg collides, trunk vs leg is filtered, and parent/child pairs are filtered by
  the importer.
- For PhysX:
  - Apply `PhysicsFilteredPairsAPI` on `body` with the same targets.
  - Enable articulation self-collision (`PhysxArticulationAPI.enabledSelfCollisions=True`).
  - Otherwise trunk vs upper-leg overlap gives the ~426 kN phantom contact the docstring records
    (`make_selfcol_asset.py:10-15`).
- Arm vs trunk is not filtered.
- Newton-only schemas PhysX will ignore: `NewtonJointAPI` (velocityLimit), `NewtonMassAPI`
  (`newton:inertia` full tensors; PhysX uses `physics:diagonalInertia` + `principalAxes`, which
  are also authored), `NewtonCollisionAPI`, `NewtonMeshCollisionAPI`, `NewtonArticulationRootAPI`
  (`selfCollisionEnabled`), and `NewtonSceneAPI`.
- Mesh colliders use `physics:approximation = "convexHull"`. PhysX cooks its own hull, limited to
  64 vertices by default, so shank and thigh hulls will differ slightly from Newton's.

### 7.5 Contact material in training (`env.py:326-368`)

- Robot shapes: mu 0.75, ke 5e4, kd 5e2, kf 1e3.
- Ground: ke 2.5e3, kd 1e2, kf 1e3. `mu = ground_mu(startup_dr) = friction_range[0] = 0.3` when
  `disturb=True` (training), and **1.0** when `disturb=False` (eval) (`events_cfg.py:44-61`).
- Training DR on **robot** shapes: per-env surface draw U(0.3, 1.0) plus per-shape jitter ±0.05,
  clamped to [0.3, 1.0], refreshed on a 16-step batch cadence (`events_cfg.py:87-141`).
- MuJoCo combines a pair's friction with **max** (equal priority, `newton/_src/solvers/mujoco/kernels.py:162-165`).
  - **Training effective foot-ground mu:** the robot's draw, in U[0.3, 1.0].
  - **Eval effective mu:** max(0.75, 1.0) = 1.0.
- For PhysX, set friction combine mode `max`, or set both materials to your target. Something in
  0.75-1.0 is squarely in distribution.
- Restitution is not written anywhere (`env.py:375-379`). Treat it as 0.
- Pyramidal cone, gravity -9.81.

---

## 8. Reference trace

File: `CL/runs/eval/2026-08-28/legcol60/port_trace.npz`, written by `scripts/eval.py:442-473, 511`
with the `.pt` checkpoint, not the ONNX. Settings: 64 envs, 500 steps, fixed command (0.5, 0, 0),
`play=True`. All keys are float32 and indexed `[step, env, ...]`, recorded **after** each control
step (`eval.py:417-440`).

| key | shape | note |
|---|---|---|
| `root_pos_w` | (500, 64, 3) | env origins included (env 0 at (8.75, -8.75)) |
| `root_quat_w` | (500, 64, 4) | **XYZW** (step 0: `[1.2e-4, -1.9e-3, -3e-6, 0.99999]`; `eval.py:420-421`) |
| `lin_vel_b` | (500, 64, 3) | COM lin vel, trunk frame |
| `ang_vel_b` | (500, 64, 3) | = obs[0:3] |
| `grav_b` | (500, 64, 3) | = obs[3:6] |
| `joint_pos` | (500, 64, 19) | deploy order, **absolute** |
| `joint_vel` | (500, 64, 19) | deploy order |
| `torque` | (500, 64, 19) | **all zeros** (never written, section 5.7) |
| `feet_pos_w` | (500, 64, 4, 3) | |
| `feet_vel_w` | (500, 64, 4, 3) | |
| `feet_height` | (500, 64, 4) | |
| `feet_pos_b` | (500, 64, 4, 3) | |
| `cmd` | (500, 64, 3) | = obs[6:9] |
| `done` | (500, 64) | sum 0 |
| `term` | (500, 64) | sum 0 |

**Missing:** the observation vector, actions, hidden state, phase and height command. The trace
therefore **cannot score ONNX actions directly**.

You can rebuild every observation term:

- The phase is deterministic: 0 at reset, then `0.022*k`.
- Height is 0.56, because `set_fixed` takes the mean of [0.52, 0.60].
- Orientation is 0.
- Last actions can be regenerated by running the ONNX recurrently with the lag-2 rule.

But the actions it produces can only be sanity-checked against the measured joint positions. I
tried that. The mean |target − q| is about 0.20 rad for every convention variant, so the check
cannot tell them apart (PD sag under load dominates):

| variant | mean \|target − q\| (rad) |
|---|---|
| lag-2 + arm fix | 0.1977 |
| lag-1 | 0.2006 |
| raw stow arm | 0.2128 |
| h = 0.58 | 0.2002 |

The report's three repeats have ~0 spread, so the rollout is deterministic.

Reference metrics from `port_report.json`:

| metric | value |
|---|---|
| mean contacts | 3.00 |
| duty | 0.73-0.77 |
| rmse vx | 0.038 |
| base height | 0.5385 m |
| falls | 0 |

**To get a replayable trace**, the eval would need to also record `obs`, `action` and the hidden
state. That is a code change, so I did not make it.

---

## 9. Minimal runner loop implied by the above (pseudocode, not an implementation)

```
at start: h = zeros(2,1,256); phase = 0.0; A = [ref_legs, ref_legs]    # last two ONNX outputs
          fh = [0.036]*4                                               # k = 0 value (section 3.4)
every 20 ms (query k):
    obs inputs: w_b (trunk-frame ang vel), g_b (unit), cmd (vx,vy,wz), q19 with arm shifted
                (q_arm - stow + carry), qd19, last_actions = A[-2], [0.58], [0,0], fh
    target12, h = onnx(...)
    target12 = clamp(target12, leg_lo, leg_hi); A.append(target12)
    for 4 physics steps (5 ms): read q, qd -> tau (section 5.2, arm target = stow) -> apply efforts -> step
    v_xy = ||COM lin vel in trunk frame [:2]||
    phase = (phase + 0.022) % 1; fh = gait(phase, cmd, v_xy)       # section 3.4
```

Note that A[-2] is the output of query k-2, **before** the soft-limit clamp. Training stored the
raw action, so use the unclamped `actions_output`. The clamp only affects the target, never the
observation.

---

## 10. Open ambiguities and unverified items

1. **Arm offset (trap 1) is a real contract bug in the export.** I verified it from code and from
   the ONNX initializers. The static test, holding everything else nominal, changes the output by
   up to 0.3 rad. **Not verified:** a closed-loop rollout with the corrected versus uncorrected
   arm input. Any other consumer of `policy.yaml` that feeds absolute arm angles has the same
   problem.
2. **Lag-2 `last_actions`.** Verified by reading `env.py:670-672` + `observations.py:51-52` and
   `FA/env.py:1006-1007`. It was not verified by a rollout, because the trace does not record
   actions.
3. **Knee mask ordering.** `FusedDelayedPDActuator` marks knees by **index** (2, 5, 8, 11)
   (`actuators_native.py:316-321`), which is only correct if IsaacLab's public joint order is
   `DEPLOY_JOINT_NAMES`. In this checkout `joint_ordering` is live (7.1), so the mask is correct.
   I did not run IsaacLab to print `robot.joint_names`. If it had been arm-first, fl_hy and fr_hy
   would have received the knee envelope, which would pin them to about -63 N·m. The good gait
   metrics argue strongly against that.
4. **Training environment variables.** Only `SPOT_USD_DIR=spot_legcol` is documented for this run
   (task brief; `docs/POSTURE_AUTHORITY.md:28`). I assumed `FASTER_ACT`, `ACT_KIND`, `NCONMAX`,
   `LIMIT_KE`, `LIMIT_KD`, `DR_CONTACT_JITTER` and `ARM_PATH` were at their defaults. There is no
   record of the actual shell environment.
5. **The reference trace's eval flags are unknown.** `--task` defaults to `crawl_ab`, whose
   physics blocks match `locomanipulation` in actuator, action scales and posture. It is also
   unknown whether `SPOT_USD_DIR` was set at eval time. The trace's arm stayed at stow.
6. **PhysX joint friction.** MuJoCo `frictionloss` is an absolute torque. I did not check the
   Isaac Sim 6.0 docs for the exact PhysX attribute. The IsaacLab ActuatorBase here exposes
   `friction`, `dynamic_friction` and `viscous_friction` (`actuator_base.py:96-103`), which
   suggests PhysX joint static/dynamic friction efforts exist. Confirm the units (effort vs.
   coefficient) before using them, or model 0.008 / 0.18 N·m Coulomb friction in the torque law
   yourself.
7. **Joint velocity limit.** The USD authors 100 rad/s (`newton:velocityLimit`). I did not check
   whether MuJoCo-Warp enforced it (MuJoCo has no joint velocity limit). In PhysX, set
   `maxJointVelocity` high or 100 rad/s. It should never bind in normal gait.
8. **Contact model.** The MuJoCo soft contact (robot ke 5e4 / kd 5e2 into solref) and soft joint
   limits cannot be reproduced exactly in PhysX. Expect small stance-height and slip differences.
   The measured shank load share (docs mention ~61% on the shanks with convex hulls) is a property
   of this collider set, so keep the convex-hull shanks if you want the same plant.
9. **Foot links.** The 1e-6 kg / 1e-9 inertia foot bodies on fixed joints are harmless in MuJoCo.
   PhysX articulations handle fixed joints as links, which may cause solver conditioning issues.
   If they do, merging the sphere into `*_lleg` at (0, 0, -0.3365) is physically equivalent.
10. **`faster` vs crawl-lab.** The task named `FA/actuators.py` as "the actuator model". It is not
    what trained this policy; the differences are in 5.7. If you copy the Warp kernel verbatim
    you add an arm speed derate the policy never had. That effect is mild while the arm is parked.
11. **First-observation foot heights** in training came from the previous episode's last gait
    update (stale for one step). This is irrelevant for deployment, but it means there is no
    "correct" k=0 value beyond what eval used (`[0.036]*4`).
