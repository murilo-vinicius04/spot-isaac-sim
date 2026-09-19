# legcol60 in Newton (crawl-lab reference eval, 64 envs) vs PhysX runs of spot_isaac6/walk.py.
# Every metric comes from crawl-lab's own scripts/eval.py score() on each trace, so both sims are graded identically.
#   python3 spot_isaac6/validation/plot.py <out.png> <label=trace.npz> [...]
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); CL = os.path.join(HERE, "..", "..", "external", "crawl-lab")   # submodule: scoring code
sys.argv, _args = [sys.argv[0]], sys.argv[1:]                 # eval.py parses argv at import


src = open(os.path.join(CL, "scripts", "eval.py")).read()
ns = {"__name__": "clevel"}; exec(compile(src.split("\nif _args.analyze:")[0], "eval.py", "exec"), ns)   # defs only, no sim
score, STEP_DT, GEOM_Z = ns["score"], ns["STEP_DT"], ns["GEOM_Z"]
out = _args[0]; runs = [("Newton (64 envs)", os.path.join(HERE, "newton_reference", "port_trace.npz"))]   # crawl-lab runs/eval/2026-08-28/legcol60
runs += [tuple(x.split("=", 1)) for x in _args[1:]]
T = {k: dict(np.load(p)) for k, p in runs}; M = {k: score(T[k]) for k in T}
cols = {"Newton (64 envs)": "#444444"}; pal = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
for i, (k, _) in enumerate(runs[1:]): cols[k] = pal[i % len(pal)]
fig = plt.figure(figsize=(15, 10)); gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.28)
# 1) key metrics side by side
keys = [("track/mean_vx", "mean vx (m/s)\ncmd 0.5"), ("track/rmse_vx", "rmse vx (m/s)"), ("track/rmse_vy", "rmse vy (m/s)"),
        ("thrash/base_height", "base height (m)"), ("thrash/roll_p99_deg", "roll p99 (deg)"), ("thrash/pitch_p99_deg", "pitch p99 (deg)"),
        ("gait/mean_contacts", "feet down (mean)"), ("slip/net_slide_hind", "hind slide / stance (m)"), ("fail/fell_frac", "fell fraction")]
axm = fig.add_subplot(gs[0, :])
w = 0.8 / len(M); x = np.arange(len(keys))
for j, (k, m) in enumerate(M.items()):
    ref = np.array([max(abs(M["Newton (64 envs)"][kk]), 1e-9) for kk, _ in keys])
    v = np.array([m[kk] for kk, _ in keys]); r = np.where(ref > 1e-6, v / ref, 1.0 + v)
    b = axm.bar(x + (j - (len(M) - 1) / 2) * w, r, w, color=cols[k], label=k)
    for xi, (vi, ri) in enumerate(zip(v, r)):
        axm.text(xi + (j - (len(M) - 1) / 2) * w, ri + 0.03, f"{vi:.3g}", ha="center", va="bottom", fontsize=6.5, rotation=90)
axm.axhline(1.0, color="k", lw=0.6, ls="--"); axm.set_xticks(x); axm.set_xticklabels([l for _, l in keys], fontsize=8)
axm.set_ylabel("ratio to Newton (bar labels = value)"); axm.set_ylim(0, 2.2); axm.legend(fontsize=8, ncol=len(M), loc="upper left")
axm.set_title("legcol60: Newton reference vs PhysX (Isaac Sim 6.0), same eval.py scoring, cmd vx 0.5 m/s, 10 s")
# 2) time series: vx, base height, pitch
t = lambda tr: np.arange(tr["root_pos_w"].shape[0]) * STEP_DT
def band(ax, tr, y, c, lab):
    if y.shape[1] > 1:
        ax.fill_between(t(tr), np.percentile(y, 5, 1), np.percentile(y, 95, 1), color=c, alpha=0.2, lw=0); ax.plot(t(tr), y.mean(1), color=c, lw=1.2, label=lab + " mean / 5-95%")
    else: ax.plot(t(tr), y[:, 0], color=c, lw=0.9, label=lab)
for col, (fn, ttl) in enumerate([(lambda tr: tr["lin_vel_b"][..., 0], "forward speed vx (m/s)"), (lambda tr: tr["root_pos_w"][..., 2], "base height (m)"),
                                 (lambda tr: np.degrees(np.arcsin(np.clip(tr["grav_b"][..., 0], -1, 1))), "pitch (deg)")]):
    ax = fig.add_subplot(gs[1, col])
    for k in M: band(ax, T[k], fn(T[k]), cols[k], k)
    ax.set_title(ttl, fontsize=10); ax.set_xlabel("t (s)"); ax.grid(alpha=0.3)
    if col == 0: ax.legend(fontsize=7)
# 3) gait diagrams (contact = foot centre within GEOM_Z of the floor), last 4 s, Newton env 0 vs the first PhysX run
for col, k in enumerate(list(M)[:3]):
    ax = fig.add_subplot(gs[2, col]); tr = T[k]; c = tr["feet_height"][:, 0] < GEOM_Z; tt = t(tr); sel = tt >= tt[-1] - 4
    for f in range(4):
        on = c[sel, f]; ax.fill_between(tt[sel], f + 0.1, f + 0.9, where=on, step="mid", color=cols[k], lw=0)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5]); ax.set_yticklabels(["FL", "FR", "HL", "HR"]); ax.set_xlabel("t (s)")
    ax.set_title(f"foot contact, last 4 s: {k}" + (" (env 0)" if tr["feet_height"].shape[1] > 1 else ""), fontsize=9)
fig.savefig(out, dpi=120, bbox_inches="tight"); print("wrote", out)
for k, m in M.items(): print(f"{k:28s} vx {m['track/mean_vx']:.4f} rmse_vx {m['track/rmse_vx']:.4f} h {m['thrash/base_height']:.4f} "
                              f"pitch99 {m['thrash/pitch_p99_deg']:.2f} roll99 {m['thrash/roll_p99_deg']:.2f} contacts {m['gait/mean_contacts']:.3f} fell {m['fail/fell_frac']:.2f}")
