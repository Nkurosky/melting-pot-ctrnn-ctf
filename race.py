"""
Stage 2 — symmetric race co-evolution.
Two agents, one shared flag. First to touch wins.
Single population, self-play vs random peers + a small hall-of-fame.

CTRNN + sensor model imported from toy_ctf so we evolve the SAME brain architecture.
Sensors still have two channels — channel 0 = flag, channel 1 = opponent agent.

Run:
    python race.py                          # full run
    python race.py --quick                  # smoke test
    python race.py --watch-every 5          # live-watch best-vs-best match
    python race.py --resume best_race.npy   # continue
    python race.py --seed-from best_genome.npy   # warm-start from Stage 1 solo agent
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from toy_ctf import (
    sigmoid, unpack, sense,
    N_PARAMS, N_NEURONS, N_SENSORS, N_RAYS, RAY_ANGLES, MAX_RANGE,
    ARENA, DT, AGENT_R, FLAG_R, CAPTURE_DIST, WHEEL_BASE, MAX_SPEED,
)

EP_STEPS = 400

# ---------------- per-agent CTRNN step ----------------
def init_agent(genome):
    W, b, taus, Ws = unpack(genome)
    return {"y": np.zeros(N_NEURONS), "W": W, "b": b, "taus": taus, "Ws": Ws}

def step_agent(st, obs):
    sig = sigmoid(st["y"] + st["b"])
    I = st["Ws"].T @ obs
    dy = (-st["y"] + st["W"] @ sig + I) / st["taus"]
    st["y"] = st["y"] + DT * dy
    out = sigmoid(st["y"] + st["b"])
    vl = (2*out[1] - 1) * MAX_SPEED
    vr = (2*out[2] - 1) * MAX_SPEED
    v = 0.5 * (vl + vr)
    w = (vr - vl) / WHEEL_BASE
    return v, w

# ---------------- one head-to-head match ----------------
def run_match(gen_a, gen_b, seed, record=False):
    rng = np.random.default_rng(seed)
    # symmetric spawn on opposite sides
    swap = rng.integers(0, 2)
    ax, bx = (3.0, ARENA-3.0) if swap == 0 else (ARENA-3.0, 3.0)
    ay = rng.uniform(4, ARENA-4)
    by = rng.uniform(4, ARENA-4)
    ah = rng.uniform(-np.pi, np.pi)
    bh = rng.uniform(-np.pi, np.pi)
    # flag near centre with small jitter — symmetric on average
    fx = np.clip(ARENA/2 + rng.normal(0, 1.5), 3, ARENA-3)
    fy = np.clip(ARENA/2 + rng.normal(0, 1.5), 3, ARENA-3)

    sa = init_agent(gen_a); sb = init_agent(gen_b)
    fit_a = fit_b = 0.0
    winner = None
    trail = []

    for t in range(EP_STEPS):
        obs_a = sense(ax, ay, ah, [[(fx, fy)], [(bx, by)]])
        obs_b = sense(bx, by, bh, [[(fx, fy)], [(ax, ay)]])

        va, wa = step_agent(sa, obs_a)
        vb, wb = step_agent(sb, obs_b)

        ah = (ah + wa*DT + np.pi) % (2*np.pi) - np.pi
        bh = (bh + wb*DT + np.pi) % (2*np.pi) - np.pi
        ax = np.clip(ax + va*np.cos(ah)*DT, AGENT_R, ARENA-AGENT_R)
        ay = np.clip(ay + va*np.sin(ah)*DT, AGENT_R, ARENA-AGENT_R)
        bx = np.clip(bx + vb*np.cos(bh)*DT, AGENT_R, ARENA-AGENT_R)
        by = np.clip(by + vb*np.sin(bh)*DT, AGENT_R, ARENA-AGENT_R)

        da = np.hypot(ax-fx, ay-fy); db = np.hypot(bx-fx, by-fy)
        fit_a -= da * DT
        fit_b -= db * DT

        if record:
            trail.append((ax, ay, ah, bx, by, bh, fx, fy,
                          obs_a.copy(), obs_b.copy()))

        a_cap = da < CAPTURE_DIST
        b_cap = db < CAPTURE_DIST
        if a_cap and b_cap:
            winner = "tie"
            fit_a += 25; fit_b += 25
            break
        if a_cap:
            winner = "A"
            fit_a += 150 + 0.3*(EP_STEPS - t)
            fit_b -= 40
            break
        if b_cap:
            winner = "B"
            fit_b += 150 + 0.3*(EP_STEPS - t)
            fit_a -= 40
            break
    else:
        fit_a -= 10; fit_b -= 10   # timeout — both failed

    return fit_a, fit_b, winner, trail

# ---------------- evaluation (self-play + HoF) ----------------
def evaluate(genomes, hof, n_peers=4, n_hof=2, seed=0):
    pop = len(genomes)
    rng = np.random.default_rng(seed)
    fits   = np.zeros(pop)
    wins   = np.zeros(pop)
    losses = np.zeros(pop)
    matches= np.zeros(pop)
    for i in range(pop):
        peers = [j for j in range(pop) if j != i]
        opps  = rng.choice(peers, min(n_peers, len(peers)), replace=False).tolist()
        opp_genomes = [genomes[j] for j in opps]
        if hof:
            k = min(n_hof, len(hof))
            hof_idx = rng.choice(len(hof), k, replace=False).tolist()
            opp_genomes += [hof[j] for j in hof_idx]
        for j, og in enumerate(opp_genomes):
            s = (seed * 1_000_003 + i*1009 + j*17) & 0xffffffff
            fa, fb, winner, _ = run_match(genomes[i], og, seed=s)
            fits[i] += fa
            matches[i] += 1
            if winner == "A": wins[i] += 1
            elif winner == "B": losses[i] += 1
    fits /= np.maximum(matches, 1)
    winrate = wins / np.maximum(matches, 1)
    return fits, winrate, wins, losses

# ---------------- GA ----------------
def evolve(pop=40, gens=100, mut=0.15, elite=2, seed=0,
           resume=None, seed_from=None, watch_every=0, hof_every=10, hof_cap=10):
    rng = np.random.default_rng(seed)
    if resume:
        g0 = np.load(resume)
        genomes = np.tile(g0, (pop, 1)) + rng.normal(0, 0.05, (pop, N_PARAMS))
        genomes[0] = g0
        print(f"Resumed from {resume}")
    elif seed_from:
        g0 = np.load(seed_from)
        genomes = np.tile(g0, (pop, 1)) + rng.normal(0, 0.15, (pop, N_PARAMS))
        genomes[0] = g0
        print(f"Warm-started from {seed_from}")
    else:
        genomes = rng.normal(0, 1.0, size=(pop, N_PARAMS))

    hof = []
    best_hist, wr_hist = [], []

    viewer = _LiveViewer() if watch_every > 0 else None

    for g in range(gens):
        fits, winrate, wins, losses = evaluate(genomes, hof, seed=g+1)
        order = np.argsort(-fits)
        genomes = genomes[order]; fits = fits[order]; winrate = winrate[order]
        best_hist.append(fits[0]); wr_hist.append(winrate[0])
        print(f"gen {g:3d}  best={fits[0]:7.1f}  mean={fits.mean():7.1f}  "
              f"winrate(best)={winrate[0]:.2f}  hof={len(hof)}")

        if g % hof_every == 0:
            hof.append(genomes[0].copy())
            if len(hof) > hof_cap:
                hof.pop(0)

        if viewer is not None and (g % watch_every == 0 or g == gens-1):
            # best vs 2nd-best — shows strategy differentiation
            opponent = genomes[1] if len(genomes) > 1 else genomes[0]
            _, _, winner, trail = run_match(genomes[0], opponent, seed=g*131+7, record=True)
            viewer.play(trail, gen=g, fit=fits[0], winner=winner)

        # next generation
        new = [genomes[i].copy() for i in range(elite)]
        while len(new) < pop:
            idxs = rng.choice(pop, 3, replace=False)
            parent = genomes[min(idxs)]
            child = parent + rng.normal(0, mut, size=N_PARAMS)
            new.append(child)
        genomes = np.array(new)

    return genomes[0], best_hist, wr_hist, hof

# ---------------- live viewer ----------------
class _LiveViewer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6.5, 6.5))
        self.ax.set_xlim(0, ARENA); self.ax.set_ylim(0, ARENA); self.ax.set_aspect('equal')
        self.A = Circle((0,0), AGENT_R, color='tab:blue');   self.ax.add_patch(self.A)
        self.B = Circle((0,0), AGENT_R, color='tab:orange'); self.ax.add_patch(self.B)
        self.F = Circle((0,0), FLAG_R,  color='tab:green');  self.ax.add_patch(self.F)
        self.hA, = self.ax.plot([], [], 'k-', lw=1.5)
        self.hB, = self.ax.plot([], [], 'k-', lw=1.5)

    def play(self, trail, gen, fit, winner, step=2):
        self.ax.set_title(f"gen {gen}  fit={fit:.1f}  winner={winner}  "
                          f"(blue=best, orange=rival)")
        for i in range(0, len(trail), step):
            ax_, ay_, ah_, bx_, by_, bh_, fx_, fy_, _, _ = trail[i]
            self.A.center = (ax_, ay_); self.B.center = (bx_, by_); self.F.center = (fx_, fy_)
            self.hA.set_data([ax_, ax_+np.cos(ah_)], [ay_, ay_+np.sin(ah_)])
            self.hB.set_data([bx_, bx_+np.cos(bh_)], [by_, by_+np.sin(bh_)])
            self.fig.canvas.draw_idle(); plt.pause(0.005)

# ---------------- replay util ----------------
def animate_match(gen_a, gen_b, seed=42):
    _, _, winner, trail = run_match(gen_a, gen_b, seed=seed, record=True)
    print(f"winner: {winner}")
    v = _LiveViewer()
    plt.ioff()
    v.ax.set_title(f"replay — winner={winner}")
    for f in trail[::2]:
        ax_, ay_, ah_, bx_, by_, bh_, fx_, fy_, _, _ = f
        v.A.center = (ax_, ay_); v.B.center = (bx_, by_); v.F.center = (fx_, fy_)
        v.hA.set_data([ax_, ax_+np.cos(ah_)], [ay_, ay_+np.sin(ah_)])
        v.hB.set_data([bx_, bx_+np.cos(bh_)], [by_, by_+np.sin(bh_)])
        v.fig.canvas.draw_idle(); plt.pause(0.02)
    plt.show()

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None,
                    help="continue from a saved race champion (.npy)")
    ap.add_argument("--seed-from", dest="seed_from", type=str, default=None,
                    help="warm-start population from a Stage-1 solo agent")
    ap.add_argument("--watch-every", type=int, default=0)
    ap.add_argument("--out", type=str, default="best_race.npy")
    args = ap.parse_args()

    pop  = args.pop  or (16 if args.quick else 40)
    gens = args.gens or (10 if args.quick else 100)

    best, fits, wrs, hof = evolve(
        pop=pop, gens=gens, seed=args.seed,
        resume=args.resume, seed_from=args.seed_from,
        watch_every=args.watch_every,
    )
    np.save(args.out, best)
    np.save(args.out.replace(".npy", "_hof.npy"), np.array(hof))
    print(f"Saved {args.out} and hall-of-fame.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(fits); ax1.set_xlabel("gen"); ax1.set_ylabel("best fitness")
    ax2.plot(wrs);  ax2.set_xlabel("gen"); ax2.set_ylabel("best winrate")
    plt.tight_layout(); plt.savefig("race_fitness.png")
    print("Saved race_fitness.png")

    animate_match(best, hof[-1] if hof else best, seed=args.seed + 999)
