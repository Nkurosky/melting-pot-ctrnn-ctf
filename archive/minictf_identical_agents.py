"""
Stage 3 — mini-CTF (v1, minimal).
Two identical agents (same genome) play symmetric sides.
No tagging, no collisions, no own-flag-at-home rule — just:
    seek enemy flag  →  touch it to pick up (carry)  →  return to own base to score.

Sensors: 5 channels × 4 rays + 1 carry bit = 21 inputs.
    ch0 enemy flag, ch1 own flag, ch2 enemy agent, ch3 own base, ch4 enemy base.
Brain: 3-neuron fully-connected CTRNN (same as Stage 1). Let's see if it fits.
Fitness: sum of both sides' rewards (self-identical pair => symmetric).

Run:
    python minictf.py                         # full evolve
    python minictf.py --quick                 # smoke test
    python minictf.py --watch-every 5         # live peek every 5 gens
    python minictf.py --seed-from best_genome.npy  # warm-start from Stage 1
    python minictf.py --replay best_minictf.npy    # just animate a saved agent
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from toy_ctf import (
    sigmoid, unpack_genome, sense_channels, param_count,
    N_NEURONS, N_RAYS, RAY_ANGLES, RAY_HALF_FOV, MAX_RANGE,
    ARENA, DT, AGENT_R, FLAG_R, CAPTURE_DIST, WHEEL_BASE, MAX_SPEED,
)

# ---- Stage 3 config ----
N_CHANNELS  = 5                         # enemy-flag, own-flag, enemy-agent, own-base, enemy-base
N_SENSORS   = N_CHANNELS * N_RAYS + 1   # + carry bit (proprioceptive)
N_PARAMS    = param_count(N_SENSORS)    # 3 neurons: 9+3+3 + 3*21 = 78 params
EP_STEPS    = 500

# Bases (static, symmetric)
RED_BASE  = np.array([3.0, ARENA/2])
BLUE_BASE = np.array([ARENA-3.0, ARENA/2])
BASE_R    = 1.0                         # "touching base" radius

# Reward shaping
R_PICKUP     = 50.0
R_SCORE      = 300.0
DIST_PRESSURE = 1.0     # coefficient on -distance_to_current_target per second


def stage_seed_from_stage1(path):
    """Take a Stage-1 (8-sensor) genome and embed it in a Stage-3 (21-sensor) genome.
    Recurrent weights + biases + taus carry over; sensor weights are zero-padded
    into the 'enemy flag' and 'enemy agent' channels (ch0 and ch2) to preserve
    the learned flag/opponent reflexes. Everything else starts at 0."""
    g1 = np.load(path)
    from toy_ctf import N_SENSORS as N_S1
    W1, b1, tr1, Ws1 = unpack_genome(g1, N_S1)  # Ws1 shape (8, 3)
    new = np.zeros(N_PARAMS)
    i = 0
    new[i:i+N_NEURONS**2] = W1.flatten(); i += N_NEURONS**2
    new[i:i+N_NEURONS]    = b1;            i += N_NEURONS
    new[i:i+N_NEURONS]    = tr1;           i += N_NEURONS
    Ws3 = np.zeros((N_SENSORS, N_NEURONS))
    # Stage 1 layout: rays 0..3 = flag channel, rays 4..7 = player channel
    # Stage 3 layout: ch0 (enemy flag) = rays 0..3, ch2 (enemy agent) = rays 8..11
    Ws3[0:N_RAYS]                 = Ws1[0:N_RAYS]           # enemy flag
    Ws3[2*N_RAYS:3*N_RAYS]        = Ws1[N_RAYS:2*N_RAYS]    # enemy agent
    new[i:i+N_SENSORS*N_NEURONS]  = Ws3.flatten()
    return new


# -------------------- one episode (two identical agents) --------------------
def run_match(genome, seed, record=False):
    rng = np.random.default_rng(seed)
    W, b, taus, Ws = unpack_genome(genome, N_SENSORS)

    # Agent positions
    rx, ry = RED_BASE + rng.normal(0, 0.5, 2)
    bx, by = BLUE_BASE + rng.normal(0, 0.5, 2)
    rh = rng.uniform(-np.pi, np.pi)
    bh = rng.uniform(-np.pi, np.pi)

    # Flags at each base initially
    rfx, rfy = RED_BASE.copy()          # red's own flag (blue wants to steal this)
    bfx, bfy = BLUE_BASE.copy()         # blue's own flag (red wants to steal this)
    red_carrying  = False                # red carries the blue flag
    blue_carrying = False                # blue carries the red flag

    yr = np.zeros(N_NEURONS)            # neuron states, red
    yb = np.zeros(N_NEURONS)            # neuron states, blue

    fit = 0.0
    red_score = blue_score = 0
    trail = []

    for t in range(EP_STEPS):
        # ----- sense -----
        # Red's perspective: enemy=blue
        obs_r_ch = [
            [(bfx, bfy)],      # enemy flag   (blue flag — wherever it currently is)
            [(rfx, rfy)],      # own flag
            [(bx, by)],        # enemy agent
            [tuple(RED_BASE)], # own base
            [tuple(BLUE_BASE)],# enemy base
        ]
        obs_r = sense_channels(rx, ry, rh, obs_r_ch)
        obs_r = np.append(obs_r, 1.0 if red_carrying else 0.0)

        obs_b_ch = [
            [(rfx, rfy)],
            [(bfx, bfy)],
            [(rx, ry)],
            [tuple(BLUE_BASE)],
            [tuple(RED_BASE)],
        ]
        obs_b = sense_channels(bx, by, bh, obs_b_ch)
        obs_b = np.append(obs_b, 1.0 if blue_carrying else 0.0)

        # ----- CTRNN steps -----
        sig_r = sigmoid(yr + b)
        yr = yr + DT * (-yr + W @ sig_r + Ws.T @ obs_r) / taus
        out_r = sigmoid(yr + b)
        vlr = (2*out_r[1]-1)*MAX_SPEED; vrr = (2*out_r[2]-1)*MAX_SPEED
        v_r = 0.5*(vlr+vrr); w_r = (vrr-vlr)/WHEEL_BASE

        sig_b = sigmoid(yb + b)
        yb = yb + DT * (-yb + W @ sig_b + Ws.T @ obs_b) / taus
        out_b = sigmoid(yb + b)
        vlb = (2*out_b[1]-1)*MAX_SPEED; vrb = (2*out_b[2]-1)*MAX_SPEED
        v_b = 0.5*(vlb+vrb); w_b = (vrb-vlb)/WHEEL_BASE

        # ----- integrate -----
        rh = (rh + w_r*DT + np.pi) % (2*np.pi) - np.pi
        bh = (bh + w_b*DT + np.pi) % (2*np.pi) - np.pi
        rx = np.clip(rx + v_r*np.cos(rh)*DT, AGENT_R, ARENA-AGENT_R)
        ry = np.clip(ry + v_r*np.sin(rh)*DT, AGENT_R, ARENA-AGENT_R)
        bx = np.clip(bx + v_b*np.cos(bh)*DT, AGENT_R, ARENA-AGENT_R)
        by = np.clip(by + v_b*np.sin(bh)*DT, AGENT_R, ARENA-AGENT_R)

        # ----- flag follows carrier -----
        if red_carrying:
            bfx, bfy = rx, ry
        if blue_carrying:
            rfx, rfy = bx, by

        # ----- pickup checks (must NOT already be carrying) -----
        if not red_carrying and np.hypot(rx-bfx, ry-bfy) < CAPTURE_DIST:
            red_carrying = True
            fit += R_PICKUP
        if not blue_carrying and np.hypot(bx-rfx, by-rfy) < CAPTURE_DIST:
            blue_carrying = True
            fit += R_PICKUP

        # ----- score checks -----
        if red_carrying and np.hypot(rx-RED_BASE[0], ry-RED_BASE[1]) < BASE_R:
            fit += R_SCORE
            red_score += 1
            red_carrying = False
            bfx, bfy = BLUE_BASE          # respawn blue flag at blue base
        if blue_carrying and np.hypot(bx-BLUE_BASE[0], by-BLUE_BASE[1]) < BASE_R:
            fit += R_SCORE
            blue_score += 1
            blue_carrying = False
            rfx, rfy = RED_BASE

        # ----- shaped distance pressure -----
        # target depends on carry state
        tgt_r = RED_BASE  if red_carrying  else np.array([bfx, bfy])
        tgt_b = BLUE_BASE if blue_carrying else np.array([rfx, rfy])
        dr = np.hypot(rx-tgt_r[0], ry-tgt_r[1])
        db = np.hypot(bx-tgt_b[0], by-tgt_b[1])
        fit -= (dr + db) * DT * DIST_PRESSURE * 0.5   # /2 so magnitude ~ Stage 1

        if record:
            trail.append((rx, ry, rh, bx, by, bh,
                          rfx, rfy, bfx, bfy,
                          red_carrying, blue_carrying,
                          obs_r.copy(), obs_b.copy()))

    return fit, red_score, blue_score, trail


# -------------------- GA --------------------
def evolve(pop=40, gens=100, trials=3, mut=0.15, elite=2, seed=0,
           resume=None, seed_from=None, watch_every=0):
    rng = np.random.default_rng(seed)
    if resume:
        g0 = np.load(resume); assert g0.shape == (N_PARAMS,)
        genomes = np.tile(g0, (pop,1)) + rng.normal(0, 0.05, (pop, N_PARAMS))
        genomes[0] = g0
        print(f"Resumed from {resume}")
    elif seed_from:
        g0 = stage_seed_from_stage1(seed_from)
        genomes = np.tile(g0, (pop,1)) + rng.normal(0, 0.1, (pop, N_PARAMS))
        genomes[0] = g0
        print(f"Warm-started from Stage-1 genome {seed_from} (shape-expanded)")
    else:
        genomes = rng.normal(0, 1.0, size=(pop, N_PARAMS))

    hist_fit, hist_scores = [], []
    viewer = _LiveViewer() if watch_every > 0 else None

    for g in range(gens):
        fits = np.zeros(pop); scs = np.zeros(pop)
        for i, gen in enumerate(genomes):
            for k in range(trials):
                # seed depends on gen & trial only — every genome in this gen
                # sees the SAME random spawns, so fitness is directly comparable
                # and the elite's score doesn't jitter just because it moved indices
                f, rs, bs, _ = run_match(gen, seed=g*10_000 + k*131)
                fits[i] += f
                scs[i]  += (rs + bs)
            fits[i] /= trials; scs[i] /= trials
        order = np.argsort(-fits)
        genomes = genomes[order]; fits = fits[order]; scs = scs[order]
        hist_fit.append(fits[0]); hist_scores.append(scs[0])
        # canonical eval: fixed seeds, unchanged across gens → elite's canonical
        # fitness should rise monotonically (or at least never drop) if inheritance works
        canon = np.mean([run_match(genomes[0], seed=77_000+k)[0] for k in range(3)])
        print(f"gen {g:3d}  best={fits[0]:7.1f}  canon={canon:7.1f}  "
              f"mean={fits.mean():7.1f}  scores(best)={scs[0]:.2f}")

        if viewer is not None and (g % watch_every == 0 or g == gens-1):
            _, rs, bs, trail = run_match(genomes[0], seed=g*9+1, record=True)
            viewer.play(trail, gen=g, fit=fits[0], red=rs, blue=bs)

        new = [genomes[i].copy() for i in range(elite)]
        while len(new) < pop:
            idxs = rng.choice(pop, 3, replace=False)
            parent = genomes[min(idxs)]
            new.append(parent + rng.normal(0, mut, N_PARAMS))
        genomes = np.array(new)

    return genomes[0], hist_fit, hist_scores


# -------------------- viewer --------------------
class _LiveViewer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_xlim(0, ARENA); self.ax.set_ylim(0, ARENA); self.ax.set_aspect('equal')
        # bases
        self.ax.add_patch(Rectangle(RED_BASE - [BASE_R,BASE_R], 2*BASE_R, 2*BASE_R,
                                    color='tab:red', alpha=0.15))
        self.ax.add_patch(Rectangle(BLUE_BASE - [BASE_R,BASE_R], 2*BASE_R, 2*BASE_R,
                                    color='tab:blue', alpha=0.15))
        self.R  = Circle((0,0), AGENT_R, color='tab:red');    self.ax.add_patch(self.R)
        self.B  = Circle((0,0), AGENT_R, color='tab:blue');   self.ax.add_patch(self.B)
        self.RF = Circle((0,0), FLAG_R,  color='darkred',  alpha=0.9); self.ax.add_patch(self.RF)
        self.BF = Circle((0,0), FLAG_R,  color='navy',     alpha=0.9); self.ax.add_patch(self.BF)
        self.hR, = self.ax.plot([], [], 'k-', lw=1.5)
        self.hB, = self.ax.plot([], [], 'k-', lw=1.5)

    def play(self, trail, gen, fit, red, blue, step=2):
        self.ax.set_title(f"gen {gen}  fit={fit:.1f}  score  red={red}  blue={blue}")
        for i in range(0, len(trail), step):
            (rx,ry,rh, bx,by,bh, rfx,rfy, bfx,bfy, rc, bc, *_) = trail[i]
            self.R.center=(rx,ry); self.B.center=(bx,by)
            self.RF.center=(rfx,rfy); self.BF.center=(bfx,bfy)
            # outline when carrying, to make it obvious
            self.BF.set_edgecolor('yellow' if rc else 'none'); self.BF.set_linewidth(2 if rc else 0)
            self.RF.set_edgecolor('yellow' if bc else 'none'); self.RF.set_linewidth(2 if bc else 0)
            self.hR.set_data([rx, rx+np.cos(rh)], [ry, ry+np.sin(rh)])
            self.hB.set_data([bx, bx+np.cos(bh)], [by, by+np.sin(bh)])
            self.fig.canvas.draw_idle(); plt.pause(0.005)


def replay(path, seed=42):
    g = np.load(path)
    _, rs, bs, trail = run_match(g, seed=seed, record=True)
    print(f"red_score={rs}  blue_score={bs}")
    v = _LiveViewer(); plt.ioff()
    v.play(trail, gen=-1, fit=0, red=rs, blue=bs, step=1)
    plt.show()


# -------------------- CLI --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--seed-from", dest="seed_from", type=str, default=None,
                    help="path to a Stage-1 best_genome.npy (8-sensor) to warm-start from")
    ap.add_argument("--watch-every", type=int, default=0)
    ap.add_argument("--out", type=str, default="best_minictf.npy")
    ap.add_argument("--replay", type=str, default=None,
                    help="skip evolution, just animate a saved genome")
    args = ap.parse_args()

    if args.replay:
        replay(args.replay, seed=args.seed + 999)
        raise SystemExit

    pop  = args.pop  or (16 if args.quick else 40)
    gens = args.gens or (10 if args.quick else 120)

    print(f"Stage 3 mini-CTF  pop={pop} gens={gens} N_PARAMS={N_PARAMS} "
          f"(3 neurons × {N_SENSORS} sensors)")
    best, hist_fit, hist_sc = evolve(pop=pop, gens=gens, seed=args.seed,
                                     resume=args.resume, seed_from=args.seed_from,
                                     watch_every=args.watch_every)
    np.save(args.out, best)
    print(f"Saved {args.out}")

    fig, (a1,a2) = plt.subplots(1,2, figsize=(10,4))
    a1.plot(hist_fit); a1.set_xlabel("gen"); a1.set_ylabel("best fitness")
    a2.plot(hist_sc);  a2.set_xlabel("gen"); a2.set_ylabel("best scores/ep (red+blue)")
    plt.tight_layout(); plt.savefig("minictf_fitness.png")
    print("Saved minictf_fitness.png")

    replay(args.out, seed=args.seed + 999)
