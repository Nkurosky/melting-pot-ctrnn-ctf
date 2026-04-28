"""
Toy Capture-the-Flag, Stage 1.
- 2D continuous arena
- 1 evolved agent (differential drive)
- 1 flag (goal), 1 wandering "player" (distractor)
- Sensors: 4 rays x 2 channels (flag, player), distance-attenuated
- Controller: 3-neuron fully-connected CTRNN (Beer-style)
- GA: tournament selection, Gaussian mutation, elitism

Run:    python toy_ctf.py            # evolve then animate best
        python toy_ctf.py --quick    # fewer gens, for smoke test
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge

# ---------------- World ----------------
ARENA        = 20.0
DT           = 0.1
EP_STEPS     = 300
AGENT_R      = 0.4
FLAG_R       = 0.4
PLAYER_R     = 0.4
CAPTURE_DIST = 1.0
WHEEL_BASE   = 1.0
MAX_SPEED    = 3.0

# ---------------- Sensors ----------------
N_RAYS       = 4
N_CHANNELS   = 2   # 0 = flag, 1 = player
N_SENSORS    = N_RAYS * N_CHANNELS
RAY_ANGLES   = np.deg2rad([-60.0, -20.0, 20.0, 60.0])
RAY_HALF_FOV = np.deg2rad(20.0)
MAX_RANGE    = 12.0

# ---------------- CTRNN ----------------
N_NEURONS    = 3            # classic minimal ER: 3 fully-connected neurons
# neuron 0 = "sensor-weighted" contributor / hidden
# neurons 1, 2 = motors (left / right wheel) — also fully recurrently connected
# All three receive weighted sensor input (let the GA pick effective roles).

BASE_PARAMS = (N_NEURONS * N_NEURONS  # recurrent weights
               + N_NEURONS            # biases
               + N_NEURONS)           # taus (raw, sigmoid-mapped)

def param_count(n_sensors):
    return BASE_PARAMS + n_sensors * N_NEURONS

N_PARAMS = param_count(N_SENSORS)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def unpack_genome(genome, n_sensors):
    i = 0
    W  = genome[i:i+N_NEURONS**2].reshape(N_NEURONS, N_NEURONS); i += N_NEURONS**2
    b  = genome[i:i+N_NEURONS];                                  i += N_NEURONS
    tr = genome[i:i+N_NEURONS];                                  i += N_NEURONS
    Ws = genome[i:i+n_sensors*N_NEURONS].reshape(n_sensors, N_NEURONS)
    taus = 0.5 + 4.5 * sigmoid(tr)   # tau in [0.5, 5.0]
    return W, b, taus, Ws

def unpack(genome):
    return unpack_genome(genome, N_SENSORS)

# ---------------- Sensing ----------------
def sense_channels(ax, ay, ah, objs_by_channel, ray_angles=RAY_ANGLES,
                   ray_half_fov=RAY_HALF_FOV, max_range=MAX_RANGE):
    """objs_by_channel: list of lists of (x, y) per channel."""
    n_rays = len(ray_angles)
    obs = np.zeros(len(objs_by_channel) * n_rays)
    for ch, objs in enumerate(objs_by_channel):
        for r, ra in enumerate(ray_angles):
            ray_dir = ah + ra
            best = 0.0
            for (ox, oy) in objs:
                dx, dy = ox - ax, oy - ay
                d = np.hypot(dx, dy)
                if d > max_range or d < 1e-6:
                    continue
                bearing = np.arctan2(dy, dx) - ray_dir
                bearing = (bearing + np.pi) % (2*np.pi) - np.pi
                if abs(bearing) <= ray_half_fov:
                    act = 1.0 - d / max_range
                    if act > best:
                        best = act
            obs[ch*n_rays + r] = best
    return obs

def sense(ax, ay, ah, objs_by_channel):
    return sense_channels(ax, ay, ah, objs_by_channel)

# ---------------- Episode ----------------
def run_episode(genome, seed, record=False):
    rng = np.random.default_rng(seed)
    W, b, taus, Ws = unpack(genome)

    ax, ay = rng.uniform(2, ARENA-2, 2)
    ah     = rng.uniform(-np.pi, np.pi)
    fx, fy = rng.uniform(2, ARENA-2, 2)
    while np.hypot(ax-fx, ay-fy) < 6.0:
        fx, fy = rng.uniform(2, ARENA-2, 2)
    px, py = rng.uniform(2, ARENA-2, 2)
    ph     = rng.uniform(-np.pi, np.pi)

    y = np.zeros(N_NEURONS)
    fitness = 0.0
    captures = 0
    trail = []

    for t in range(EP_STEPS):
        obs = sense(ax, ay, ah, [[(fx, fy)], [(px, py)]])

        # CTRNN step (Euler)
        sig = sigmoid(y + b)
        I = Ws.T @ obs
        dy = (-y + W @ sig + I) / taus
        y = y + DT * dy

        out = sigmoid(y + b)
        vl = (2*out[1] - 1) * MAX_SPEED
        vr = (2*out[2] - 1) * MAX_SPEED
        v = 0.5 * (vl + vr)
        w = (vr - vl) / WHEEL_BASE

        ah = (ah + w*DT + np.pi) % (2*np.pi) - np.pi
        ax = np.clip(ax + v*np.cos(ah)*DT, AGENT_R, ARENA-AGENT_R)
        ay = np.clip(ay + v*np.sin(ah)*DT, AGENT_R, ARENA-AGENT_R)

        # wandering distractor
        ph += rng.normal(0, 0.4)
        px = np.clip(px + np.cos(ph)*1.0*DT, PLAYER_R, ARENA-PLAYER_R)
        py = np.clip(py + np.sin(ph)*1.0*DT, PLAYER_R, ARENA-PLAYER_R)

        d_flag = np.hypot(ax-fx, ay-fy)
        fitness -= d_flag * DT  # steady pressure to close distance

        if d_flag < CAPTURE_DIST:
            captures += 1
            fitness += 50.0 + 0.2 * (EP_STEPS - t)  # big + earlier = better
            # respawn flag far away
            nfx, nfy = rng.uniform(2, ARENA-2, 2)
            while np.hypot(ax-nfx, ay-nfy) < 6.0:
                nfx, nfy = rng.uniform(2, ARENA-2, 2)
            fx, fy = nfx, nfy

        if record:
            trail.append((ax, ay, ah, fx, fy, px, py, obs.copy()))

    return fitness, captures, trail

# ---------------- GA ----------------
def evolve(pop=50, gens=80, trials=3, mut=0.15, elite=2, seed=0, verbose=True,
           resume=None, watch_every=0):
    rng = np.random.default_rng(seed)
    if resume is not None:
        seed_g = np.load(resume)
        assert seed_g.shape == (N_PARAMS,), f"resume genome shape {seed_g.shape} != {N_PARAMS}"
        genomes = np.tile(seed_g, (pop, 1)) + rng.normal(0, 0.05, size=(pop, N_PARAMS))
        genomes[0] = seed_g   # keep an exact copy
        if verbose: print(f"Resumed from {resume}")
    else:
        genomes = rng.normal(0, 1.0, size=(pop, N_PARAMS))
    best_hist = []

    live = None
    if watch_every > 0:
        live = _LiveViewer()

    for g in range(gens):
        fits = np.empty(pop)
        caps = np.empty(pop)
        for i, gen in enumerate(genomes):
            fs, cs = [], []
            for k in range(trials):
                f, c, _ = run_episode(gen, seed=g*10_000 + k*131 + 7)
                fs.append(f); cs.append(c)
            fits[i] = np.mean(fs)
            caps[i] = np.mean(cs)

        order = np.argsort(-fits)
        genomes = genomes[order]; fits = fits[order]; caps = caps[order]
        best_hist.append(fits[0])
        if verbose:
            print(f"gen {g:3d}  best={fits[0]:8.1f}  mean={fits.mean():8.1f}  captures(best)={caps[0]:.2f}")

        if live is not None and (g % watch_every == 0 or g == gens-1):
            _, _, trail = run_episode(genomes[0], seed=g*7+11, record=True)
            live.play(trail, gen=g, fit=fits[0])

        # next generation
        new = [genomes[i].copy() for i in range(elite)]
        while len(new) < pop:
            idxs = rng.choice(pop, 3, replace=False)
            parent = genomes[min(idxs)]  # sorted => smallest idx is fittest
            child = parent + rng.normal(0, mut, size=N_PARAMS)
            new.append(child)
        genomes = np.array(new)

    return genomes[0], best_hist

# ---------------- Live viewer (non-blocking, re-used each gen) ----------------
class _LiveViewer:
    def __init__(self):
        plt.ion()
        self.fig, (self.axw, self.axs) = plt.subplots(1, 2, figsize=(11, 5.5),
                                                      gridspec_kw={'width_ratios':[2,1]})
        self.axw.set_xlim(0, ARENA); self.axw.set_ylim(0, ARENA)
        self.axw.set_aspect('equal')
        self.agent  = Circle((0,0), AGENT_R, color='tab:blue');  self.axw.add_patch(self.agent)
        self.flag   = Circle((0,0), FLAG_R,  color='tab:green'); self.axw.add_patch(self.flag)
        self.player = Circle((0,0), PLAYER_R,color='tab:red');   self.axw.add_patch(self.player)
        self.rays = [self.axw.plot([], [], color='gray', alpha=0.4, lw=1)[0] for _ in RAY_ANGLES]
        self.heading, = self.axw.plot([], [], color='black', lw=2)
        self.axs.set_xlim(-0.5, N_SENSORS-0.5); self.axs.set_ylim(0, 1)
        self.bars = self.axs.bar(range(N_SENSORS), np.zeros(N_SENSORS),
                                 color=['tab:green']*N_RAYS + ['tab:red']*N_RAYS)
        self.axs.axvline(N_RAYS-0.5, color='k', lw=0.5)
        self.axs.set_title("sensors [flag | player]")

    def play(self, trail, gen, fit, step=2):
        self.axw.set_title(f"gen {gen}  best fit={fit:.1f}")
        for i in range(0, len(trail), step):
            ax_, ay_, ah_, fx_, fy_, px_, py_, obs_ = trail[i]
            self.agent.center  = (ax_, ay_)
            self.flag.center   = (fx_, fy_)
            self.player.center = (px_, py_)
            self.heading.set_data([ax_, ax_+np.cos(ah_)], [ay_, ay_+np.sin(ah_)])
            for line, ra in zip(self.rays, RAY_ANGLES):
                d = ah_ + ra
                line.set_data([ax_, ax_+MAX_RANGE*np.cos(d)],
                              [ay_, ay_+MAX_RANGE*np.sin(d)])
            for rect, v in zip(self.bars, obs_):
                rect.set_height(v)
            self.fig.canvas.draw_idle()
            plt.pause(0.005)

# ---------------- Viz ----------------
def animate(best, seed=12345, save=None):
    _, caps, trail = run_episode(best, seed=seed, record=True)
    print(f"Demo episode captures: {caps}")

    fig, (ax_world, ax_sens) = plt.subplots(1, 2, figsize=(12, 6),
                                            gridspec_kw={'width_ratios':[2,1]})
    ax_world.set_xlim(0, ARENA); ax_world.set_ylim(0, ARENA)
    ax_world.set_aspect('equal'); ax_world.set_title("Toy CTF — evolved CTRNN")
    agent_dot  = Circle((0,0), AGENT_R, color='tab:blue'); ax_world.add_patch(agent_dot)
    flag_dot   = Circle((0,0), FLAG_R,  color='tab:green'); ax_world.add_patch(flag_dot)
    player_dot = Circle((0,0), PLAYER_R,color='tab:red');   ax_world.add_patch(player_dot)
    ray_lines  = [ax_world.plot([], [], color='gray', alpha=0.4, lw=1)[0] for _ in RAY_ANGLES]
    heading_ln,= ax_world.plot([], [], color='black', lw=2)

    ax_sens.set_xlim(-0.5, N_SENSORS-0.5); ax_sens.set_ylim(0, 1)
    ax_sens.set_title("sensors  [flag×4 | player×4]")
    bar = ax_sens.bar(range(N_SENSORS), np.zeros(N_SENSORS),
                      color=['tab:green']*N_RAYS + ['tab:red']*N_RAYS)
    ax_sens.axvline(N_RAYS-0.5, color='k', lw=0.5)

    def update(i):
        ax_, ay_, ah_, fx_, fy_, px_, py_, obs_ = trail[i]
        agent_dot.center  = (ax_, ay_)
        flag_dot.center   = (fx_, fy_)
        player_dot.center = (px_, py_)
        heading_ln.set_data([ax_, ax_+np.cos(ah_)], [ay_, ay_+np.sin(ah_)])
        for line, ra in zip(ray_lines, RAY_ANGLES):
            d = ah_ + ra
            line.set_data([ax_, ax_+MAX_RANGE*np.cos(d)],
                          [ay_, ay_+MAX_RANGE*np.sin(d)])
        for rect, v in zip(bar, obs_):
            rect.set_height(v)
        return [agent_dot, flag_dot, player_dot, heading_ln, *ray_lines, *bar]

    anim = FuncAnimation(fig, update, frames=len(trail), interval=40, blit=False)
    if save:
        anim.save(save, fps=25)
    plt.tight_layout(); plt.show()

# ---------------- Main ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--pop",  type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None,
                    help="path to a saved .npy genome to seed the population")
    ap.add_argument("--watch-every", type=int, default=0,
                    help="animate the best of every Nth generation during evolution (0=off)")
    ap.add_argument("--out", type=str, default="best_genome.npy")
    args = ap.parse_args()

    pop  = args.pop  or (20 if args.quick else 50)
    gens = args.gens or (15 if args.quick else 80)

    print(f"Evolving: pop={pop} gens={gens} N_PARAMS={N_PARAMS}"
          + (f"  resume={args.resume}" if args.resume else "")
          + (f"  watch_every={args.watch_every}" if args.watch_every else ""))
    best, hist = evolve(pop=pop, gens=gens, seed=args.seed,
                       resume=args.resume, watch_every=args.watch_every)
    np.save(args.out, best)
    print(f"Saved {args.out}")

    plt.figure(); plt.plot(hist); plt.xlabel("gen"); plt.ylabel("best fitness")
    plt.title("evolution"); plt.tight_layout(); plt.savefig("fitness.png")
    print("Saved fitness.png")

    animate(best, seed=args.seed + 999)
