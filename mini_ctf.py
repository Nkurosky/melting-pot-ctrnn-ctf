"""
Stage 2 - symmetric mini-CTF.

Two agents, two flags, two bases. Each agent must steal the enemy flag and
return it to its own base. If agents collide while one is carrying a flag,
the carrier is sent back home and the stolen flag resets.

The controller stays "classic minimal ER":
- 3 fully recurrent CTRNN neurons
- differential drive
- directional ray sensors

Sensor channels per ray:
0 = enemy flag
1 = own flag
2 = opponent agent
3 = home base

Run:
    python mini_ctf.py
    python mini_ctf.py --quick --no-animate
    python mini_ctf.py --seed-from best_genome.npy
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from toy_ctf import (
    ARENA,
    DT,
    AGENT_R,
    FLAG_R,
    CAPTURE_DIST,
    WHEEL_BASE,
    MAX_SPEED,
    N_NEURONS,
    N_RAYS,
    N_SENSORS as STAGE1_SENSORS,
    BASE_PARAMS,
    sigmoid,
    sense_channels,
    unpack_genome,
    param_count,
)

EP_STEPS = 500
BASE_R = 1.2
BASE_X_OFFSET = 2.5
TAG_DIST = 0.9

CHAN_ENEMY_FLAG = 0
CHAN_OWN_FLAG = 1
CHAN_OPPONENT = 2
CHAN_HOME_BASE = 3
N_CHANNELS = 4
N_SENSORS = N_RAYS * N_CHANNELS
N_PARAMS = param_count(N_SENSORS)
STAGE1_PARAMS = param_count(STAGE1_SENSORS)

FIT_SEEK = 1.0
FIT_RETURN = 1.5
FIT_RECOVER = 1.2
PROGRESS_GAIN = 4.0
STEAL_BONUS = 35.0
SCORE_BONUS = 220.0
TAG_BONUS = 60.0
TAGGED_PENALTY = 55.0
TIMEOUT_PENALTY = 10.0

TEAM_LABELS = ("A", "B")
TEAM_COLORS = ("tab:blue", "tab:orange")
FLAG_COLORS = ("tab:cyan", "tab:red")


def base_pos(team):
    x = BASE_X_OFFSET if team == 0 else ARENA - BASE_X_OFFSET
    return np.array([x, ARENA / 2.0], dtype=float)


def spawn_pose(rng, team):
    base = base_pos(team)
    x = base[0] + (1.5 if team == 0 else -1.5) + rng.normal(0, 0.2)
    y = np.clip(base[1] + rng.normal(0, 2.0), 3.0, ARENA - 3.0)
    h = (0.0 if team == 0 else np.pi) + rng.normal(0, 0.45)
    return float(np.clip(x, AGENT_R, ARENA - AGENT_R)), float(y), float(h)


def init_flag(team):
    home = base_pos(team)
    return {
        "team": team,
        "home": home.copy(),
        "x": float(home[0]),
        "y": float(home[1]),
        "carrier": None,
    }


def reset_flag(flag):
    flag["carrier"] = None
    flag["x"] = float(flag["home"][0])
    flag["y"] = float(flag["home"][1])


def init_agent(genome, team, rng):
    W, b, taus, Ws = unpack_genome(genome, N_SENSORS)
    x, y, h = spawn_pose(rng, team)
    return {
        "team": team,
        "base": base_pos(team),
        "x": x,
        "y": y,
        "h": h,
        "carrying": None,
        "y_state": np.zeros(N_NEURONS),
        "W": W,
        "b": b,
        "taus": taus,
        "Ws": Ws,
    }


def reset_agent(agent, rng):
    agent["x"], agent["y"], agent["h"] = spawn_pose(rng, agent["team"])
    agent["carrying"] = None
    agent["y_state"].fill(0.0)


def step_agent(agent, obs):
    sig = sigmoid(agent["y_state"] + agent["b"])
    I = agent["Ws"].T @ obs
    dy = (-agent["y_state"] + agent["W"] @ sig + I) / agent["taus"]
    agent["y_state"] = agent["y_state"] + DT * dy
    out = sigmoid(agent["y_state"] + agent["b"])
    vl = (2.0 * out[1] - 1.0) * MAX_SPEED
    vr = (2.0 * out[2] - 1.0) * MAX_SPEED
    v = 0.5 * (vl + vr)
    w = (vr - vl) / WHEEL_BASE
    return v, w


def flag_position(flag, agents):
    if flag["carrier"] is None:
        return np.array([flag["x"], flag["y"]], dtype=float)
    carrier = agents[flag["carrier"]]
    return np.array([carrier["x"], carrier["y"]], dtype=float)


def observation_for(agent, opponent, own_flag, enemy_flag, agents):
    own_flag_pos = flag_position(own_flag, agents)
    enemy_flag_pos = flag_position(enemy_flag, agents)
    return sense_channels(
        agent["x"],
        agent["y"],
        agent["h"],
        [
            [tuple(enemy_flag_pos)],
            [tuple(own_flag_pos)],
            [(opponent["x"], opponent["y"])],
            [tuple(agent["base"])],
        ],
    )


def objective_state(team, agents, flags):
    agent = agents[team]
    if agent["carrying"] is not None:
        return ("return",), agent["base"], FIT_RETURN
    own_flag = flags[team]
    if own_flag["carrier"] is not None and own_flag["carrier"] != team:
        return ("recover", own_flag["carrier"]), flag_position(own_flag, agents), FIT_RECOVER
    enemy_flag = flags[1 - team]
    return ("seek", enemy_flag["team"]), flag_position(enemy_flag, agents), FIT_SEEK


def apply_motion(agent, v, w):
    agent["h"] = (agent["h"] + w * DT + np.pi) % (2.0 * np.pi) - np.pi
    agent["x"] = float(np.clip(agent["x"] + v * np.cos(agent["h"]) * DT, AGENT_R, ARENA - AGENT_R))
    agent["y"] = float(np.clip(agent["y"] + v * np.sin(agent["h"]) * DT, AGENT_R, ARENA - AGENT_R))


def maybe_pickup(team, agents, flags, fitness):
    agent = agents[team]
    enemy_flag = flags[1 - team]
    if agent["carrying"] is not None or enemy_flag["carrier"] is not None:
        return False
    fx, fy = flag_position(enemy_flag, agents)
    if np.hypot(agent["x"] - fx, agent["y"] - fy) >= CAPTURE_DIST:
        return False
    enemy_flag["carrier"] = team
    agent["carrying"] = enemy_flag["team"]
    fitness[team] += STEAL_BONUS
    return True


def maybe_score(team, agents, flags, fitness, t):
    agent = agents[team]
    if agent["carrying"] is None:
        return False
    if np.hypot(agent["x"] - agent["base"][0], agent["y"] - agent["base"][1]) >= BASE_R:
        return False
    stolen_team = agent["carrying"]
    reset_flag(flags[stolen_team])
    agent["carrying"] = None
    fitness[team] += SCORE_BONUS + 0.4 * (EP_STEPS - t)
    fitness[1 - team] -= 80.0
    return True


def tag_carrier(carrier_team, tagger_team, agents, flags, fitness, rng):
    carrier = agents[carrier_team]
    stolen_team = carrier["carrying"]
    if stolen_team is None:
        return False
    reset_flag(flags[stolen_team])
    reset_agent(carrier, rng)
    fitness[carrier_team] -= TAGGED_PENALTY
    fitness[tagger_team] += TAG_BONUS
    return True


def snapshot(agents, flags):
    f0 = flag_position(flags[0], agents)
    f1 = flag_position(flags[1], agents)
    return (
        agents[0]["x"], agents[0]["y"], agents[0]["h"], int(agents[0]["carrying"] is not None),
        agents[1]["x"], agents[1]["y"], agents[1]["h"], int(agents[1]["carrying"] is not None),
        float(f0[0]), float(f0[1]), -1 if flags[0]["carrier"] is None else int(flags[0]["carrier"]),
        float(f1[0]), float(f1[1]), -1 if flags[1]["carrier"] is None else int(flags[1]["carrier"]),
    )


def run_match(gen_a, gen_b, seed, record=False):
    rng = np.random.default_rng(seed)
    agents = [init_agent(gen_a, 0, rng), init_agent(gen_b, 1, rng)]
    flags = [init_flag(0), init_flag(1)]
    fitness = np.zeros(2, dtype=float)
    prev_goal_key = [None, None]
    prev_goal_dist = [None, None]
    trail = []
    winner = "timeout"

    for t in range(EP_STEPS):
        obs_a = observation_for(agents[0], agents[1], flags[0], flags[1], agents)
        obs_b = observation_for(agents[1], agents[0], flags[1], flags[0], agents)
        v_a, w_a = step_agent(agents[0], obs_a)
        v_b, w_b = step_agent(agents[1], obs_b)

        apply_motion(agents[0], v_a, w_a)
        apply_motion(agents[1], v_b, w_b)

        maybe_pickup(0, agents, flags, fitness)
        maybe_pickup(1, agents, flags, fitness)

        if maybe_score(0, agents, flags, fitness, t):
            winner = TEAM_LABELS[0]
            if record:
                trail.append(snapshot(agents, flags))
            break
        if maybe_score(1, agents, flags, fitness, t):
            winner = TEAM_LABELS[1]
            if record:
                trail.append(snapshot(agents, flags))
            break

        if np.hypot(agents[0]["x"] - agents[1]["x"], agents[0]["y"] - agents[1]["y"]) < TAG_DIST:
            tag_carrier(0, 1, agents, flags, fitness, rng)
            tag_carrier(1, 0, agents, flags, fitness, rng)

        for team in (0, 1):
            goal_key, goal_pos, goal_weight = objective_state(team, agents, flags)
            d = np.hypot(agents[team]["x"] - goal_pos[0], agents[team]["y"] - goal_pos[1])
            fitness[team] -= goal_weight * d * DT
            if prev_goal_key[team] == goal_key and prev_goal_dist[team] is not None:
                fitness[team] += goal_weight * PROGRESS_GAIN * (prev_goal_dist[team] - d)
            prev_goal_key[team] = goal_key
            prev_goal_dist[team] = d

        if record:
            trail.append(snapshot(agents, flags))
    else:
        fitness -= TIMEOUT_PENALTY

    return fitness[0], fitness[1], winner, trail


def evaluate(genomes, hof, n_peers=4, n_hof=2, seed=0):
    pop = len(genomes)
    rng = np.random.default_rng(seed)
    fits = np.zeros(pop)
    wins = np.zeros(pop)
    losses = np.zeros(pop)
    matches = np.zeros(pop)

    for i in range(pop):
        peers = [j for j in range(pop) if j != i]
        opponents = rng.choice(peers, min(n_peers, len(peers)), replace=False).tolist()
        opp_genomes = [genomes[j] for j in opponents]
        if hof:
            k = min(n_hof, len(hof))
            hof_idx = rng.choice(len(hof), k, replace=False).tolist()
            opp_genomes += [hof[j] for j in hof_idx]

        for j, opp in enumerate(opp_genomes):
            s = (seed * 1_000_003 + i * 1009 + j * 17) & 0xFFFFFFFF
            fa, _, winner, _ = run_match(genomes[i], opp, seed=s)
            fits[i] += fa
            matches[i] += 1
            if winner == TEAM_LABELS[0]:
                wins[i] += 1
            elif winner == TEAM_LABELS[1]:
                losses[i] += 1

    fits /= np.maximum(matches, 1)
    winrate = wins / np.maximum(matches, 1)
    return fits, winrate, wins, losses


def widen_stage1_genome(genome):
    widened = np.zeros(N_PARAMS, dtype=float)
    widened[:BASE_PARAMS] = genome[:BASE_PARAMS]
    old_ws = genome[BASE_PARAMS:].reshape(STAGE1_SENSORS, N_NEURONS)
    new_ws = np.zeros((N_SENSORS, N_NEURONS), dtype=float)

    old_flag = old_ws[:N_RAYS]
    old_player = old_ws[N_RAYS:]
    new_ws[CHAN_ENEMY_FLAG * N_RAYS:(CHAN_ENEMY_FLAG + 1) * N_RAYS] = old_flag
    new_ws[CHAN_OPPONENT * N_RAYS:(CHAN_OPPONENT + 1) * N_RAYS] = old_player
    new_ws[CHAN_OWN_FLAG * N_RAYS:(CHAN_OWN_FLAG + 1) * N_RAYS] = 0.35 * old_player
    new_ws[CHAN_HOME_BASE * N_RAYS:(CHAN_HOME_BASE + 1) * N_RAYS] = 0.25 * old_flag
    widened[BASE_PARAMS:] = new_ws.ravel()
    return widened


def load_seed_genome(path, index=None):
    genome = np.load(path)
    if genome.ndim == 2 and genome.shape[1] == N_PARAMS:
        selected = -1 if index is None else index
        if not -len(genome) <= selected < len(genome):
            raise IndexError(f"HoF index {selected} is outside 0..{len(genome) - 1}")
        print(f"Loaded HoF genome {selected} from {path} ({len(genome)} entries)")
        return genome[selected], f"mini-ctf-hof[{selected}]"
    if genome.shape == (N_PARAMS,):
        return genome, "mini-ctf"
    if genome.shape == (STAGE1_PARAMS,):
        return widen_stage1_genome(genome), "stage-1-widened"
    raise ValueError(
        f"Unsupported genome shape {genome.shape}; expected {(N_PARAMS,)}, "
        f"{(STAGE1_PARAMS,)}, or a HoF array with {N_PARAMS} columns"
    )


def evolve(pop=40, gens=100, mut=0.15, elite=2, seed=0,
           resume=None, seed_from=None, watch_every=0, hof_every=10, hof_cap=12):
    rng = np.random.default_rng(seed)
    if resume:
        g0, mode = load_seed_genome(resume)
        jitter = 0.05
        print(f"Resumed from {resume} ({mode})")
        genomes = np.tile(g0, (pop, 1)) + rng.normal(0, jitter, (pop, N_PARAMS))
        genomes[0] = g0
    elif seed_from:
        g0, mode = load_seed_genome(seed_from)
        jitter = 0.12 if mode == "mini-ctf" else 0.18
        print(f"Warm-started from {seed_from} ({mode})")
        genomes = np.tile(g0, (pop, 1)) + rng.normal(0, jitter, (pop, N_PARAMS))
        genomes[0] = g0
    else:
        genomes = rng.normal(0, 1.0, size=(pop, N_PARAMS))

    hof = []
    best_hist = []
    wr_hist = []
    viewer = _LiveViewer() if watch_every > 0 else None

    for g in range(gens):
        fits, winrate, _, _ = evaluate(genomes, hof, seed=g + 1)
        order = np.argsort(-fits)
        genomes = genomes[order]
        fits = fits[order]
        winrate = winrate[order]
        best_hist.append(fits[0])
        wr_hist.append(winrate[0])

        print(
            f"gen {g:3d}  best={fits[0]:7.1f}  mean={fits.mean():7.1f}  "
            f"winrate(best)={winrate[0]:.2f}  hof={len(hof)}"
        )

        if g % hof_every == 0:
            hof.append(genomes[0].copy())
            if len(hof) > hof_cap:
                hof.pop(0)

        if viewer is not None and (g % watch_every == 0 or g == gens - 1):
            rival = genomes[1] if len(genomes) > 1 else genomes[0]
            _, _, winner, trail = run_match(genomes[0], rival, seed=g * 131 + 7, record=True)
            viewer.play(trail, gen=g, fit=fits[0], winner=winner)

        new_genomes = [genomes[i].copy() for i in range(elite)]
        while len(new_genomes) < pop:
            idxs = rng.choice(pop, 3, replace=False)
            parent = genomes[min(idxs)]
            child = parent + rng.normal(0, mut, size=N_PARAMS)
            new_genomes.append(child)
        genomes = np.array(new_genomes)

    return genomes[0], best_hist, wr_hist, hof


class _LiveViewer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7.0, 6.5))
        self.ax.set_xlim(0, ARENA)
        self.ax.set_ylim(0, ARENA)
        self.ax.set_aspect("equal")
        self.ax.axvline(ARENA / 2.0, color="0.75", ls="--", lw=1)

        self.base_a = Circle(tuple(base_pos(0)), BASE_R, fill=False, ec=TEAM_COLORS[0], lw=2)
        self.base_b = Circle(tuple(base_pos(1)), BASE_R, fill=False, ec=TEAM_COLORS[1], lw=2)
        self.ax.add_patch(self.base_a)
        self.ax.add_patch(self.base_b)

        self.agent_a = Circle((0, 0), AGENT_R, color=TEAM_COLORS[0])
        self.agent_b = Circle((0, 0), AGENT_R, color=TEAM_COLORS[1])
        self.flag_a = Circle((0, 0), FLAG_R, color=FLAG_COLORS[0], alpha=0.85)
        self.flag_b = Circle((0, 0), FLAG_R, color=FLAG_COLORS[1], alpha=0.85)
        for patch in (self.agent_a, self.agent_b, self.flag_a, self.flag_b):
            self.ax.add_patch(patch)

        self.h_a, = self.ax.plot([], [], "k-", lw=1.5)
        self.h_b, = self.ax.plot([], [], "k-", lw=1.5)
        self.carry_a, = self.ax.plot([], [], marker="o", color="k", ms=4, ls="")
        self.carry_b, = self.ax.plot([], [], marker="o", color="k", ms=4, ls="")

    def set_frame(self, frame):
        ax_, ay_, ah_, a_carry, bx_, by_, bh_, b_carry, f0x, f0y, _, f1x, f1y, _ = frame
        self.agent_a.center = (ax_, ay_)
        self.agent_b.center = (bx_, by_)
        self.flag_a.center = (f0x, f0y)
        self.flag_b.center = (f1x, f1y)
        self.h_a.set_data([ax_, ax_ + np.cos(ah_)], [ay_, ay_ + np.sin(ah_)])
        self.h_b.set_data([bx_, bx_ + np.cos(bh_)], [by_, by_ + np.sin(bh_)])
        self.carry_a.set_data([ax_], [ay_ + 0.6]) if a_carry else self.carry_a.set_data([], [])
        self.carry_b.set_data([bx_], [by_ + 0.6]) if b_carry else self.carry_b.set_data([], [])
        return [
            self.agent_a,
            self.agent_b,
            self.flag_a,
            self.flag_b,
            self.h_a,
            self.h_b,
            self.carry_a,
            self.carry_b,
        ]

    def play(self, trail, gen, fit, winner, step=2):
        self.ax.set_title(f"gen {gen}  fit={fit:.1f}  winner={winner}")
        for frame in trail[::step]:
            self.set_frame(frame)
            self.fig.canvas.draw_idle()
            plt.pause(0.005)


def animate_match(gen_a, gen_b, seed=42, save=None, show=True):
    _, _, winner, trail = run_match(gen_a, gen_b, seed=seed, record=True)
    print(f"winner: {winner}")
    viewer = _LiveViewer()
    plt.ioff()
    viewer.ax.set_title(f"replay - winner={winner}")
    if save:
        frames = trail[::2]

        def update(i):
            return viewer.set_frame(frames[i])

        anim = FuncAnimation(viewer.fig, update, frames=len(frames), interval=40, blit=False)
        anim.save(save, writer="pillow", fps=25)
        print(f"Saved replay to {save}")
    if show:
        viewer.play(trail, gen=-1, fit=0.0, winner=winner, step=2)
        plt.show()
    else:
        plt.close(viewer.fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--seed-from", dest="seed_from", type=str, default=None)
    ap.add_argument("--watch-every", type=int, default=0)
    ap.add_argument("--out", type=str, default="best_mini_ctf.npy")
    ap.add_argument("--no-animate", action="store_true")
    ap.add_argument("--replay", type=str, default=None,
                    help="load a saved genome and replay a match instead of training")
    ap.add_argument("--replay-index", type=int, default=None,
                    help="if --replay is a hall-of-fame array, select this entry")
    ap.add_argument("--opponent", type=str, default=None,
                    help="optional saved opponent genome for --replay")
    ap.add_argument("--opponent-index", type=int, default=None,
                    help="if --opponent is a hall-of-fame array, select this entry")
    ap.add_argument("--save-replay", type=str, default=None,
                    help="write a replay GIF, for example replay.gif")
    args = ap.parse_args()

    if args.replay:
        gen_a, _ = load_seed_genome(args.replay, index=args.replay_index)
        gen_b, _ = load_seed_genome(args.opponent, index=args.opponent_index) if args.opponent else (gen_a, "self")
        animate_match(gen_a, gen_b, seed=args.seed + 999, save=args.save_replay, show=not args.no_animate)
        raise SystemExit(0)

    pop = args.pop or (16 if args.quick else 40)
    gens = args.gens or (10 if args.quick else 100)

    print(
        f"Evolving mini-CTF: pop={pop} gens={gens} N_PARAMS={N_PARAMS}"
        + (f"  resume={args.resume}" if args.resume else "")
        + (f"  seed_from={args.seed_from}" if args.seed_from else "")
    )
    best, fits, wrs, hof = evolve(
        pop=pop,
        gens=gens,
        seed=args.seed,
        resume=args.resume,
        seed_from=args.seed_from,
        watch_every=args.watch_every,
    )
    np.save(args.out, best)
    np.save(args.out.replace(".npy", "_hof.npy"), np.array(hof))
    print(f"Saved {args.out} and hall-of-fame.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(fits)
    ax1.set_xlabel("gen")
    ax1.set_ylabel("best fitness")
    ax1.set_title("mini-CTF fitness")
    ax2.plot(wrs)
    ax2.set_xlabel("gen")
    ax2.set_ylabel("best winrate")
    ax2.set_title("mini-CTF winrate")
    plt.tight_layout()
    plt.savefig("mini_ctf_fitness.png")
    print("Saved mini_ctf_fitness.png")

    if args.save_replay or not args.no_animate:
        animate_match(
            best,
            hof[-1] if hof else best,
            seed=args.seed + 999,
            save=args.save_replay,
            show=not args.no_animate,
        )
