# Melting Pot CTRNN Capture the Flag

Minimal evolutionary robotics experiments for capture-the-flag with a 3-neuron CTRNN controller.

This repo currently contains two main stages:

- `toy_ctf.py`: stage 1, a solo toy environment with one agent, one target flag, and a distractor player.
- `mini_ctf.py`: stage 2, a symmetric mini-CTF with two agents, two flags, two bases, flag carrying, and tag-to-reset.

There is also an intermediate self-play script:

- `race.py`: two agents race to a single central flag.

Older experiments live in `archive/`.

## Core Setup

All controllers use the same "classic minimal ER" setup:

- 3 fully recurrent CTRNN neurons
- Differential drive motor outputs
- Directional ray sensors
- Simple genetic algorithm with elitism and Gaussian mutation

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

Stage 1:

```bash
python toy_ctf.py
python toy_ctf.py --quick
```

Stage 2:

```bash
python mini_ctf.py
python mini_ctf.py --quick --no-animate
python mini_ctf.py --seed-from best_genome.npy
python mini_ctf.py --replay best_mini_ctf.npy
python mini_ctf.py --replay best_mini_ctf.npy --save-replay replay.gif --no-animate
```

Intermediate race baseline:

```bash
python race.py
python race.py --quick
```

## Review Guide

Start with these docs before reading the scripts:

- `docs/code_map.md`: a gentle map of the main files and where the important logic lives
- `docs/melting_pot_bridge.md`: the plan for moving from toy simulation to Google's Melting Pot substrate

## Files

- `toy_ctf.py`: shared CTRNN pieces plus the original solo training setup
- `mini_ctf.py`: symmetric mini-CTF environment and co-evolution loop
- `race.py`: single-flag competitive self-play baseline
- `best_genome.npy`, `best_mini_ctf.npy`: saved example champions
- `fitness.png`, `mini_ctf_fitness.png`, `demo.png`: generated artifacts

## Notes

- `mini_ctf.py` can warm-start from a stage-1 genome by widening the old sensor weights into the new 4-channel sensor layout.
- The current mini-CTF scoring rule awards a point when an agent returns the enemy flag to its home base. It does not yet require the home flag to be present.
- This repo is still in the toy-simulation phase. The later goal is to bridge these controllers into Google's Melting Pot CTF substrate.

## Suggested Next Steps

- Add richer curricula so evolution does not stall in local optima
- Compare multiple evolutionary runs and hall-of-fame variants
- Build an observation/action bridge for the real Melting Pot environment
