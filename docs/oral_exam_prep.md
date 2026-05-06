# Oral Exam Prep

The goal of the oral exam is to show that you understand the code and the modeling choices. You do not need to memorize every line. You should be able to explain the agent loop and trace where major pieces happen.

## Two-Minute Project Summary

I built a toy capture-the-flag environment to study minimal embodied cognition. The agent is controlled by a three-neuron continuous-time recurrent neural network. It receives simple ray-based sensory inputs, produces differential-drive motor outputs, and evolves through a genetic algorithm. The main question is whether such a tiny recurrent controller can learn multi-step behavior like finding a flag, carrying it home, and dealing with an opponent.

## Core Code Path

Start with `mini_ctf.py`.

- `run_match(...)`: runs one complete game between two genomes.
- `observation_for(...)`: turns world state into the agent's sensor vector.
- `step_agent(...)`: updates the CTRNN and converts neural outputs into movement.
- `maybe_pickup(...)`: checks whether an agent picked up the enemy flag.
- `maybe_score(...)`: checks whether a carrier returned to home base.
- `tag_carrier(...)`: resets a carrier if tagged by the opponent.
- `evaluate(...)`: tests each genome against opponents.
- `evolve(...)`: mutates, selects, and carries genomes across generations.

## Questions You Should Be Ready For

### What is a CTRNN?

A continuous-time recurrent neural network is a neural controller where neuron states change gradually over time. Each neuron receives recurrent input from the other neurons, bias input, and sensor input. Because the network is recurrent, its current state can depend on previous states, which gives it a simple form of memory-like dynamics.

### What does the genome encode?

The genome is a flat vector of numbers. It encodes recurrent weights, biases, time constants, and sensor weights. `unpack_genome(...)` turns that flat vector into the matrices and vectors used by the CTRNN.

### What are the sensors?

The agent has four directional rays. Each ray has channels for different object types, such as enemy flag, own flag, opponent, and home base. A sensor value is stronger when an object is closer and inside that ray's field of view.

### What are the actions?

The CTRNN outputs control left and right wheel speeds. The simulator uses those speeds as differential-drive movement, so unequal wheel speeds turn the agent.

### What is the fitness function doing?

Fitness rewards progress toward the current behavioral objective. If the agent is not carrying a flag, it is encouraged to seek the enemy flag. If it is carrying, it is encouraged to return to base. It also gets bonuses for pickup, scoring, and tagging, plus penalties for being tagged or timing out.

### Why use a hall of fame?

Self-play can forget older strategies. A hall of fame keeps earlier strong agents around as possible opponents, which makes evaluation less dependent on only the current population.

### What does a successful result look like?

A successful result would show increasing best fitness, nonzero or high win rates, and replays where agents actually approach flags, pick them up, and return home. A partial success is still useful if agents learn approach or pickup behavior but fail at return or defense.

### What are the biggest limitations?

The environment is still a toy world. The fitness function shapes behavior strongly. The genetic algorithm is noisy. Also, three neurons may simply be too few for robust capture-the-flag behavior.

## Good Places To Point In The Code

- `toy_ctf.py`: shared CTRNN and sensor helpers.
- `mini_ctf.py`: the main CTF environment.
- `docs/code_map.md`: the reading guide.
- `docs/melting_pot_bridge.md`: how the project could connect to Google's Melting Pot later.
