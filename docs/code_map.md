# Code Map

This project is easiest to read as a sequence of increasingly complex environments that reuse the same tiny controller.

## The Brain

The CTRNN pieces live in `toy_ctf.py`.

- `N_NEURONS = 3`: the whole controller has only three recurrent neurons.
- `param_count(n_sensors)`: computes how many genome values are needed for a sensor layout.
- `unpack_genome(genome, n_sensors)`: turns a flat genome into recurrent weights, biases, time constants, and sensor weights.
- `sense_channels(...)`: casts four directional rays and returns one sensor value per ray per object channel.

Every later script imports these helpers instead of redefining the brain.

## Stage 1: `toy_ctf.py`

This is the solo training ground.

- One agent
- One flag
- One wandering distractor player
- Fitness mainly rewards getting close to the flag and touching it

Read this first if you want to understand the controller without the game logic fighting for attention.

## Intermediate: `race.py`

This is the first self-play step.

- Two agents
- One shared flag
- First agent to touch the flag wins
- Same 3-neuron CTRNN genome shape as stage 1

This is useful because it introduces opponent sensing without flag carrying or bases.

## Stage 2: `mini_ctf.py`

This is the current main implementation.

- Two agents
- Two flags
- Two bases
- Picking up the enemy flag
- Returning the enemy flag to home base
- Tagging a carrier sends them home and resets the stolen flag
- Self-play evolution with a hall of fame

Important functions:

- `run_match(...)`: the whole game loop for one match.
- `observation_for(...)`: builds the four sensor channels for one agent.
- `objective_state(...)`: decides whether an agent is seeking, returning, or recovering.
- `maybe_pickup(...)`: handles flag pickup.
- `maybe_score(...)`: handles returning a stolen flag to base.
- `tag_carrier(...)`: handles carrier reset after a tag.
- `evaluate(...)`: scores a population through self-play.
- `evolve(...)`: the genetic algorithm.
- `animate_match(...)`: live replay or GIF export.

## Archived: `archive/minictf_identical_agents.py`

This was an earlier simplified experiment. It is kept for reference, but it is not the main route now.

## Suggested Reading Order

1. Read the top constants in `toy_ctf.py`.
2. Read `sense_channels(...)` and `unpack_genome(...)`.
3. Skim `toy_ctf.py::run_episode(...)`.
4. Read `mini_ctf.py::run_match(...)`.
5. Read `mini_ctf.py::evaluate(...)` and `mini_ctf.py::evolve(...)`.
