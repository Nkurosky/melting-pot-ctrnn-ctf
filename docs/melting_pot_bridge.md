# Melting Pot Bridge Plan

The current project is a toy simulator. The goal is to eventually keep the 3-neuron CTRNN controller but replace the toy world with Google's Melting Pot capture-the-flag substrate.

## What We Have Now

The current controller expects a small numeric observation vector:

- Four directional rays
- Multiple semantic channels per ray
- Channel examples: enemy flag, own flag, opponent, home base

The controller outputs two continuous values:

- Left wheel speed
- Right wheel speed

The toy simulator uses those outputs directly as differential drive movement.

## What Melting Pot Will Need

Melting Pot environments usually expose richer observations and expect discrete actions. The bridge needs two adapters.

## Adapter 1: Observation Adapter

Purpose: turn a Melting Pot observation into the same compact sensor vector used by the CTRNN.

Likely inputs:

- RGB observation
- Symbolic/object layers if the substrate exposes them
- The controlled player's orientation and position if available

Likely output:

- A NumPy vector matching `mini_ctf.N_SENSORS`
- Same channel ordering as `mini_ctf.py`

First version:

- Detect approximate flag locations
- Detect nearby players
- Detect home/enemy base cues
- Project those objects into the four ray sensors

Review checkpoint:

- Before adding learning, run the adapter on saved observations and print/plot the resulting sensor vector.

## Adapter 2: Action Adapter

Purpose: turn CTRNN motor outputs into Melting Pot discrete actions.

Current output:

- `vl`: left wheel speed
- `vr`: right wheel speed

Possible discrete mapping:

- both forward: move forward
- left slower than right: turn left
- right slower than left: turn right
- both low or opposing: stay/no-op

Review checkpoint:

- Make a tiny table of motor-output ranges to discrete actions.
- Test it independently before connecting to Melting Pot.

## Integration Order

1. Keep training in toy `mini_ctf.py` until replays show basic steal-return behavior.
2. Create `melting_pot_adapter.py` with observation and action conversion only.
3. Write a small script that runs a random or saved genome inside Melting Pot without evolution.
4. Add logging that records raw observations, adapted sensors, CTRNN outputs, and chosen actions.
5. Only then reintroduce evolution against the Melting Pot substrate.

## Main Risk

The 3-neuron controller may be too small for full Melting Pot CTF. That is not failure; it is the experiment. The staged setup exists so we can see exactly where the minimal controller stops being enough.
