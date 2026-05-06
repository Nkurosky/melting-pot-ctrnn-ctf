# Final Paper Outline

Target length: roughly 1500-2000 words plus figures.

## 1. Introduction

Goal: explain why this project is a model of cognition, not just a game.

Possible thesis:

This project tests whether a very small recurrent neural controller can evolve useful capture-the-flag behavior from limited perception, continuous action, and selection over repeated episodes.

Points to cover:

- Capture-the-flag requires perception, action selection, and context-sensitive behavior.
- The agent must do different things depending on state: seek a flag, carry it home, or respond to an opponent.
- A three-neuron CTRNN is intentionally minimal, so successes and failures are both informative.

## 2. Model

Goal: explain the computational system clearly enough that someone could reproduce the logic.

Subsections:

- Environment: continuous 2D arena, flags, bases, agents, tagging.
- Sensors: four rays, semantic channels, distance attenuation.
- Controller: three-neuron CTRNN, recurrent weights, biases, time constants.
- Actions: differential-drive movement.
- Evolution: population, mutation, selection, elitism, hall of fame.

Figure ideas:

- Arena diagram showing bases, flags, agents, and rays.
- CTRNN diagram with three recurrent neurons and sensor inputs.

## 3. Experiments

Goal: describe what was run and why.

Experiments:

- Stage 1 solo flag capture.
- Stage 2 symmetric mini-CTF.
- Optional: race baseline with two agents and one shared flag.

Measures:

- Best fitness over generations.
- Mean fitness over generations.
- Win rate of best agent.
- Qualitative replay observations.

## 4. Results

Goal: show what the model learned.

Points to look for:

- Does fitness improve from early to late generations?
- Do win rates become nonzero or approach 1.0 in some generations?
- Do replays show flag approach, pickup, return, tagging, circling, wall-hugging, or failure modes?
- Does the all-time best genome behave better than the final-generation genome?

Figure ideas:

- Fitness curve from `mini_ctf_fitness.png`.
- Screenshot sequence or GIF frames from a replay.
- Table comparing early HoF, late HoF, final best, and all-time best.

## 5. Discussion

Goal: interpret the model as cognitive science.

Questions:

- What behaviors emerged without being directly programmed?
- What does recurrence add, given that the controller is so small?
- Where does the three-neuron architecture seem insufficient?
- How much of the behavior comes from the controller versus the shaped fitness function?
- What would need to change to scale toward Google's Melting Pot?

## 6. Limitations and Future Work

Likely limitations:

- Toy environment is much simpler than real Melting Pot.
- Fitness shaping influences what can be learned.
- Self-play can be unstable.
- The saved final genome may not be the best all-time genome, so all-time tracking matters.
- Three neurons may be too few for robust multi-phase CTF strategy.

Future work:

- Add curriculum stages.
- Improve evaluation with fixed test seeds.
- Build a Melting Pot observation/action adapter.
- Compare three neurons against larger CTRNNs.

## 7. Conclusion

End by returning to the central question: what can a minimal recurrent controller learn when cognition is treated as embodied action in an environment?
