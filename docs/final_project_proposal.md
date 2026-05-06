# Final Project Proposal

## Title

Evolving a Minimal Neural Agent for Capture-the-Flag

## Project Description

For my final project, I will build and analyze a computational model of cognition in which a very small neural controller learns to act in a capture-the-flag environment. The agent is controlled by a continuous-time recurrent neural network (CTRNN) with only three neurons. Rather than hand-coding a strategy, I evolve the network parameters with a genetic algorithm and evaluate whether simple neural dynamics can produce useful behaviors such as orienting toward a flag, avoiding or tagging an opponent, carrying a flag, and returning it to a home base.

The project starts with a simple toy environment: one agent, one flag, and a distractor player. This first stage lets me test the basic agent loop: sensors, neural dynamics, motor outputs, movement, and fitness. The second stage extends the task to a symmetric mini capture-the-flag game with two agents, two flags, and two bases. In this environment, an agent must find the opponent's flag, carry it home, and respond to the possibility of being tagged.

The cognitive question is how much adaptive behavior can emerge from a very small recurrent controller when the environment gives only limited sensory information. The model is interesting because it connects perception, action, memory-like recurrent dynamics, and learning through selection. The agent does not receive an explicit rule like "if carrying flag, go home." Instead, the genetic algorithm searches for CTRNN parameters that make useful behavior emerge from the interaction between the agent and environment.

## Model Components

- Environment: a continuous 2D capture-the-flag world.
- Sensors: four directional rays with separate channels for objects such as flags, opponents, and bases.
- Controller: a three-neuron CTRNN with recurrent connections, biases, time constants, and sensor weights.
- Actions: differential-drive movement from left and right motor outputs.
- Learning: a genetic algorithm with mutation, tournament-style selection, elitism, and a hall of fame.

## Planned Analysis

I will compare behavior across stages of the task. First, I will show whether the agent learns to reach a single flag. Then I will analyze whether evolved agents in the mini-CTF setting learn partial or complete capture-the-flag behavior. I will use training curves, win rates, replay visualizations, and qualitative behavior analysis to describe what the agents learn and where they fail. A central question will be whether a three-neuron brain is expressive enough for the full task or whether the task exposes the limits of this minimal architecture.

## Expected Final Submission

The final paper will describe the model, explain the evolutionary setup, present results with figures, and discuss what the agent's successes and failures suggest about minimal cognition. The accompanying notebook will run the model, generate plots, and show replay examples that I can explain during the oral exam.
