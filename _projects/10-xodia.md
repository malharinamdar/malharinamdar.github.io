---
layout: page
title: Xodia
description: a multi-agent reinforcement learning competition environment
img: assets/img/xodia.png
importance: 2
category: previous
---

Xodia is a reinforcement learning environment I built for the PISB Credenz 2026 competition. Rather than solving an RL problem myself, the goal here was to *design the problem* — a set of environments that other participants would train their agents to solve, along with the engine and scoring system to run the whole contest fairly.

The theme is loosely based on Stranger Things, and it's split into three separate "universes", each a distinct RL task that gets progressively harder.

## The three universes

- **Universe 1 — Find Will:** a 20×20 grid where the agent has to navigate to a target while avoiding roaming enemies. The agent sees a compact 6-value observation describing its own position and how close the threats are.
- **Universe 2 — Escape Room:** the agent has to escape past patrolling guards. This one uses more involved reward shaping, with detection ranges and rewards that build up as the agent makes progress toward the exit.
- **Universe 3 — Fight Vecna:** the hardest of the three. The agent has to rescue two children while dodging an enemy, projectiles, and other hazards, working from a richer 12-value observation that tracks both the targets and the threats at once.

## How it's built

Everything runs on a custom game engine I wrote in Python (with NumPy, Pillow and OpenCV for the state handling and rendering). Each universe defines its own observation space and reward structure, and the framework is built around **Q-learning** with discretised states. Participants get a clean submission template and only have to implement their agent's logic — the engine handles the environment, stepping, and rendering.

The contest side is handled by a separate runner that evaluates each submission across all three universes (with a fixed step budget per environment) and adds up the best scores into a single cumulative total, so every entry is judged on the same footing.

It was a fun change of perspective — instead of being the one training an agent, I had to think carefully about reward design, what information to expose in the observation, and how to make the tasks challenging but fair.

code available at <a href="https://github.com/malharinamdar/credenz26-xodia">repo</a>
