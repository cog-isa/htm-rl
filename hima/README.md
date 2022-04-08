# Project details

- [Project details](#project-details)
  - [Experiment setting](#experiment-setting)
  - [Configuration based building](#configuration-based-building)

## Experiment setting

Quote from our paper:

> In our experiments, we used classic grid world environments, which were represented as mazes on a square grid. Each state can be defined by an agent's position; thus, the state space $S$ contains all possible agent positions. An agent begins in a fixed state $s_0$, with the goal of locating a single reward in a fixed position $s_g$. The environment's transition function is deterministic. The action space is made up of four actions that move the agent to each adjacent grid cell. However, when the agent attempts to move into a maze wall, the position of the agent remains unchanged. It is assumed that the maze is surrounded by obstacles, making it impossible for an agent to move outside. Every timestep, an agent receives an observation -- a binary image of a small square window encircling it (we used $5 \times 5$ size with an agent being at its center). The observation has six binary channels: three color channels for the floor, walls, out-of-bounds obstacles, and the goal. We use maze floor coloring to add semantic clues to observations. When an agent achieves the goal state $s_g$, it receives a large positive value $+1.0$. Each step is also slightly punished with $-0.002$, causing an agent to seek the shortest path. The optimal path from the starting point to the goal in each testing environment was between 8 and 20 steps. We also set a time limit for episodes: 200 for an 8x8 environment and 350 for a 12x12 environment.

## Configuration based building

This section describes configs syntax, rules and how to use it.

We use [YAML 1.2](https://yaml.org/spec/1.2/spec.html) format to represent configs and parse them with [ruamel.yaml](https://yaml.readthedocs.io/en/latest/overview.html) python package.

If you're new to YAML, check out [design section](https://en.wikipedia.org/wiki/YAML#Design) on Wikipedia - it provides required basics of the format in a very short form.

___
_Here's a little side note covering all information sources:_

For more details on YAML format we encourage you to use [1.2 standard specification](https://yaml.org/spec/1.2/spec.html).

`ruamel.yaml` itself has a very shallow [documentation](https://yaml.readthedocs.io/en/latest/overview.html) which is not of a much use. But, since it's a fork of PyYAML package, PyYAML's [docs](https://pyyaml.org/wiki/PyYAMLDocumentation) are mostly applicable as well. Both packages have slightly different API and internals.

There're also some useful answers on Stackoverflow from the author of `ruamel.yaml` (mostly in questions on PyYAML). And the last bastion of truth is, of course, `ruamel.yaml` code sources.
