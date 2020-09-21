# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

- git LFS support
- rich yaml configs based building
- gridworld maps generator
- transfer learning experiments setup
- shift to the goal based agent
  - remove rewards from encoding scheme and memorization

## [0.1] - 2020.07.24

Naive agent based on htm framework described in [report](./reports/v0_1/report.md). Key features:

- memorizes all transitions (r, s, a) -> (r', s', a') with a single [single- or multicolumn] Temporal Memory
  - (r, s, a) triplets are encoded into single (s, a, r) SDR
  - every part of SDR is encoded with naive integer encoder without overlaps then concatenated together
- can infer policy [a1, a2, .. aT] to the rewarding state if it's in the radius N of memorized transitions
  - planning horizon is a hyperparameter
- make random action if planner fail to make a plan
  - with planning horizon = 0 it degrades to random agent

Agent was tested on three gridworld MDPs (multi_way_v0-2) with different planning horizon and compared with random agent and simple DQN.

Key results:

- learns (=progresses) faster than DQN
- even planning horizon 1 is better than random
  - with fixed planning horizon N advantage diminishes as environment complexity grows
- if planning horizon N is enough to plan to the reward from the initial state, it works perfect after very small number of training episodes (~ equal to the distance to the goal state)

[unreleased]: https://github.com/cog-isa/htm_rl/compare/v0.1...HEAD
[0.1]: https://github.com/cog-isa/htm_rl/releases/tag/v0.1
