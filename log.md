# Research and development log

- [Research and development log](#research-and-development-log)
  - [2020.08.08 Sat](#20200808-sat)
    - [TODO](#todo)
    - [Thoughts](#thoughts)
    - [Gridworld transfer learning experiment](#gridworld-transfer-learning-experiment)
    - [TM related interesting links](#tm-related-interesting-links)

## 2020.08.08 Sat

### TODO

**Urgent**:

- [x] add ruamel.yaml to requirements
- [ ] describe config based building details
  - conventions
  - implementation
  - patches
- [x] update FAQ part on Terminology, Encoding, Planning
- [ ] update FAQ part on Parameters

Research + functional tasks

- [x] Adapt planning to goal-based strategy
  - [x] Switch from reward-based planning to goal-based
    - [x] Cut out reward from encoding and memorizing
    - [x] Track history of rewarding states and plan according to any of them
      - add naive list-based rewarding states tracking
  - [x] Test transfer learning capabilities
    - [x] Adapt environments for random initial states
    - [x] Adapt environments for random rewarding states
    - [x] Make the set of testing environments
    - [x] Adapt test runners for a multi-environment tests
    - [x] Make config for an experiment
    - [x] Run experiments
  - [x] Report results
    - [x] Update method description
    - [x] Add experiment results
    - [ ] TBD
- Not acknowledged and questionable:
  - [ ] Split SAR TM into 2 TMs
    - State TM: (s, a) $\rightarrow$ s'
    - Action TM: s $\rightarrow$ a
    - Direct external rewards aren't a thing
    - Reinforcement isn't tracked ATM
  - [ ] Investigate `MaxSegmentsPerCell` parameter impact
  - [ ] Implement integer encoder w/ overlapping buckets
    - overlapping should be a parameter
    - it defines the level of uncertainty
    - MDP planning becomes a light version of POMDP planning because of uncertainty
  - [ ] Investigate relation between overlapping and sufficient activation thresholds
  - [ ] Investigate `MaxSynapsesPerSegment` parameter impact
  - [ ] Start testing on POMDPs

Non-critical issues needing further investigation

Auxialiary tasks, usability improvements and so on

- [x] config based tests
  - [x] test config + builder classes
  - [x] improve config based building:
    - one config file for one test run (=all agents one test)
    - or even one config file for the whole experiment (=all agents all tests)
- [x] fine grained trace verbosity levels
- [x] setup release-based dev cycle
  - add tagging to git commits
  - how to add release notes
  - ?notes for major releases should contain algo details from FAQ
- [x] release v0.1 version of the SAR-based agent
- [ ] for v1.x
  - [ ] ? gym-like env interface
  - [ ] ? refactor envs and env generators (naming, names)
  - [ ] start live-logging
- [ ] for v2.x
  - [ ] remove legacy SAR-based parts
- [ ] extend Quick intro based on recent experience with students
  - [ ] add task on SP to the Readme
    - [x] fix entropy formula
    - [ ] add questions and requirements to test that learning is working
    - [ ] the same for boosting
  - [ ] update intro to TM part
    - [ ] mini-task on prediction
    - [ ] mini-task on backtracking

### Thoughts

- consider using SP between an input an TM
  - only states need SP, because actions and reward are just ints (naive encoding is enough)
  - concat them together
  - it will take care of sparsity
  - maybe smoothes the difference in size for a range of diff environments
    - bc even large envs may have a very small signal
- consider TD($\lambda$)-based approach from Sungur's work
- split SAR-based TM into State TM + Action TM
  - both has apical connections to each other
  - reward or goal-based approach? Could be interchangeable
- goal-based hierarchies of TM blocks
- SP hierarchies for large input images
  - with every SP working similar to convolution filter
- consider doing live-logging experiments in markdown there

### Gridworld transfer learning experiment

First results show that the agent with 1-goal goal list performs better than the agents with larger goal list size.

### TM related interesting links

- [Temporal Pooler](https://github.com/numenta/htmresearch/wiki/Overview-of-the-Temporal-Pooler)
  - a concept of the algo by htm.research
  - makes TM more robust to intra-sequence noise
  - general idea as I understand it - to add exponential averaging of activations over time
  - but.. found that on the forum (May 2020):
  > As far as I know, there is no official implementation of a "temporal pooler", just experimental research code. If you are talking about the union pooler logic in the research repo, I’m not sure anyone is actually working on this anymore.
- [Network API FAQ](https://github.com/htm-community/htm.core/blob/master/docs/NetworkAPI.md)
- [Tiebreak TM](https://github.com/htm-community/htm.core/blob/master/py/htm/advanced/algorithms/apical_tiebreak_temporal_memory.py)
  - basal + apical connections
  - hence two kinds of depolatizations
  - cell is depolarized iff
    - apical + basal depolarization
    - only basal and no apical at all
