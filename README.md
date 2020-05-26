# htm_rl

Repo [link](https://github.com/cog-isa/htm-rl).

## TODO

- [x] repo setup
- [x] check htm works on any hello world example
- [x] rl env
  - [ ] breakout for early testing
  - [x] chase the target as simpler alternative
    - [x] choose base env
- [x] check env works
  - [x] naive random
- [ ] conda env config
- [ ] build up htm & neuroscience knowledge
  - [x] refresh `htm school` videos
  - [x] get used to terminology with `brains explained` videos (1-5, especially on Basal Ganglia)
  - [x] read BAMI part on TM
- [ ] implement schema rl
  - [x] refresh inference logic
  - [x] make forward prediction pass on simple synthetic sequences
  - [ ] try out making backtracking on simple synthetic sequences
- [ ] implement TD($\lambda$)-based approach from Sungur's work
  - [ ] get understanding of the work
- [ ] checkout wandb [and, optionally, dvc]
- consider doing live-logging here

## Ideas

- consider using SP between an input an TM
  - make separate SPs for states, actions and rewards
  - concat them together
  - it will take care of sparsity
  - maybe smoothes volume differences for a range of diff environments
    - bc even large envs may have a very small signal
