# FAQ

## Terminology

SAR

- in short: tuple (state, action, reward)
- given trajectory: $(s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2, ...)$
- $sar_t = (s_t, a_t, r_t)$
  - $sar_t$ is centered around the current state $s_t$
  - has reward $r_t$ given for getting __to__ this state
  - has action $a_t$ taken __from__ this state.
  - NB: so actually it's better be called RSA, but it's still called SAR
  - edge cases are
    - reward $r_0$ for initial state is 0
    - action $a_T$ for terminal state is whatever you want
- TM accepts SAR encoded to SDR as [proximal] input
  - __Important:__ I call SAR encoded to SDR as SAR SDR, but
  - as conversion is straightforward both ways
  - I often just call it SAR when it's obvious or doesn't matter
- TM remembers SAR sequences: $(sar_0, sar_1, ... )$

SAR Superposition

- union of any number of SAR
  - for SDR format it's a bitwise OR of corresponding SAR SDRs
- why it's even needed
  - TM works with SAR superpositions both as input and output
  - for any particular SAR it predicts all possible next SARs
    - they represented as union
    - i.e. they can't be separated into single SARs
    - hence SAR superposition
  - TM accepts SAR superpositions as input as well
    - I think it's obvious
- even though TM remembers SAR sequences
  - depolarization in general: active SDR $\rightarrow$ depolarized SDR
  - for SARs: active SAR superposition $\rightarrow$ depolarized SAR superposition

## Encoding

SAR SDR encoder

- encodes SAR to SDR
- uses separate state, action and reward SDR encoders
  - resulting SDRs are concatenated
- we consider only discrete environments
  - hence both states and actions are discrete sets
  - rewards are discrete too: $r \in \{0, 1\}$
- so states, actions and rewards can be represented as integer numbers
  - i.e. $s_t \in [0, |S|)$ and $a_t \in [0, |A|)$
  - for this Integer SDR encoder is used

Integer SDR encoder

- encodes integer numbers from [0, `n_values`) to SDR
- parameters:
  - `n_values` - size of the range, i.e. it's a number of unique values
  - `value_bits` - how many bits are used to encode every unique value
- resulting SDR has `n_values` $\times$ `value_bits` bits
  - which can be divided into `n_values` buckets of `value_bits` contiguous bits encoding every value from the range
  - e.g. `0000 0000 1111` is the result of encoding 2 by 3-by-4 integer encoder
    - 3 buckets are separated by space to make it clear
    - every bucket encoded by 4 bits
    - note that buckets don't intersect
- PROS
  - easy to encode/decode
  - easy to debug
  - easy to pretty print
- CONS
  - cuts out information about states similarity
    - as different values have no intersection
  - you have no control on sparsity, which is `1 / n_values`
    - you may have problems with too low or too high sparsity levels
    - it tested that TM works well with low sparsity
      - which doesn't mean it has not negative effects at all
    - just remember that sparsity may lead to problems

State SDR encoder

- for the most simple MDP environments we use integer encoding
  - because it's very good for debugging
- for complex environments encoding states [or observations] becomes tricky
  - of course, you still can enumerate all possible states and use Integer SDR encoder
  - but the number of all possible states grows very fast
    - so in practice it works well only for small environments
  - also Integer SDR encoder cut out information about states similarity
- one possible solution is to encode every pixel [or grid cell] separately then concatenate results
  - it preserves information about similarity between states

## Planning

Initial: agent is in state $s_0$  
Goal: find a sequence of actions leading to reward from initial state $s_0$ if it's possible with $n_max$ steps

Planning consists of 3 main phases:

- Forward prediction phase - predict every possible outcomes until reward (= rewarding state) is found
- Backtracking phase - backward unrolling predictions from reward
- Re-check phase - check that backtracked sequence of transitions is correct

### Planning. Forward prediction

Setting up

- Given initial state $s_0$
- To find a reward, we want to check all possible paths, starting from $s_0$
  - Then SAR superposition  
    (state = $s_0$, {all actions}, reward = 0)
  - .. represents all possible beginnings of these paths
  - So, encode it to SDR and set it as initial proximal input to TM

Forward prediction

- Given some SAR superposition as proximal input at time $t$
  - induces cells activation
- TM prediction: some SAR superposition at time $t+1$
  - prediction is in form of depolarized cells
- Save all active segments (discussed later in section)
- Check if depolarized SAR superposition contains reward = 1
- Continue forward prediction by taking predicted SAR superposition as input at $t+1$
  - until reward is found
  - or max forward steps are done

A bit of details on TM work, step by step:

- Given current proximal input [at time $t$]
- It directly induces columns activation
- Active columns [and depolarization from $t-1$] induce cells activation
  - any depolarized cell in active columns becomes active
  - or all cells become active if active column has no depolarized cells (bursting)
- Cells activation induces segments activation
  - number of active presynaptic cells $\geq$ activation threshold
- Segments activation induces cells depolarization
  - any active segment depolarizes its postsynaptic cell
- Depolarized cells can be collapsed into depolarized columns
  - any depolarized cell depolarizes its column
- Depolarized columns define predicted SAR SDR superposition [at time $t+1$]

What is saved during forward prediction phase:

- at each time step $t$
- there's a set of segments that are active
- all active segments are saved into `active_segments_timeline[t]`
  - as dictionary: `depolarized_cell` $\rightarrow$ `cell_active_segments_list`
  - `cell_active_segments_list` - list of cell's active segments
  - every active segment represented as a set of active presynaptice cells

### Planning. Backtracking

Backtracking step

- Given a set of depolarized cells `{dep_cells}` [at time $t$]
- 

- Backtrack from depolarized cells in reward = 1 columns
- Backtrack for any active segment, that induces enough depolarization

### Planning. Re-check
