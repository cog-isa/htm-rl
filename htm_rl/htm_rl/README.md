# FAQ

- [FAQ](#faq)
  - [Terminology](#terminology)
  - [Encoding](#encoding)
  - [Planning](#planning)
    - [Planning. Forward prediction](#planning-forward-prediction)
    - [Planning. Backtracking](#planning-backtracking)
    - [Planning. Re-check](#planning-re-check)

## Terminology

SAR

- In short: tuple (state, action, reward)
- Given trajectory: $(s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2, ...)$
- $sar_t = (s_t, a_t, r_t)$
  - $sar_t$ is centered around the current state $s_t$
  - has reward $r_t$ given for getting __to__ this state
  - has action $a_t$ taken __from__ this state.
  - NB: so actually it's better be called RSA, but it's still called SAR
  - Edge cases are
    - reward $r_0$ for initial state is 0
    - action $a_T$ for terminal state is whatever you want
- TM accepts SAR encoded to SDR as [proximal] input
  - __Important:__ I call SAR encoded to SDR as SAR SDR, but
  - as conversion is straightforward both ways
  - I often just call it SAR when it's obvious or doesn't matter
- TM remembers SAR sequences: $(sar_0, sar_1, ... )$

SAR Superposition

- Union of any number of SAR
  - For SDR format it's a bitwise OR of corresponding SAR SDRs
- Why it's even needed
  - TM works with SAR superpositions both as input and output
  - For any particular SAR it predicts all possible next SARs
    - they represented as union
    - i.e. they can't be separated into single SARs
    - hence SAR superposition
  - TM accepts SAR superpositions as input as well
    - I think it's obvious
- Even though TM remembers SAR sequences
  - Depolarization in general: active SDR $\rightarrow$ depolarized SDR
  - For SARs: active SAR superposition $\rightarrow$ depolarized SAR superposition

## Encoding

SAR SDR encoder

- Encodes SAR to SDR
- Uses separate state, action and reward SDR encoders
  - Resulting SDRs are concatenated
- I consider only discrete environments
  - Hence both states and actions are discrete sets
  - Rewards are discrete too: $r \in \{0, 1\}$
- So states, actions and rewards can be represented as integer numbers
  - i.e. $s_t \in [0, |S|)$ and $a_t \in [0, |A|)$
  - for this Integer SDR encoder is used

Integer SDR encoder

- Encodes integer numbers from [0, `n_values`) to SDR
- Parameters:
  - `n_values` - size of the range, i.e. it's a number of unique values
  - `value_bits` - how many bits are used to encode every unique value
- Resulting SDR has `n_values` $\times$ `value_bits` bits
  - which can be divided into `n_values` buckets of `value_bits` contiguous bits encoding every value from the range
  - e.g. `0000 0000 1111` is the result of encoding 2 by 3-by-4 integer encoder
    - 3 buckets are separated by space to make it clear
    - every bucket encoded by 4 bits
    - note that buckets don't intersect
- PROS
  - Easy to encode/decode
  - Easy to debug
  - Easy to pretty print
- CONS
  - Cuts out information about states similarity
    - as different values have no intersection
  - You have no direct control on sparsity, which is `1 / n_values`
    - you may have problems with too low or too high sparsity levels
    - it tested that TM works well with low sparsity
      - which doesn't mean it has not negative effects at all
    - just remember that sparsity may lead to problems

State SDR encoder

- For the most simple MDP environments we use integer encoding
  - because it's very good for debugging
- For complex environments encoding states [or observations] becomes tricky
  - Of course, you still can enumerate all possible states and use Integer SDR encoder
  - But the number of all possible states grows very fast
    - so in practice it works well only for small environments
  - Also Integer SDR encoder cut out information about states similarity
- One possible solution is to encode every pixel [or grid cell] separately then concatenate results
  - It preserves information about similarity between states

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
- To find a reward, I want to check all possible paths, starting from $s_0$
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

- At each time step $t$ there's a set of segments that are active
- These active segments are saved into `active_segments_timeline[t]`
  - as dictionary: `depolarized_cell` $\rightarrow$ `cell_active_segments_list`
  - `cell_active_segments_list` - list of cell's active segments
  - each active segment represented as a set of active presynaptic cells

### Planning. Backtracking

Backtracking step

- Given a set of depolarized cells `dep_cells` [at time $t$]
- Call active segment as _potential_ if it induces enough depolarization among `dep_cells`
  - _ehough_ means $\geq$ activation threshold
  - i.e. a number of depolarized cells will be enough for subsequent depolarizations
- Take presynaptic active cells of a potential segment as depolarized cells [at time $t-1$]

Backtracking algorithm

- Start from depolarized cells in reward = 1 columns [at time T]
- Recursively take backtracking steps for any potential segment until time 0 is reached

Details

- Given a set of depolarized cells `dep_cells` [at time $t$]
  - Each depolarized cell have at least one active segment
  - Each active segment is associated with a set of active presynaptic cells
    - This set defines some SAR SDR
    - i.e. each segment is conditioned on some correct SAR
- I want to find all potential sets of active presynaptic cells that induce enough depolarization of `dep_cells`
  - perfect case
    - `dep_cells` conditioned on the same SAR
    - i.e. this SAR induces full `dep_cells` depolarization
  - perfect multiple case
    - `dep_cells` conditioned on more than one SAR
    - i.e. each of these SAR induces full `dep_cells` depolarization
- real world relaxations
  - some SARs don't induce enough depolarization
    - i.e. $\lt$ threshold
  - segments don't exactly match each other
    - cells are conditioned on very similar SARs but not exactly the same

What is saved during this phase

- active [presynaptic] cells at time $t$ are saved as `backtracking_SAR_conditions[t]`
- they define condition on SAR at time $t-1$ to get desired depolarization at time $t$

### Planning. Re-check

- Given active presynaptic cells at time $t$ needed to induce desired depolarization
- Start forward prediction again from initial sar superposition
- But now
  - Check that SAR `backtracking_SAR_conditions[t]` is consistent with the current active SAR
  - Extract action $a_t$ from `backtracking_SAR_conditions[t]`
  - Action $a_t$ defines a path to make at time $t$
  - Replace action superposition in proximal input with $a_t$
- Check that given `backtracking_SAR_conditions` lead to reward
