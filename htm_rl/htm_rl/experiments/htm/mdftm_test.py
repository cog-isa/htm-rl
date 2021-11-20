from htm_rl.modules.htm.temporal_memory import MovingDelayedFeedbackTM
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.common.sdr_encoders import IntBucketEncoder
from htm_rl.envs.biogwlab.module import EntityType
from htm_rl.agents.hima.utils import get_arrow

from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import yaml


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def main():
    config_env = yaml.load("""
      seed: 43058
      shape_xy: [5, 5]
    
      actions_cost:
        base_cost: -0.01
        weights:
          stay: 2.0
          turn: 1.0
          move: 1.0
      actions:
        - move right
        - move down
        - move left
        - move up
    
      rendering:
        view_rectangle: [[-1, -1], [1, 1]]
    
      areas:
        n_types: 4
    
      obstacle:
        density: 0.1
        map_name: '../../../experiments/hima/maps/empty.map'
    
      food:
        n_items: 0
        reward: 1.
        # positions: [ [ 5, 5 ] ]
    
      agent:
        positions: [[2, 2]]
        change_position: false
        direction: up
      #  rendering:
      #    what:
      #      - position
      ##      - view direction
      #    bucket_size: 3
    
      terminate:
        # episode_max_steps: -1
        early_stop: false
        n_items_to_collect: 1
    """, Loader=yaml.Loader)

    env = BioGwLabEnvironment(**config_env)

    seed = 333
    n_actions = 4
    action_bucket = 10
    action_encoder = IntBucketEncoder(n_actions, action_bucket)

    output_columns = 4000
    output_sdr_size = 40
    cells_per_column = 16
    noise_tolerance_apical = 0.1
    learning_margin_apical = 0.2
    noise_tolerance_basal = 0.1
    learning_margin_basal = 0.2

    config_sp = dict(
        inputDimensions=[env.output_sdr_size],
        columnDimensions=[output_columns],
        potentialRadius=env.output_sdr_size,
        potentialPct=0.5,
        globalInhibition=True,
        localAreaDensity=0,
        numActiveColumnsPerInhArea=output_sdr_size,
        stimulusThreshold=3,
        synPermInactiveDec=0.01,
        synPermActiveInc=0.1,
        synPermConnected=0.5,
        minPctOverlapDutyCycle=0.001,
        dutyCyclePeriod=1000,
        boostStrength=0.0,
        seed=seed,
        spVerbosity=0,
        wrapAround=True)
    history = 5
    config_tm = dict(
        max_steps_history=history,
        columns=action_encoder.output_sdr_size,
        cells_per_column=cells_per_column,
        context_cells=output_columns,
        feedback_cells=output_columns,
        activation_threshold_basal=int(output_sdr_size * (1 - noise_tolerance_basal)),
        learning_threshold_basal=int(output_sdr_size * (1 - learning_margin_basal)),
        activation_threshold_apical=int(output_sdr_size * (1 - noise_tolerance_apical)),
        learning_threshold_apical=int(output_sdr_size * (1 - learning_margin_apical)),
        connected_threshold_basal=0.5,
        permanence_increment_basal=0.1,
        permanence_decrement_basal=0.01,
        initial_permanence_basal=0.4,
        predicted_segment_decrement_basal=0.001,
        sample_size_basal=output_sdr_size,
        max_synapses_per_segment_basal=output_sdr_size,
        max_segments_per_cell_basal=32,
        connected_threshold_apical=0.5,
        permanence_increment_apical=0.1,
        permanence_decrement_apical=0.01,
        initial_permanence_apical=0.4,
        predicted_segment_decrement_apical=0.001,
        sample_size_apical=output_sdr_size,
        max_synapses_per_segment_apical=output_sdr_size,
        max_segments_per_cell_apical=32,
        prune_zero_synapses=True,
        timeseries=True,
        anomaly_window=1000,
        confidence_window=1000,
        noise_tolerance=0.0,
        sm_ac=0.99
    )

    sp = SpatialPooler(**config_sp)
    tm = MovingDelayedFeedbackTM(**config_tm)
    pretrain_sp(sp, env, 1000)
    steps = 5000
    run(tm, sp, action_encoder, env, steps)
    path = f'./h_{history}_n_{steps}'
    Path(path).mkdir(exist_ok=True)
    draw_policy(tm, sp, env, action_encoder, (5, 5), path, sample=1)


def pretrain_sp(sp: SpatialPooler, env: BioGwLabEnvironment, iterations):
    input_sp = SDR(sp.getInputDimensions())
    output_sp = SDR(sp.getNumColumns())
    all_states = np.flatnonzero(~env.env.aggregated_mask[EntityType.Obstacle])
    positions = [env.env.agent._unflatten_position(i_flat) for i_flat in all_states]
    indices = np.arange(len(positions))
    print('pretrain...')
    for _ in tqdm(range(iterations)):
        np.random.shuffle(indices)
        for i in indices:
            env.env.agent.position = positions[i]
            _, obs, _ = env.observe()
            input_sp.sparse = obs
            sp.compute(input_sp, True, output_sp)


def run(tm: MovingDelayedFeedbackTM, sp: SpatialPooler, action_encoder: IntBucketEncoder, env: BioGwLabEnvironment, num_steps: int):
    input_sp = SDR(sp.getInputDimensions())
    output_sp = SDR(sp.getNumColumns())

    context_tm = SDR(sp.getNumColumns())
    feedback_tm = SDR(sp.getNumColumns())
    input_tm = SDR(action_encoder.output_sdr_size)
    print('running...')
    for _ in tqdm(range(num_steps)):
        reward, obs, is_first = env.observe()

        input_sp.sparse = obs
        sp.compute(input_sp, True, output_sp)

        context_tm.sparse = output_sp.sparse
        feedback_tm.sparse = output_sp.sparse
        action = np.random.randint(action_encoder.n_values)
        input_tm.sparse = action_encoder.encode(action)

        tm.set_active_feedback_cells(feedback_tm.sparse)
        tm.activate_apical_dendrites(True)
        tm.propagate_feedback()

        tm.set_active_context_cells(context_tm.sparse)

        tm.activate_basal_dendrites(True)
        tm.predict_cells()

        tm.set_active_columns(input_tm.sparse)
        tm.activate_cells(True)

        env.act(action)


def draw_policy(tm: MovingDelayedFeedbackTM, sp: SpatialPooler, env: BioGwLabEnvironment,
                action_encoder: IntBucketEncoder,
                env_shape,
                path,
                positions=None,
                sample=1.0,
                actions_map=None):
    print('drawing...')
    if actions_map is None:
        actions_map = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

    sp_input = SDR(sp.getInputDimensions())
    sp_output = SDR(sp.getNumColumns())
    action_codes = [action_encoder.encode(x) for x in range(action_encoder.n_values)]

    all_states = np.flatnonzero(~env.env.aggregated_mask[EntityType.Obstacle])
    if positions is None:
        if sample != 1:
            positions = np.random.choice(all_states,
                                         int(sample*all_states.size),
                                         replace=False)
        else:
            positions = all_states
        positions = [env.env.agent._unflatten_position(i_flat) for i_flat in positions]
    for pos in tqdm(positions):
        env.env.agent.position = pos
        _, obs, _ = env.observe()
        sp_input.sparse = obs
        sp.compute(sp_input, False, sp_output)
        goal = sp_output.sparse.copy()
        p = np.zeros(env_shape)
        a = np.zeros(env_shape)
        for s in all_states:
            coords = env.env.agent._unflatten_position(s)
            env.env.agent.position = coords
            _, obs, _ = env.observe()
            sp_input.sparse = obs
            sp.compute(sp_input, False, sp_output)
            state = sp_output.sparse.copy()

            tm.set_active_context_cells(state)
            tm.set_active_feedback_cells(goal)
            tm.activate_apical_dendrites(False)
            tm.activate_basal_dendrites(False)
            tm.predict_cells()

            prediction = tm.get_predicted_columns()
            overlaps = np.zeros(len(action_codes))
            for j, a_code in enumerate(action_codes):
                overlaps[j] = np.intersect1d(prediction, a_code).size
            action = np.argmax(overlaps)
            probs = softmax(overlaps)
            a_prob = probs[action]
            if coords != pos:
                p[coords[0], coords[1]] = a_prob
                a[coords[0], coords[1]] = action
            else:
                p[coords[0], coords[1]] = 0.25

        # draw probabilities
        plt.figure(figsize=(13, 11))
        ax = sns.heatmap(p, cbar=True, vmin=0.25, vmax=1, linewidths=3)
        # draw action labels
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if (i, j) != pos:
                    action = a[i, j]
                    row, col, drow, dcol = get_arrow(actions_map[action])
                    ax.arrow(j + col, i + row, dcol, drow, length_includes_head=True,
                             head_width=0.2,
                             head_length=0.2)
        figure = ax.get_figure()
        figure.savefig(path + '/' + f'goal_{pos}.png')
        plt.close(figure)


if __name__ == '__main__':
    main()
