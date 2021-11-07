from htm_rl.agents.cc.utils import ImageMovement
from htm_rl.agents.cc.spatial_pooler import TemporalDifferencePooler
from htm_rl.agents.cc.temporal_memory import GeneralFeedbackTM
from htm_rl.common.sdr_encoders import IntBucketEncoder
from sklearn.datasets import load_digits
from sklearn.preprocessing import binarize
from sklearn.model_selection import train_test_split
from htm.bindings.sdr import SDR
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np


def run(images: np.ndarray, tm: GeneralFeedbackTM, sp: TemporalDifferencePooler,
        n_steps: int,
        n_epochs: int,
        crop_image_pos: ImageMovement,
        crop_image_neg: ImageMovement,
        action_encoder: IntBucketEncoder,
        learn=True):
    for epoch in tqdm(range(n_epochs)):
        for image in tqdm(images):
            crop_image_pos.set_image(image)
            crop_image_neg.set_image(1 - image)
            crop_image_pos.set_position((image.shape[0] // 2, image.shape[1] // 2))
            crop_image_neg.set_position((image.shape[0] // 2, image.shape[1] // 2))

            tm.reset()
            input_sp = SDR(sp.getInputDimensions())
            predicted_input_sp = SDR(sp.getInputDimensions())
            output_sp = SDR(sp.getColumnDimensions())
            action_code = np.empty(0)

            for step in range(n_steps):
                obs_pos = crop_image_pos.observe()
                obs_neg = crop_image_neg.observe()
                obs = np.hstack([obs_pos, obs_neg])
                input_sp.sparse = np.flatnonzero(obs)

                tm.set_active_context_cells(output_sp.sparse)
                tm.set_active_feedback_cells(action_code)

                tm.activate_apical_dendrites(learn)
                tm.activate_basal_dendrites(learn)
                tm.predict_cells()

                predicted_input_sp.sparse = tm.predicted_columns.sparse

                output_sp = sp.compute(input_sp, predicted_input_sp, learn=learn)

                tm.set_active_columns(input_sp.sparse)
                tm.activate_cells(learn)

                possible_actions = crop_image_pos.get_possible_actions()
                if len(possible_actions) != 0:
                    action = np.random.choice(possible_actions)
                    action_code = action_encoder.encode(action)
                    crop_image_pos.act(action)
                    crop_image_neg.act(action)
                else:
                    action_code = np.empty(0)


def main(seed):
    X, y = load_digits(return_X_y=True)
    X = binarize(X, threshold=X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed
    )

    general_config = dict(bucket_size=3,
                          input_x=8,
                          input_y=16,
                          col_x=50,
                          col_y=50,
                          sdr_sparsity=0.05,
                          noise_tolerance=0.01
                          )

    crop_config = dict(window_pos=[-1, -1, 1, 1],
                       actions=[[0, 1], [0, -1], [1, 0], [-1, 0],
                                [1, 1], [-1, -1], [-1, 1], [1, -1]]
                       )

    action_encoder = IntBucketEncoder(len(crop_config['actions']), bucket_size=general_config['bucket_size'])

    tm_config = dict(columns=general_config['input_x'] * general_config['input_y'],
                     cells_per_column=15,
                     context_cells=general_config['col_x'] * general_config['col_y'],
                     feedback_cells=action_encoder.output_sdr_size,
                     activation_threshold_basal=int(
                         general_config['sdr_sparsity'] * general_config['col_x'] * general_config['col_y'] *
                         (1 - general_config['noise_tolerance'])),
                     learning_threshold_basal=int(
                         general_config['sdr_sparsity'] * general_config['col_x'] * general_config['col_y'] *
                         (1 - general_config['noise_tolerance'])),
                     activation_threshold_apical=action_encoder.n_active_bits,
                     learning_threshold_apical=action_encoder.n_active_bits,
                     connected_threshold_basal=0.5,
                     permanence_increment_basal=0.1,
                     permanence_decrement_basal=0.001,
                     initial_permanence_basal=0.6,
                     predicted_segment_decrement_basal=0.001,
                     sample_size_basal=int(
                         general_config['sdr_sparsity'] * general_config['col_x'] * general_config['col_y']),
                     max_synapses_per_segment_basal=int(
                         general_config['sdr_sparsity'] * general_config['col_x'] * general_config['col_y']),
                     max_segments_per_cell_basal=32,
                     connected_threshold_apical=0.5,
                     permanence_increment_apical=0.1,
                     permanence_decrement_apical=0.001,
                     initial_permanence_apical=0.6,
                     predicted_segment_decrement_apical=0.001,
                     sample_size_apical=-1,
                     max_synapses_per_segment_apical=action_encoder.n_active_bits,
                     max_segments_per_cell_apical=32,
                     prune_zero_synapses=True,
                     timeseries=False,
                     anomaly_window=1000,
                     confidence_window=1000,
                     noise_tolerance=0.0,
                     sm_ac=0.99)
    sp_config = dict(inputDimensions=[general_config['input_x'], general_config['input_y']],
                     columnDimensions=[general_config['col_x'], general_config['col_y']],
                     potentialRadius=general_config['input_y'],
                     potentialPct=0.5,
                     globalInhibition=True,
                     localAreaDensity=general_config['sdr_sparsity'],
                     numActiveColumnsPerInhArea=0,
                     stimulusThreshold=2,
                     synPermInactiveDec=0.01,
                     synPermActiveInc=0.1,
                     synPermConnected=0.5,
                     minPctOverlapDutyCycle=0.001,
                     dutyCyclePeriod=1000,
                     boostStrength=0.0,
                     seed=seed,
                     spVerbosity=0,
                     wrapAround=False)

    crop_image_pos = ImageMovement(window_pos=crop_config['window_pos'], actions=crop_config['actions'])
    crop_image_neg = ImageMovement(window_pos=crop_config['window_pos'], actions=crop_config['actions'])

    tm = GeneralFeedbackTM(**tm_config)
    sp = TemporalDifferencePooler(**sp_config)

    run(X_train.reshape((-1, 8, 8)), tm, sp, 15, 10,
        crop_image_pos, crop_image_neg, action_encoder, True)


if __name__ == '__main__':
    main(543)
