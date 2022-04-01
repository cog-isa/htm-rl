from hima.common.image import ImageMovement
from hima.modules.htm.utils import get_receptive_field
from hima.modules.htm.spatial_pooler import TemporalDifferencePooler
from hima.modules.htm.temporal_memory import GeneralFeedbackTM
from hima.common.sdr_encoders import IntBucketEncoder
from sklearn.datasets import load_digits
from sklearn.preprocessing import binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from htm.bindings.sdr import SDR
import numpy as np
import wandb


def run(images: np.ndarray,
        labels: np.ndarray,
        tm: GeneralFeedbackTM,
        sp: TemporalDifferencePooler,
        n_steps: int,
        n_epochs: int,
        crop_image_pos: ImageMovement,
        crop_image_neg: ImageMovement,
        action_encoder: IntBucketEncoder,
        learn=True,
        logger=None,
        use_prediction=True,
        receptive_field_sample_size=40,
        behavior_sample_prob=0.01):
    classes, counts = np.unique(labels, return_counts=True)
    class_counts = {cls: counts[i] for i, cls in enumerate(classes)}

    indices = np.arange(images.shape[0])
    cells = np.random.choice(np.arange(sp.getNumColumns()), size=receptive_field_sample_size, replace=False)
    for epoch in range(n_epochs):
        total_f1_per_class = {cls: 0 for cls in classes}
        step_f1_per_class = {cls: np.zeros(n_steps) for cls in classes}
        np.random.shuffle(indices)
        for i, idx in enumerate(indices):
            image = images[idx]
            label = labels[idx]
            crop_image_pos.set_image(image)
            crop_image_neg.set_image(1 - image)
            crop_image_pos.set_position((image.shape[0] // 2, image.shape[1] // 2))
            crop_image_neg.set_position((image.shape[0] // 2, image.shape[1] // 2))

            tm.reset()
            input_sp = SDR(sp.getInputDimensions())
            predicted_input_sp = SDR(sp.getInputDimensions())
            prev_input_sp = SDR(sp.getInputDimensions())
            output_sp = SDR(sp.getColumnDimensions())
            action = None
            action_code = np.empty(0)
            f1s = np.zeros(n_steps)
            animation = list()
            if np.random.random() < behavior_sample_prob:
                draw_frame = True
            else:
                draw_frame = False

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

                if use_prediction:
                    predicted_input_sp.sparse = tm.predicted_columns.sparse

                output_sp = sp.compute(input_sp, predicted_input_sp, learn=learn)

                tm.set_active_columns(input_sp.sparse)
                tm.activate_cells(learn)
                # log
                if step != 0:
                    f1s[step] = f1_score(input_sp.dense.flatten(), predicted_input_sp.dense.flatten())

                if draw_frame:
                    animation.append(draw_animation_frame(crop_image_pos,
                                                          input_sp,
                                                          predicted_input_sp,
                                                          prev_input_sp,
                                                          action))
                # log
                possible_actions = crop_image_pos.get_possible_actions()
                if len(possible_actions) != 0:
                    action = np.random.choice(possible_actions)
                    action_code = action_encoder.encode(action)
                    crop_image_pos.act(action)
                    crop_image_neg.act(action)
                else:
                    action_code = np.empty(0)

                prev_input_sp.sparse = np.copy(input_sp.sparse)
        # log
            step_f1_per_class[label] += (f1s / (class_counts[label]))
            total_f1_per_class[label] += (f1s[1:].mean() / (class_counts[label]))
            if len(animation) > 0:
                if logger is not None:
                    logger.log(
                        {f'behavior_samples/animation': [wandb.Image(x) for x in animation]})
                animation.clear()

        if logger is not None:
            logger.log(
                {f'receptive_field': [wandb.Image(0.9 - 0.85 * get_receptive_field(sp, cell)) for cell
                                      in cells]},
                commit=False)
            logger.log(
                {'f1_per_step':
                    wandb.plot.line_series(
                        xs=list(range(n_steps)),
                        ys=list(step_f1_per_class.values()),
                        keys=list(step_f1_per_class.keys()),
                        title="F1 Per Step",
                        xname="Step"
                    )},
                commit=False
            )
            table = wandb.Table(data=[[label, val] for (label, val) in total_f1_per_class.items()],
                                columns=["class", "total_f1"])
            logger.log(
                {'f1_per_class':
                     wandb.plot.bar(table, "class", "total_f1",
                                    title="Total F1 Per Class")}
            )
        # log


def draw_animation_frame(crop_image_pos: ImageMovement, input_sp: SDR, predicted_input_sp: SDR, prev_input_sp: SDR,
                         action: int):
    frame_x_size = crop_image_pos.image.shape[0] + 2 * input_sp.dimensions[0] + 2
    frame_y_size = max(crop_image_pos.image.shape[1], 2 * input_sp.dimensions[1]) + 1
    frame = np.zeros((frame_x_size, frame_y_size))
    # full image
    frame[:crop_image_pos.image.shape[0], :crop_image_pos.image.shape[1]] = crop_image_pos.image
    # highlight window
    frame[crop_image_pos.top_left[0]:crop_image_pos.bottom_right[0]+1,
    crop_image_pos.top_left[1]:crop_image_pos.bottom_right[1]+1] = 0.8 * frame[crop_image_pos.top_left[0]:
                                                                             crop_image_pos.bottom_right[0]+1,
                                                                       crop_image_pos.top_left[1]:
                                                                       crop_image_pos.bottom_right[1]+1] + 0.1
    # previous crop
    frame[crop_image_pos.image.shape[0]+1: crop_image_pos.image.shape[0] + input_sp.dimensions[0] + 1,
    :input_sp.dimensions[1]] = prev_input_sp.dense
    # current crop
    frame[crop_image_pos.image.shape[0]+1: crop_image_pos.image.shape[0] + input_sp.dimensions[0] + 1,
    input_sp.dimensions[1]+1: 2 * input_sp.dimensions[1]+1] = input_sp.dense
    # predicted crop
    frame[crop_image_pos.image.shape[0] + input_sp.dimensions[0]+2:,
    input_sp.dimensions[1]+1: 2 * input_sp.dimensions[1]+1] = predicted_input_sp.dense
    # action
    action_pixel = [frame.shape[0] - 2, 1]
    if action is not None:
        action_pixel[0] += crop_image_pos.actions[action][0]
        action_pixel[1] += crop_image_pos.actions[action][1]
    frame[action_pixel[0], action_pixel[1]] = 1
    return frame


def main(seed):
    np.random.seed(seed)
    X, y = load_digits(return_X_y=True)
    X = binarize(X, threshold=X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed
    )

    general_config = dict(bucket_size=3,
                          input_x=5,
                          input_y=10,
                          col_x=10,
                          col_y=10,
                          sdr_sparsity=0.05,
                          noise_tolerance=0.01,
                          use_prediction=True
                          )

    crop_config = dict(window_pos=[-2, -2, 2, 2],
                       actions=[[0, 1], [0, -1], [1, 0], [-1, 0],
                                [1, 1], [-1, -1], [-1, 1], [1, -1]]
                       )

    action_encoder = IntBucketEncoder(len(crop_config['actions']), bucket_size=general_config['bucket_size'])

    tm_config = dict(columns=general_config['input_x'] * general_config['input_y'],
                     cells_per_column=16,
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
                     permanence_decrement_basal=0.01,
                     initial_permanence_basal=0.1,
                     predicted_segment_decrement_basal=0.005,
                     sample_size_basal=int(
                         general_config['sdr_sparsity'] * general_config['col_x'] * general_config['col_y']),
                     max_synapses_per_segment_basal=int(
                         general_config['sdr_sparsity'] * general_config['col_x'] * general_config['col_y']),
                     max_segments_per_cell_basal=8,
                     connected_threshold_apical=0.5,
                     permanence_increment_apical=0.1,
                     permanence_decrement_apical=0.01,
                     initial_permanence_apical=0.1,
                     predicted_segment_decrement_apical=0.005,
                     sample_size_apical=-1,
                     max_synapses_per_segment_apical=action_encoder.n_active_bits,
                     max_segments_per_cell_apical=8,
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

    config = dict(general=general_config,
                  tm=tm_config,
                  sp=sp_config,
                  crop=crop_config)

    crop_image_pos = ImageMovement(window_pos=crop_config['window_pos'], actions=crop_config['actions'])
    crop_image_neg = ImageMovement(window_pos=crop_config['window_pos'], actions=crop_config['actions'])

    tm = GeneralFeedbackTM(**tm_config)
    sp = TemporalDifferencePooler(**sp_config)

    logger = wandb.init(project='test_tdsp', entity='hauska', config=config)

    run(X_train.reshape((-1, 8, 8)), y_train, tm, sp, 30, 3,
        crop_image_pos, crop_image_neg, action_encoder, True, logger, use_prediction=general_config['use_prediction'])


if __name__ == '__main__':
    main(543)
