from hima.common.sdr_encoders import IntBucketEncoder

wandb_entity = 'irodkin'
wandb_project = 'temporal_pooling'

n_actions = 4
action_bucket = 10
n_states = 25
state_bucket = 3
action_encoder = IntBucketEncoder(n_actions, action_bucket)
state_encoder = IntBucketEncoder(n_states, state_bucket)

input_columns = action_encoder.output_sdr_size
cells_per_column = 16
output_columns = 4000
output_union_sparsity = 0.01
noise_tolerance_apical = 0.1
learning_margin_apical = 0.2
seed = 42
config_tm = dict(
    columns=input_columns,
    cells_per_column=cells_per_column,
    context_cells=state_encoder.output_sdr_size,
    feedback_cells=output_columns,
    activation_threshold_basal=state_bucket,
    learning_threshold_basal=state_bucket,
    activation_threshold_apical=int(
        output_union_sparsity * output_columns * (1 - noise_tolerance_apical)),
    learning_threshold_apical=int(
        output_union_sparsity * output_columns * (1 - learning_margin_apical)),
    connected_threshold_basal=0.5,
    permanence_increment_basal=0.1,
    permanence_decrement_basal=0.01,
    initial_permanence_basal=0.4,
    predicted_segment_decrement_basal=0.001,
    sample_size_basal=state_bucket,
    max_synapses_per_segment_basal=state_bucket,
    max_segments_per_cell_basal=32,
    connected_threshold_apical=0.5,
    permanence_increment_apical=0.1,
    permanence_decrement_apical=0.01,
    initial_permanence_apical=0.4,
    predicted_segment_decrement_apical=0.001,
    sample_size_apical=int(output_union_sparsity * output_columns),
    max_synapses_per_segment_apical=int(output_union_sparsity * output_columns),
    max_segments_per_cell_apical=32,
    prune_zero_synapses=True,
    timeseries=False,
    anomaly_window=1000,
    confidence_window=1000,
    noise_tolerance=0.0,
    sm_ac=0.99,
    seed=42
)

config_sp_lower = dict(
    boostStrength=0.0,
    columnDimensions=[output_columns],
    inputDimensions=[input_columns * cells_per_column],
    potentialRadius=input_columns * cells_per_column,
    dutyCyclePeriod=1000,
    globalInhibition=True,
    localAreaDensity=0.01,
    minPctOverlapDutyCycle=0.001,
    numActiveColumnsPerInhArea=0,
    potentialPct=0.5,
    spVerbosity=0,
    stimulusThreshold=3,
    synPermConnected=0.5,
    synPermActiveInc=0.1,
    synPermInactiveDec=0.01,
    wrapAround=True,
    seed=seed
)

config_tp = dict(
    activeOverlapWeight=1,
    predictedActiveOverlapWeight=2,
    maxUnionActivity=output_union_sparsity,
    exciteFunctionType='Logistic',
    decayFunctionType='Exponential',
    decayTimeConst=10.0,
    synPermPredActiveInc=0.1,
    synPermPreviousPredActiveInc=0.05,
    historyLength=20,
    minHistory=3,
    **config_sp_lower
)
config_sp_upper = dict(
    boostStrength=0.0,
    columnDimensions=[output_columns],
    inputDimensions=config_sp_lower['columnDimensions'],
    potentialRadius=output_columns,
    dutyCyclePeriod=1000,
    globalInhibition=True,
    localAreaDensity=0.01,
    minPctOverlapDutyCycle=0.001,
    numActiveColumnsPerInhArea=0,
    potentialPct=0.5,
    spVerbosity=0,
    stimulusThreshold=3,
    synPermConnected=0.5,
    synPermActiveInc=0.1,
    synPermInactiveDec=0.01,
    wrapAround=True,
    seed=seed
)

utp_conf = dict(
    inputDimensions=[input_columns * cells_per_column],
    columnDimensions=[output_columns],
    initial_pooling=0.5,
    pooling_decay=0.1,
    permanence_inc=0.1,
    permanence_dec=0.01,
    sparsity=0.004,
    active_weight=0.5,
    predicted_weight=2.0,
    receptive_field_sparsity=0.5,
    activation_threshold=0.6,
    history_length=20,
    union_sdr_sparsity=0.01,
    prev_perm_inc=0.05
)

stp_config = dict(
    initial_pooling=1,
    pooling_decay=0.05,
    lower_sp_conf=config_sp_lower,
    upper_sp_conf=config_sp_upper
)
