from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from copy import deepcopy

def configure(config):
    new_config = dict()
    new_config['environment'] = config['environment']
    new_config['hierarchy'] = config['hierarchy']

    environment = BioGwLabEnvironment(**config['environment'])

    # define input blocks
    new_config['input_blocks'] = [{'level': 0, 'columns': environment.env.output_sdr_size},
                                  {'level': 0, 'columns': config['muscles_size'] * 2}]

    # other blocks
    input_blocks = config['hierarchy']['input_blocks']
    output_block = config['hierarchy']['output_block']
    connections = config['hierarchy']['block_connections'][len(input_blocks):]

    blocks = [{'block': deepcopy(config['block_default']),
               'sm': deepcopy(config['spatial_memory_default']),
               'tm': deepcopy(config['temporal_memory_default'])} for _ in range(len(config['blocks']))]

    for i, block, con in zip(range(len(config['blocks'])), config['blocks'].values(), connections):
        basal_input_size = 0
        for inb in con['basal_in']:
            if inb in input_blocks:
                basal_input_size += new_config['input_blocks'][inb]['columns']
            else:
                basal_input_size += config['blocks'][inb - len(input_blocks)]['tm']['basal_columns']

        feedback_in_size = 0
        active_feedback_columns = 0
        for inf in con['feedback_in']:
            feedback_in_size += config['blocks'][str(inf - len(input_blocks))]['tm']['basal_columns']
            active_feedback_columns += int(config['blocks'][str(inf - len(input_blocks))]['sp']['localAreaDensity'] *
                                           config['blocks'][str(inf - len(input_blocks))]['tm']['basal_columns'])

        blocks[i]['block'].update(deepcopy(block['block']))
        blocks[i]['sm'].update(deepcopy(block['sm']))
        blocks[i]['tm'].update(deepcopy(block['tm']))
        blocks[i]['tm'].update({
                        'feedback_columns': feedback_in_size
        })

        if block['bg'] is not None:
            blocks[i]['bg'] = deepcopy(config['basal_ganglia_default'])
            blocks[i]['bg'].update(deepcopy(block['bg']))
        else:
            blocks[i]['bg'] = None

        if block['bg_sp'] is not None:
            blocks[i]['bg_sp'] = deepcopy(config['spatial_pooler_default'])
        else:
            blocks[i]['bg_sp'] = None

        if block['sp'] is not None:
            blocks[i]['sp'] = deepcopy(config['spatial_pooler_default'])
            blocks[i]['sp'].update(deepcopy(block['sp']))
            blocks[i]['sp'].update({'inputDimensions': [basal_input_size],
                                    'columnDimensions': [block['tm']['basal_columns']],
                                    'potentialRadius': basal_input_size})

        blocks[i]['tm'].update(dict(
            activation_inhib_feedback_threshold=int(active_feedback_columns * (1 - blocks[i]['tm']['noise_tolerance'])),
            learning_inhib_feedback_threshold=int(active_feedback_columns * (1 - blocks[i]['tm']['noise_tolerance'])),
            activation_exec_threshold=int(active_feedback_columns * (1 - blocks[i]['tm']['noise_tolerance'])),
            learning_exec_threshold=int(active_feedback_columns * (1 - blocks[i]['tm']['noise_tolerance'])),
            max_inhib_synapses_per_segment=active_feedback_columns + int(blocks[i]['sp']['localAreaDensity'] *
                                                                         blocks[i]['tm']['basal_columns']),
            max_exec_synapses_per_segment=active_feedback_columns,
            sample_inhib_feedback_size=active_feedback_columns,
            sample_exec_size=active_feedback_columns
        ))

    for i, block, con in zip(range(len(config['blocks'])), blocks, connections):
        apical_input_size = 0
        apical_active_size = 0
        for inap in con['apical_in']:
            apical_input_size += blocks[inap - len(input_blocks)]['tm']['basal_columns']
            apical_active_size += int(config['blocks'][inap - len(input_blocks)]['sp']['localAreaDensity'] *
                                      config['blocks'][inap - len(input_blocks)]['tm']['basal_columns'])

        block['tm'].update({'apical_columns': apical_input_size})
        n_active_bits = int(block['sp']['numActiveColumnsPerInhArea'] * block['sp']['localAreaDensity'])
        block['tm'].update(dict(
            activation_threshold=int(n_active_bits*(1 - block['tm']['noise_tolerance'])),
            learning_threshold=int(n_active_bits*(1 - block['tm']['noise_tolerance'])),
            max_synapses_per_segment=n_active_bits,
            sample_size=n_active_bits,

            activation_inhib_basal_threshold=n_active_bits,
            learning_inhib_basal_threshold=n_active_bits,

            activation_apical_threshold=int(apical_active_size*(1 - block['tm']['noise_tolerance'])),
            learning_apical_threshold=int(apical_active_size*(1 - block['tm']['noise_tolerance'])),

            max_apical_synapses_per_segment = apical_active_size,
            sample_inhib_basal_size=n_active_bits,
            sample_apical_size=apical_active_size
        ))

        if block['bg'] is not None:
            block['bg'].update({'input_size': apical_input_size + block['tm']['basal_columns']})

        if block['bg_sp'] is not None:
            block['bg_sp'].update({'inputDimensions': [apical_input_size + block['tm']['basal_columns']]})
    new_config['blocks'] = blocks
    # agent
    new_config['agent'] = config['agent']
    new_config['agent']['state_size'] = environment.env.output_sdr_size
    new_config['agent']['action'].update(
        dict(
            muscles_size=config['muscles_size'] * 2,
            n_actions=environment.n_actions
        )
    )
    n_active_bits = int(blocks[output_block - len(input_blocks)]['sp']['localAreaDensity'] *
                        blocks[output_block - len(input_blocks)]['tm']['basal_columns'])
    new_config['agent']['muscles'].update(
        dict(
            input_size=blocks[output_block - len(input_blocks)]['tm']['basal_columns'],
            muscles_size=config['muscles_size'] * 2,
            activation_threshold=int(n_active_bits * (1 - config['agent']['muscles']['noise_tolerance'])),
            learning_threshold=int(n_active_bits * (1 - config['agent']['muscles']['noise_tolerance'])),
            max_synapses_per_segment=n_active_bits,
            sample_size=n_active_bits
             )
    )
    new_config['seed'] = config['seed']
    return new_config
