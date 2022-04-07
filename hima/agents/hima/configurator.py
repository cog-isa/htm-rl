from copy import deepcopy


def configure(config):
    print('Configure ... ')
    new_config = dict()
    if 'workspace_limits' in config.keys():
        new_config['workspace_limits'] = config['workspace_limits']

    new_config['environment'] = config['environment']
    new_config['hierarchy'] = config['hierarchy']
    if 'vis_options' in config.keys():
        new_config['vis_options'] = config['vis_options']
    new_config['environment_type'] = config['environment_type']

    if 'scenario' in config.keys():
        new_config['scenario'] = config['scenario']

    if config['environment_type'] == 'gridworld':
        from hima.envs.biogwlab.env import BioGwLabEnvironment
        environment = BioGwLabEnvironment(**config['environment'])
        obs_sdr_size = environment.env.output_sdr_size
    elif config['environment_type'] == 'coppelia':
        from hima.envs.coppelia.environment import ArmEnv
        from hima.agents.hima.adapters import ArmObsAdapter
        headless = config['environment']['headless']
        config['environment'].update({'headless': True})
        environment = ArmEnv(workspace_limits=config['workspace_limits'], **config['environment'])
        config['environment'].update({'headless': headless})
        obs_adapter = ArmObsAdapter(environment, config['observation_adapter'])
        obs_sdr_size = obs_adapter.output_sdr_size
        environment.shutdown()
    else:
        raise ValueError(f'Unknown environment type: "{config["environment_type"]}"')
    print(f'obs sdr size: {obs_sdr_size}')
    # other blocks
    input_blocks = config['hierarchy']['input_blocks']
    output_block = config['hierarchy']['output_block']
    visual_block = config['hierarchy']['visual_block']
    # define input blocks
    if 'elementary_actions' in config['agent_config'].keys():
        motor_size = config['agent_config']['elementary_actions']['n_actions'] * config['agent_config']['elementary_actions']['bucket_size']
        motor_sparsity = config['agent_config']['elementary_actions']['bucket_size']/motor_size
    else:
        pmc_conf = deepcopy(config['pmc_default'])
        pmc_conf.update(config['blocks'][output_block-2]['pmc'])
        motor_size = pmc_conf['n_neurons']
        motor_sparsity = pmc_conf['sparsity']
    new_config['input_blocks'] = [{'level': 0, 'columns': obs_sdr_size,
                                   'sparsity': 1},
                                  {'level': 0, 'columns': motor_size,
                                   'sparsity': motor_sparsity}]

    connections = config['hierarchy']['block_connections'][len(input_blocks):]

    config['spatial_pooler_default']['seed'] = config['seed']
    config['temporal_memory_default']['seed'] = config['seed']
    config['basal_ganglia_default']['seed'] = config['seed']
    if 'pmc_default' in config.keys():
        config['pmc_default']['seed'] = config['seed']
    config['agent_config']['empowerment']['seed'] = config['seed']

    blocks = [{'block': deepcopy(config['block_default']),
               'tm': deepcopy(config['temporal_memory_default'])} for _ in range(len(config['blocks']))]

    for i, block, con in zip(range(len(config['blocks'])), config['blocks'].values(), connections):
        if block['sp'] is not None:
            blocks[i]['sparsity'] = config['blocks'][i]['sp']['localAreaDensity']
        else:
            if con['basal_in'][0] not in input_blocks:
                blocks[i]['sparsity'] = config['blocks'][con['basal_in'][0] - len(input_blocks)]['sparsity']
            else:
                blocks[i]['sparsity'] = new_config['input_blocks'][con['basal_in'][0]]['sparsity']

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
            feedback_in_size += config['blocks'][inf - len(input_blocks)]['tm']['basal_columns']
            active_feedback_columns += int(blocks[inf - len(input_blocks)]['sparsity'] *
                                           config['blocks'][inf - len(input_blocks)]['tm']['basal_columns'])

        blocks[i]['block'].update(deepcopy(block['block']))

        if block['sm'] is not None:
            blocks[i]['sm'] = deepcopy(config['spatial_memory_default'])
            blocks[i]['sm'].update(deepcopy(block['sm']))
        else:
            blocks[i]['sm'] = None

        blocks[i]['tm'].update(deepcopy(block['tm']))
        blocks[i]['tm'].update({
                        'feedback_columns': feedback_in_size
        })

        if block['bg'] is not None:
            blocks[i]['bg'] = deepcopy(config['basal_ganglia_default'])
            blocks[i]['bg'].update(deepcopy(block['bg']))
            if 'continuous_action' in block['bg'].keys():
                blocks[i]['block']['continuous_output'] = block['bg']['continuous_action']
            if 'pmc_default' in config.keys():
                blocks[i]['pmc'] = deepcopy(config['pmc_default'])
            if 'pmc' in block['bg'].keys():
                blocks[i]['pmc'].update(deepcopy(block['pmc']))
        else:
            blocks[i]['bg'] = None

        if block['sp'] is not None:
            blocks[i]['sp'] = deepcopy(config['spatial_pooler_default'])
            blocks[i]['sp'].update(deepcopy(block['sp']))
            blocks[i]['sp'].update({'inputDimensions': [basal_input_size],
                                    'columnDimensions': [block['tm']['basal_columns']],
                                    'potentialRadius': basal_input_size})
        else:
            blocks[i]['sp'] = None

        blocks[i]['tm'].update(dict(
            activation_inhib_feedback_threshold=int(
                active_feedback_columns * (1 - blocks[i]['tm']['noise_tolerance'])),
            learning_inhib_feedback_threshold=int(
                active_feedback_columns * (1 - blocks[i]['tm']['noise_tolerance'])),
            activation_exec_threshold=int(active_feedback_columns * (1 - blocks[i]['tm']['noise_tolerance'])),
            learning_exec_threshold=int(active_feedback_columns * (1 - blocks[i]['tm']['noise_tolerance'])),
            max_inhib_synapses_per_segment=active_feedback_columns + int(blocks[i]['sparsity'] *
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
            apical_active_size += int(blocks[inap - len(input_blocks)]['sparsity'] *
                                      config['blocks'][inap - len(input_blocks)]['tm']['basal_columns'])

        block['tm'].update({'apical_columns': apical_input_size})
        n_active_bits = int(block['tm']['basal_columns'] * blocks[i]['sparsity'])
        block['tm'].update(dict(
            activation_threshold=int(n_active_bits*(1 - block['tm']['noise_tolerance'])),
            learning_threshold=int(n_active_bits*(1 - block['tm']['noise_tolerance'])),
            max_synapses_per_segment=n_active_bits,
            sample_size=n_active_bits,

            activation_inhib_basal_threshold=n_active_bits,
            learning_inhib_basal_threshold=n_active_bits,

            activation_apical_threshold=int(apical_active_size*(1 - block['tm']['noise_tolerance'])),
            learning_apical_threshold=int(apical_active_size*(1 - block['tm']['noise_tolerance'])),

            max_apical_synapses_per_segment=apical_active_size,
            sample_inhib_basal_size=n_active_bits,
            sample_apical_size=apical_active_size
        ))

        if block['bg'] is not None:
            block['bg'].update({'input_size': apical_input_size, 'output_size': block['tm']['basal_columns']})

    new_config['blocks'] = blocks
    # agent
    new_config['agent'] = config['agent_config']
    new_config['agent']['state_size'] = obs_sdr_size

    noise_tolerance = config['agent_config']['empowerment']['tm_config']['noise_tolerance']
    learning_margin = config['agent_config']['empowerment']['tm_config']['learning_margin']
    input_size = blocks[visual_block - len(input_blocks)]['tm']['basal_columns']
    input_sparsity = blocks[visual_block - len(input_blocks)]['sparsity']

    new_config['agent']['empowerment'] = deepcopy(config['agent_config']['empowerment'])
    new_config['agent']['empowerment']['tm_config'].pop('noise_tolerance')
    new_config['agent']['empowerment']['tm_config'].pop('learning_margin')
    new_config['agent']['empowerment']['encode_size'] = input_size
    new_config['agent']['empowerment']['sparsity'] = input_sparsity
    new_config['agent']['empowerment']['tm_config'].update(
        dict(
            activationThreshold=int((1-noise_tolerance)*input_size*input_sparsity),
            minThreshold=int((1-learning_margin)*input_size*input_sparsity),
            maxNewSynapseCount=int((1+noise_tolerance)*input_size*input_sparsity),
            maxSynapsesPerSegment=int((1+noise_tolerance)*input_size*input_sparsity)
        )
    )

    if 'dreaming' in config['agent_config'].keys():
        new_config['agent']['dreaming'] = deepcopy(config['agent_config']['dreaming'])

    new_config['seed'] = config['seed']
    new_config['levels'] = config['levels']
    new_config['path_to_store_logs'] = config['path_to_store_logs']

    if config['environment_type'] == 'coppelia':
        if config['environment']['action_type'] != 'tip':
            if config['environment']['joints_to_manage'] != 'all':
                new_config['agent']['n_actions_to_accumulate'] = len(config['environment']['joints_to_manage'])
            else:
                new_config['agent']['n_actions_to_accumulate'] = 6
        else:
            new_config['agent']['n_actions_to_accumulate'] = 1
        new_config['observation_adapter'] = config['observation_adapter']
        if 'action_adapter' in config.keys():
            new_config['action_adapter'] = config['action_adapter']
            new_config['action_adapter'].update(dict(
                seed=config['seed'],
                bucket_size=config['agent_config']['elementary_actions']['bucket_size']
            ))
            new_config['agent']['elementary_actions']['n_actions'] = 3
        else:
            new_config['action_adapter_continuous'] = config['action_adapter_continuous']
    elif config['environment_type'] == 'gridworld':
        new_config['agent']['n_actions_to_accumulate'] = 1
        new_config['action_adapter'] = config['action_adapter']
        new_config['action_adapter'].update(dict(
            seed=config['seed'],
            bucket_size=config['agent_config']['elementary_actions']['bucket_size'],
            n_actions=config['agent_config']['elementary_actions']['n_actions']
        ))
    else:
        raise ValueError(f'Unknown environment type: "{config["environment_type"]}"')
    print('Configured.')
    return new_config
