import argparse


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # new
        self.seed = None
        self.state_dim = None
        self.action_dim = None

        self.cr_learning_rate = None
        self.cr_softmax_temp = None
        self.cr_eps_greedy = None
        self.cr_train_schedule = None
        self.cr_batch_size = None
        self.cr_replay_buffer_size = None
        self.cr_optimizer_fn = None

        self.ac_softmax_temp = None
        self.ac_eps_greedy = None
        self.ac_learning_rate = None
        self.ac_optimizer_fn = None

        self.num_options = None
        self.hidden_units = None
        self.hidden_act_f = None
        self.enable_sparse_init = None
        self.w_scale = None
        self.w_regularization = None

        # old
        self.task_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.discount = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.log_level = 0
        self.history_length = None
        self.double_q = False
        self.tag = 'vanilla'
        self.num_workers = 1
        self.gradient_clip = None
        self.entropy_weight = 0
        self.use_gae = False
        self.gae_tau = 1.0
        self.target_network_mix = 0.001
        self.min_memory_size = None
        self.max_steps = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.termination_regularizer = 0
        self.sgd_update_frequency = None
        self.random_action_prob = None
        self.__eval_env = None
        self.tasks = False

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
