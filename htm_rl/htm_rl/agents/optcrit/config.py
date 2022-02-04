class Config:
    def __init__(self):
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
        self.ac_train_schedule = None
        self.ac_learning_rate = None
        self.ac_optimizer_fn = None

        self.num_options = None
        self.hidden_units = None
        self.hidden_act_f = None
        self.enable_sparse_init = None
        self.w_scale = None
        self.w_regularization = None
        self.gradient_clip = None
        self.network_fn = None
        self.discount = None
        self.entropy_weight = 0
        self.termination_regularizer = 0

    def merge(self, config_dict=None):
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
