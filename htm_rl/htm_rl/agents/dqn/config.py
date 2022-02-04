class Config:
    def __init__(self):
        self.state_dim = None
        self.action_dim = None
        self.batch_size = None
        self.train_schedule = None
        self.softmax_temp = None
        self.eps_greedy = None
        self.seed = None
        self.learning_rate = None
        self.replay_buffer_size = None
        self.hidden_units = None
        self.hidden_act_f = None
        self.enable_sparse_init = None
        self.w_scale = None
        self.w_regularization = None
        self.gradient_clip = None
        self.network_fn = None
        self.optimizer_fn = None
        self.discount = None

    def merge(self, config_dict=None):
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
