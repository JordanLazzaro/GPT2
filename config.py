class GPT2Config:
    """ GPT-2 117M config """
    # model
    vocab_size = 50257
    ctx_size = 1024
    emb_size = 768
    nlayers = 12
    nheads = 12
    bias = True # do Linear layers have bias?
    
    # regularization
    dropout = 0.1
    weight_decay = 0.1
    
    # training
    min_lr = 6e-5
    max_lr = 6e-4 # TODO: use lr finder
    betas = (0.9, 0.95) # betas for first and second moments in Adam/AdamW
    batch_size = 8
    training_iters = 70000
    eval_iters = 50

    eval_interval = 100
    accumulate_grads = True
    accumulate_iters = 5
    log_interval = 20

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)