from config import GPT2Config
from model import GPT2LMHead
from data.pgessays.dataset import PaulGrahamEssaysDataset
from trainer import Trainer

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import tqdm

# for logging metrics to wandb
import wandb
wandb.login()

# TODO
# wandb.init(project='project', entity='entity', name='run_name')
# wandb.config = { 'figure_out': 'what_to_put_here'}

# TODO: take in hyperparam and training config values
# as command line args and populate them into config separately

config = GPT2Config()

################
# Prepare data #
################
train_data = PaulGrahamEssaysDataset(ctx_size=config.ctx_size, split='train')
val_data = PaulGrahamEssaysDataset(ctx_size=config.ctx_size, split='val')

train_dataloader = DataLoader(train_data)
val_dataloader = DataLoader(val_data)

############################
# Prepare model + optmizer #
############################
model = GPT2LMHead.from_pretrained()
optimizer = AdamW(model.parameters(), lr=config.min_lr)

###################
# Prepare Trainer #
###################
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    config=config
)

#############
# Pump Iron #
#############
trainer.fit(train_dataloader, val_dataloader)

# wandb.finish()