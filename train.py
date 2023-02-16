import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import tqdm

from config import GPT2Config
from model import GPT2LMHead
from data.data import PaulGrahamEssaysDataset

# for logging metrics to wandb
import wandb
wandb.login()

################
# Prepare data #
################
train_data = PaulGrahamEssaysDataset(ctx_size=1024, split='train')
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, pin_memory=True, num_workers=2)

val_data = PaulGrahamEssaysDataset(ctx_size=1024, split='val')
val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True, pin_memory=True, num_workers=2)

#################
# Prepare model #
#################
model = GPT2LMHead.from_pretrained()
config = GPT2Config()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device) # TODO: use DistributedDataParallel

optimizer = AdamW(model.parameters(), lr=config.min_lr)

log_interval = 100
accumulate_iters = 5
num_epochs = 2

scaler = torch.cuda.amp.GradScaler()

training_iter = 0
for epoch in range(num_epochs):
    for i, (X, Y) in tqdm(enumerate(train_dataloader)):
        X, Y = X.to(device), Y.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = model(X, Y)
                loss = loss / accumulate_iters
        
        scaler.scale(loss).backward()
        
        # TODO: set LR, use a schedule
        if (i+1) % accumulate_iters == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # TODO: log train/val loss and learning rate
        if training_iter % log_interval == 0:
            # wandb.log({ })
            pass

        training_iter += 1