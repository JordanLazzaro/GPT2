import torch
from tqdm import tqdm
import time

# for logging metrics to wandb
import wandb

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        config
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    @torch.no_grad()
    def estimate_loss(self):
        ''' eval helper adopted from nanoGPT '''
        out = {}
        self.model.eval()
        for split, data in [('train', self.train_dataloader), ('val', self.val_dataloader)]:
            data_iter = iter(data)
            losses = torch.zeros(self.config.eval_iters)
            for i in range(self.config.eval_iters):
                X, Y = next(data_iter)
                X, Y = X.to(self.device), Y.to(self.device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = self.model(X, Y)
                losses[i] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        
        return out
    
    def fit(self):
        self.model.train()
        self.model.to(self.device) # TODO: use DistributedDataParallel

        train_dataloader_iter = iter(self.train_dataloader)
        
        scaler = torch.cuda.amp.GradScaler()
        
        training_iter = 0
        t0 = time.time()
        while training_iter < self.config.training_iters:
            if training_iter % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"iter {training_iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # wandb.log({
                #     "iter": training_iter,
                #     "train/loss": losses['train'],
                #     "val/loss": losses['val'],
                # })

            X, Y = next(train_dataloader_iter)
            X, Y = X.to(self.device), Y.to(self.device)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = self.model(X, Y)
                if self.config.accumulate_grads:
                    loss = loss / self.config.accumulate_iters
            
            scaler.scale(loss).backward()
            
            if self.config.accumulate_grads:
                if training_iter % self.config.accumulate_iters == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if training_iter % self.config.log_interval == 0:
                lossf = loss.item()
                print(f"iter {training_iter}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            
            training_iter += 1