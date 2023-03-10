import torch
from tqdm import tqdm
import time
import os

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
        
        best_val_loss = float('inf')
        iters_since_best_val_loss = 0
        training_iter = 0
        t0 = time.time()
        while training_iter < self.config.training_iters:
            if training_iter % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"iter {training_iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if self.config.logging:
                    wandb.log({
                        "step": training_iter,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                    })
                if self.config.save_checkpoints:
                    if losses['val'] < best_val_loss:
                        best_val_loss = losses['val']
                        checkpoint = {
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'training_iter': training_iter,
                            'best_val_loss': best_val_loss,
                            'config': self.config
                        }
                        print(f'saving checkpoint to: {self.config.checkpoint_dir}/best_val_checkpoint.pt')
                        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, f'best_val_checkpoint.pt'))
                    else:
                        iters_since_best_val_loss += self.config.eval_interval
                if self.config.early_stopping:
                    if iters_since_best_val_loss >= self.config.patience:
                        print(f'[STOPPING] training_iter: {training_iter}, current val/loss: {losses["val"]}, best val/loss: {best_val_loss}')
                        break
                        
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
                lossf = loss.item() * self.config.accumulate_iters if self.config.accumulate_grads else loss.item()
                print(f"training_iter {training_iter}: train/loss {lossf:.4f}, time {dt*1000:.2f}ms")
            
            training_iter += 1