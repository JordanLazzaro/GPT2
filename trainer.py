import torch
from tqdm import tqdm

# for logging metrics to wandb
import wandb

class Trainer:
    def __init__(
        self,
        model,
        opitimizer,
        config
    ):
        self.model = model
        self.optimizer = opitimizer
        self.config = config
    
    def fit(self, train_dataloader, val_dataloader):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device) # TODO: use DistributedDataParallel

        scaler = torch.cuda.amp.GradScaler()

        training_iter = 0
        for epoch in range(self.config.num_epochs):
            for i, (X, Y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                X, Y = X.to(device), Y.to(device)
                # TODO: test any differnce between float16 and bfloat16
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, loss = self.model(X, Y)
                        loss = loss / self.config.accumulate_iters
                
                scaler.scale(loss).backward()
                
                # TODO: set LR using a warmup/decay schedule (ie linear + cosine or 1-cycle)
                if self.config.accumulate_grads:
                    if (i+1) % self.config.accumulate_iters == 0:
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                # TODO: log train/val loss and learning rate
                if training_iter % self.config.eval_interval == 0:
                    val_loss = self.evaluate(val_dataloader)
                    wandb.log({
                        'epoch': epoch,
                        'iter': training_iter,
                        'val/loss': val_loss
                        # log train loss on same scale
                     })
                     # TODO: save checkpoint if good enough

                if training_iter >= self.config.training_iters:
                    break
                training_iter += 1

    @torch.no_grad()
    def evaluate(self, val_dataloader):
        self.model.eval()
        # TODO: put an eval loop
        self.model.train()
        
        return None, None