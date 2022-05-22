import torch
from tqdm import tqdm 
from sam.sam import SAM


class Trainer:
    def __init__(self):
        pass

    def compile(self, optim_cls, decay_fn, loss_fn, metric_dict, is_sam, device, **kwargs): 
        # self.train_augmentation = train_augmentation
        # self.validation_augmentation = validation_augmentation
        self.decay_fn = decay_fn 
        self.device = device
        self.loss_fn = loss_fn
        self.metric_dict = metric_dict
        self.is_sam = is_sam
        self.optim_cls = optim_cls
        self.kwargs = kwargs  

    def _get_optimizer(self, model, optim_cls):
        if self.is_sam:
            return SAM(model.parameters(), optim_cls, **self.kwargs)
            # return SAM(model.parameters(), torch.optim.Adam, lr=lr)   
        return optim_cls(model.parameters(), **self.kwargs)
        # return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    def _get_scheduler(self, decay_fn):
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda= decay_fn)

    def fit(self, model, train_loader, validation_loader, num_epoch, save_config, track=False):
        self.optimizer = self._get_optimizer(model, self.optim_cls)
        self.scheduler = self._get_scheduler(self.decay_fn)
        self.scaler = torch.cuda.amp.GradScaler()
        best_performance, best_epoch = -100, 0

        for epoch in range(1, num_epoch+1):
            print(f"Epoch {epoch}, train_loss:{self._training_step(model, train_loader, self.loss_fn)}")
            val_loss = self._validation_step(model, validation_loader, self.loss_fn, self.metric_dict)

            if epoch % save_config["freq"] == 0:
                torch.save(model, save_config["path"])
            if best_performance < -val_loss:
                best_performance, best_epoch = -val_loss, epoch
                torch.save(model, save_config["best_path"]) 
            if track: 
                self.scheduler.step(val_loss.item())
            else: 
                self.scheduler.step()
        print(f"Best loss :{-best_performance} at {epoch} epoch.")

    def _training_step(self, model, train_loader, loss_fn): 
        model.train()
        loop = tqdm(train_loader)
        total_loss = 0 

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=self.device)
            targets = targets.type(torch.float).to(device=self.device)
         
            # forward
            if self.is_sam:
                # first forward-backward pass
                loss = loss_fn(model(data), targets)  # use this loss for any training statistics
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                total_loss += loss.item() * data.size(0)
                loop.set_postfix(loss=loss.item())

                # second forward-backward pass
                loss_fn(model(data), targets).backward()  # make sure to do a full forward pass
                self.optimizer.second_step(zero_grad=True)
                
            else: 
                with torch.cuda.amp.autocast():
                    predictions = model(data)
                    loss = loss_fn(predictions, targets)   # .to(device=self.device)

                    # backward
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()            

                    total_loss += loss.item() * data.size(0)
                    loop.set_postfix(loss=loss.item())
        return total_loss/len(train_loader.dataset)

    @torch.no_grad()
    def _validation_step(self, model, validation_loader, loss_fn, metric_dict): 
        model.eval()
        test_loss = 0
        size = 0 
        metric_eval_dict = {k:0 for k in metric_dict}

        for data, targets in validation_loader:
            data = data.to(device=self.device)
            targets = targets.type(torch.float).to(device=self.device)

            if data.ndim != 4: 
                data = data.flatten(0, - 4)
                targets = targets.flatten(0, - 4)  

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            test_loss += loss.item() * targets.size(0)
            size += targets.size(0)
            
            for metric_name in metric_dict: 
                metric_eval_dict[metric_name] += metric_dict[metric_name](predictions, targets)*targets.size(0)
                
        test_loss /= size

        print({k:(metric_eval_dict[k]/size).item() for k in metric_eval_dict})
        return test_loss
