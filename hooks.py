from abc import ABC, abstractmethod
import torch
from utils import mixup_data, cutmix_data, mixup_criterion


class Base_Iter_Hook(ABC): 
    @abstractmethod
    def run_iter(self, model, data, targets, trainer, criterion): 
        pass


class SAM_Iter_Hook(Base_Iter_Hook): 
    def run_iter(self, model, data, targets, trainer, criterion):
        # first forward-backward pass
        loss = criterion(model(data), targets)  # use this loss for any training statistics
        loss.backward()
        trainer.optimizer.first_step(zero_grad=True)
        
        # second forward-backward pass
        criterion(model(data), targets).backward()  # make sure to do a full forward pass
        trainer.optimizer.second_step(zero_grad=True)
        return loss.item()
        
    
class Mixup_Iter_Hook(Base_Iter_Hook):
    def run_iter(self, model, data, targets, trainer, criterion):
        with torch.cuda.amp.autocast():
            # forward
            mixed_data, targets_a, targets_b, lam = mixup_data(data, targets)
            predictions = model(mixed_data)
            loss = mixup_criterion(y_a, y_b, lam)(criterion, predictions)
     
            # backward
            trainer.optimizer.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        return loss.item() 
     

class Cutmix_Iter_Hook(Base_Iter_Hook): 
    def run_iter(self, model, data, targets, trainer, criterion):
        with torch.cuda.amp.autocast():
            # forward
            data, targets = cutmix_data(data, targets)
            predictions = model(data)
            loss = criterion(predictions, targets)
            
            # backward
            trainer.optimizer.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        return loss.item()           
    

class Normal_Iter_Hook(Base_Iter_Hook): 
    def run_iter(self, model, data, targets, trainer, criterion):
        with torch.cuda.amp.autocast():
            # forward
            predictions = model(data)
            loss = criterion(predictions, targets)
            # backward
            trainer.optimizer.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        return loss.item()