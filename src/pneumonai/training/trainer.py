from torch import nn
from torch.utils.data import DataLoader
import torch

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        predctition = model(inputs).squeeze(1)
        loss = criterion(predctition, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+= loss.item()
    return total_loss/len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        predctition = model(inputs).squeeze(1)
        loss = criterion(predctition, labels)
        predicted = predctition > 0  
        correct+= (predicted == labels.bool()).sum().item()
        total+= len(labels)
        total_loss+= loss.item()
    return (total_loss/len(loader), correct/total)