from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pneumonai.training.model import build_model
from pneumonai.training.dataset import ChestXrayDataset
from pneumonai.training.trainer import validate
from pneumonai.preprocessing.specification import load_preprocessing_spec

REPO = Path(".")
CHECKPOINT = REPO / "checkpoints" / "resnet18_lr0.001_best.pt"
SPEC = load_preprocessing_spec(REPO / "configs" / "preprocessing.yaml")
device = torch.device("cpu")

test_dataset = ChestXrayDataset(REPO / "data" / "splits" / "test.csv", SPEC)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

model = build_model("resnet18", num_classes=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

criterion = torch.nn.BCEWithLogitsLoss()
test_loss, test_acc = validate(model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
