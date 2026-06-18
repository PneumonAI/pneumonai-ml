from pathlib import Path
import torch
import yaml
import argparse
from torch.utils.data import DataLoader
from pneumonai.training.dataset import ChestXrayDataset
from pneumonai.training.trainer import train_one_epoch, validate
from pneumonai.training.model import build_model
from pneumonai.preprocessing.specification import load_preprocessing_spec

def main(config_path : Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["arch"], config["num_classes"])
    model.to(device)
    spec = load_preprocessing_spec(Path("configs/preprocessing.yaml"))
    train_dataset = ChestXrayDataset(Path(config["paths"]["train"]), spec)
    val_dataset = ChestXrayDataset(Path(config["paths"]["validation"]), spec)
    if config.get("smoke"):
        n = config["smoke_samples"]
        train_dataset.df = train_dataset.df.head(n).reset_index(drop=True)
        val_dataset.df = val_dataset.df.head(n).reset_index(drop=True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.BCEWithLogitsLoss()
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    for epoch in range(config["epochs"]):
        mean_loss_train = train_one_epoch(model, train_loader, optimizer, criterion, device)
        mean_loss_val, mean_accuracy = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {mean_loss_train:.4f} - Val Loss: {mean_loss_val:.4f} - Val Accuracy: {mean_accuracy:.4f}")
        if mean_loss_val < best_val_loss:
            best_val_loss = mean_loss_val
            torch.save(model.state_dict(),checkpoint_dir/"best.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    main(args.config)
