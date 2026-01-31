from random import random
import os
import numpy as np
import torch
import wandb
from PIL import Image, ImageShow
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

def getWandbRun(architecture="CNN Simple Neuer Versuch", dataset="rgb",batch_size=32, epochs=10,project_name="clip-extended-assessment"):

    run = wandb.init(
        entity="maximilian-speer-hasso-plattner-institut",
        project=project_name,
        name=f"{architecture}_{dataset}_bs{batch_size}_ep{epochs}",
        config={
            "architecture": architecture,
            "dataset": dataset,
            "batch_size": batch_size,
            "epochs": epochs,
        },
        settings=wandb.Settings(
            disable_code=True,      # verhindert Notebook/Code-Saving
            disable_git=True,       # verhindert Git-Status/Diff
            notebook_name="run.ipynb",  # umgeht Notebook-Name-Detection
            init_timeout=60,
        ),
    )

    run.define_metric("valid_loss", summary="min")
    run.define_metric("valid_accuracy", summary="max")
    run.define_metric("val_f1", summary="max")

    print("wandb.init fertig")
    return run

def _preds_from_outputs(outputs, threshold=0.5):
    """
    Returns predicted class indices {0,1} for binary classification.
    Supports:
      - BCEWithLogits-style: outputs shape [B] or [B,1] (logits)
      - CrossEntropy-style: outputs shape [B,2] (logits)
    """
    # ensure tensor
    if not torch.is_tensor(outputs):
        outputs = torch.tensor(outputs)

    # BCE-style (single logit)
    if outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[-1] == 1):
        logits = outputs.view(-1)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).long()

    # CE-style (two logits)
    if outputs.ndim == 2 and outputs.shape[-1] == 2:
        return outputs.argmax(dim=1).long()

    raise ValueError(f"Unsupported outputs shape for binary preds: {tuple(outputs.shape)}")


def _targets_to_01(targets):
    """
    Returns target class indices {0,1}.
    Supports:
      - targets already int labels [B]
      - one-hot targets [B,2]
      - float targets [B] in {0,1}
    """
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)

    if targets.ndim == 2 and targets.shape[-1] == 2:
        return targets.argmax(dim=1).long()

    return targets.view(-1).long()


@torch.no_grad()
def f1_binary_from_preds(preds, targets, eps=1e-12):
    """
    Binary F1 for positive class = 1
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return f1, precision, recall


def train_model(
    model, optimizer, train_dataloader, valid_dataloader,
    loss_func, get_outputs, wandbrun, epochs=20, f1_threshold=0.5
):
    train_losses, valid_losses = [], []
    fixed_val_batch = next(iter(valid_dataloader))

    for epoch in range(epochs):
        # ---------------- train ----------------
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs, target = get_outputs(model, batch)
            loss = loss_func(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= (step + 1)
        train_losses.append(train_loss)
        print(f"Epoch {epoch}: train loss = {train_loss:.4f}")

        # ---------------- valid ----------------
        model.eval()
        valid_loss = 0.0

        all_preds = []
        all_tgts  = []

        for step, batch in enumerate(valid_dataloader):
            outputs, target = get_outputs(model, batch)

            valid_loss += loss_func(outputs, target).item()

            preds = _preds_from_outputs(outputs, threshold=f1_threshold)
            tgts  = _targets_to_01(target)

            all_preds.append(preds.detach().cpu())
            all_tgts.append(tgts.detach().cpu())

        valid_loss /= (step + 1)
        valid_losses.append(valid_loss)

        all_preds = torch.cat(all_preds)
        all_tgts  = torch.cat(all_tgts)

        val_f1, val_precision, val_recall = f1_binary_from_preds(all_preds, all_tgts)

        wandbrun.log(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch}: valid loss = {valid_loss:.4f} | "
            f"F1={val_f1:.4f} P={val_precision:.4f} R={val_recall:.4f}"
        )

        table = log_predictions_table_binary(
            model, fixed_val_batch, n=16, class_names=("cube", "sphere")
        )
        wandbrun.log({f"val/sample_predictions_epoch_{epoch:03d}": table}, step=epoch)

    return train_loss, valid_loss


@torch.no_grad()
def log_predictions_table_binary(model, batch, class_names=("cube","sphere"), n=5):
    model.eval()
    device = next(model.parameters()).device

    rgb = batch[0].to(device)
    lidar_xyza = batch[1].to(device)
    y = batch[-1].to(device)

    lidar_xyza = lidar_xyza.to(device)
    rgb = rgb.to(device)
    # lidar_xyza may be tensor or list; just pass through to your fusion function/model
    # y is (B,1) float -> (B,)
    y = y.to(device).float().view(-1)

    logits, target = get_outputs_fusion(model, batch)
    logits = logits.view(-1)
    target = target.view(-1).float()

    probs = torch.sigmoid(logits)
    preds = (logits >= 0).long()
    y_int = target.long()

    table = wandb.Table(columns=["rgb", "y_true", "pred", "logit", "prob", "correct"])

    rgb_cpu = rgb.detach().cpu()
    for i in range(min(n, rgb_cpu.size(0))):
        yi = int(y_int[i].item())
        pi = int(preds[i].item())

        table.add_data(
            wandb.Image(rgb_cpu[i], caption=f"e.g. gt={class_names[yi]} pred={class_names[pi]}"),
            class_names[yi],
            class_names[pi],
            float(logits[i].item()),
            float(probs[i].item()),
            bool(yi == pi),
        )
    return table


def get_outputs_fusion(model, batch):
    target = batch[-1].to(device)
    outputs = model(batch[0].to(device), batch[1].to(device))

    return outputs, target

def set_seeds(seed=51):
    """
    Set seeds for complete reproducibility across all libraries and operations.

    Args:
        seed (int): Random seed value
    """
    # Set environment variables before other imports
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # CUDA deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # OpenCV
    cv2.setRNGSeed(seed)

    # Albumentations (for data augmentation)
    try:
        A.seed_everything(seed)
    except AttributeError:
        # Older versions of albumentations
        pass

    # PyTorch deterministic algorithms (may impact performance)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # Some operations don't have deterministic implementations
        print("Warning: Some operations may not be deterministic")

    print(f"All random seeds set to {seed} for reproducibility")


def print_CILP_results(epoch, loss, logits_per_img, is_train=True):
    if is_train:
        print(f"Epoch {epoch}")
        print(f"Train Loss: {loss} ")
    else:
        print(f"Valid Loss: {loss} ")
    print("Similarity:")
    print(logits_per_img)


def print_loss(epoch, loss, is_train=True, is_debug=False):
    loss_type = "Train" if is_train else "Valid"
    out_string = f"Epoch {epoch:3d} | {loss_type} Loss: {loss:2.4f}"
    print(out_string)


def train_projector_model(model, optimizer, loss_func, epochs, train_dataloader, valid_dataloader, wandbrun=None):
    for epoch in range(epochs):
        # ---------------- train ----------------
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_func(model, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / (step + 1)
        print_loss(epoch, train_loss, is_train=True)

        # ---------------- valid ----------------
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(valid_dataloader):
                loss = loss_func(model, batch)
                valid_loss += loss.item()

        valid_loss = valid_loss / (step + 1)
        print_loss(epoch, valid_loss, is_train=False)

        # ---------------- wandb log ----------------
        if wandbrun is not None:
            wandbrun.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

    return train_loss, valid_loss