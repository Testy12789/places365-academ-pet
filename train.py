from sklearn.metrics import recall_score, f1_score, precision_score
from torchvision.datasets import Places365
from torchvision import transforms, models
from torch.utils.data import DataLoader
from plot import plot_train
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import os
import json

# ========================= CONFIG =======================
config = {
    "BATCH": 24,
    "LR": 1e-4,
    "WEIGHT_DECAY": 1e-2,
    "AUG": "cutmix",
    "CUTMIX_ALPHA": 1.0,
    "PATH_DS": "path/to/places365",
    "LOGS_F": "logs/metrics_first_stage.json",
    "LOGS_S": "logs/metrics_second_stage.json",
    "TARGET": "precision",
    "BEST_TARGET": 0.0,
    "EPOCHS": 100,
    "PATIENCE": 8, 
    "NO_IMPROVE": 0,
    "MIN_DELTA": 0.01,
    "SAVE_WEIGHTS": True,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ============================ CUTMIX UTILS ======================
def cutmix_collate(batch):
    alpha = config.get("CUTMIX_ALPHA", 1.0)
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])

    lam = np.random.beta(alpha, alpha)
    batch_size = imgs.size(0)
    index = torch.randperm(batch_size)

    y_a, y_b = labels, labels[index]
    # прямоугольник
    W, H = imgs.size(2), imgs.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    return imgs, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """cutmix loss on batch"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==================== DATA =========================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness= 0.2,contrast= 0.2, saturation= 0.2, hue= 0.1),  
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
])

train_ds = Places365(root= config["PATH_DS"], split= "train-standard", transform= train_tf, download= True)
val_ds = Places365(root= config["PATH_DS"], split= "val", transform= val_tf, download= True)

def make_train_loader():
    return DataLoader(
        train_ds,
        batch_size= config["BATCH"],
        shuffle= True,
        persistent_workers= True,
        num_workers= 8 ,
        pin_memory= True,
        collate_fn= cutmix_collate
        )

def make_val_loader():
    return DataLoader(
        val_ds,
        batch_size= config["BATCH"],
        persistent_workers= True,
        shuffle= False,
        num_workers= 8,
        pin_memory= True
        )

# ============================ MODEL =======================
model = models.efficientnet_b2(weights= models.EfficientNet_B2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 365)

if os.path.exists("models/best.pth"):
    model.load_state_dict(torch.load("models/best.pth"))

model.to(config["DEVICE"], memory_format= torch.channels_last)

criterion = nn.CrossEntropyLoss(label_smoothing= 0.1)
scaler = torch.amp.GradScaler()

# ======================= TRAIN / VALIDATION ===============================
def train_epoch(model, optim, loader, criterion, device, cur_epoch, epochs, scaler):
    """ 1 проход с cutmix"""
    model.train()
    running_loss = 0.0
    for imgs, y_a, y_b, lam in tqdm(loader, desc= f"Epoch[{cur_epoch+1}/{epochs}]", ascii= "-#", ncols= 120):
        imgs, y_a, y_b = imgs.to(device), y_a.to(device), y_b.to(device)

        optim.zero_grad()
        with torch.amp.autocast(device_type= "cuda"):
            outputs = model(imgs)
            loss = cutmix_criterion(criterion, outputs, y_a, y_b, lam)

        # 🔹 scale → backward → step
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
   

        running_loss += loss.item()
        
    train_loss = running_loss / len(loader)
    return train_loss

def validate(model, loader, criterion, device):
        """ Валидация """
        model.eval()
        corect, total, running_loss = 0.0, 0.0, 0.0
        all_labels, all_probs, all_preds = [], [], []
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc= "validation", ascii= "-#", ncols= 120):
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                probs = torch.softmax(outputs, dim= 1)
                preds = torch.argmax(outputs, dim= 1)
                corect += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            val_loss = running_loss / len(loader)
            acc = corect / total
            recal = recall_score(all_labels, all_preds, average= "weighted")
            f1 = f1_score(all_labels, all_preds, average= "weighted")
            prec = precision_score(all_labels, all_preds, average= "weighted", zero_division= 0)
            

            return val_loss, acc, recal, f1, prec

# ====================== EARLY STOP ================================ 
def stop(model, target, config):
    """ Early stop с сохранением весов """
    STOP = False
    best_target = config["BEST_TARGET"]
    no_improve = config["NO_IMPROVE"]

    if target > best_target + config["MIN_DELTA"]:
        best_target = target
        no_improve = 0
        if config["SAVE_WEIGHTS"]:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best.pth")
    else:
        no_improve += 1
        if no_improve >= config["PATIENCE"]:
            if config["SAVE_WEIGHTS"]:
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/last.pth")
            STOP = True

    # Обновляем config
    config["BEST_TARGET"] = best_target
    config["NO_IMPROVE"] = no_improve

    return STOP

def train_stage(model, config, log_path,stage, optimizer, scheduler, train_loader, val_loader):
    """ Проход 1 стадии"""
    for epoch in range(config["EPOCHS"]//2):
            train_loss = train_epoch(model, optimizer, train_loader, criterion, config["DEVICE"], epoch, config["EPOCHS"], scaler)
            scheduler.step()
            val_loss, acc, recall, f1, prec = validate(model, val_loader, criterion, config["DEVICE"])
            
    
            metrics = {
                "train_loss": float(f"{train_loss:.4f}"),
                "val_loss": float(f"{val_loss:.4f}"),
                "acc": float(f"{acc:.4f}"),
                "recall": float(f"{recall:.4f}"),
                "f1": float(f"{f1:.4f}"),
                "precision": float(f"{prec:.4f}")
            }

            # Сохранение метрик для графиков
            if config["SAVE_WEIGHTS"]:
                os.makedirs("logs", exist_ok= True)
                if os.path.exists(log_path):
                    with open(log_path, "r") as f:
                        history = json.load(f)

                else:
                    history = {"train_loss": [], "val_loss": [], "acc": [], "recall": [], "f1": [], "precision": []}

                for k in history.keys():
                    history[k].append(metrics[k])
            
                with open(log_path, "w") as f:
                    json.dump(history, f, indent= 1)
            
            
            STOP = stop(model, metrics[config["TARGET"]], config)

            if STOP:
                 break
            
            if metrics[config["TARGET"]] > 0.25 and (epoch+1 >= 1):
                try:
                    choice = input("➡ Достигнут разумный acc. Перейти к Stage 2? [y/N]: ").strip().lower()
                except KeyboardInterrupt:
                    choice = "n"
                if choice == "y":
                    print("\n➡ Переход к Stage 2...\n")
                    """ Пересоздаем loader'ы """
                    try:
                        if hasattr(train_loader, "_iterator") and train_loader._iterator is not None:
                            train_loader._iterator._shutdown_workers()
                        if hasattr(val_loader, "_iterator") and val_loader._iterator is not None:
                            val_loader._iterator._shutdown_workers()
                    except Exception:
                        pass

                    new_train_loader = make_train_loader()
                    new_val_loader = make_val_loader()
                    return config["BEST_TARGET"], new_train_loader, new_val_loader

            os.makedirs("models", exist_ok= True)
            torch.save(model.state_dict(), "models/last.pth")
            print(f'\n[Stage {stage}]: train loss: {metrics["train_loss"]} | val loss: {metrics["val_loss"]} | precision: {metrics["precision"]:.4f} |acc: {metrics["acc"]} | recall: {metrics["recall"]} | f1: {metrics["f1"]} | best {config["TARGET"]}: {config["BEST_TARGET"]}')
            
    return config["BEST_TARGET"], train_loader, val_loader


if __name__ == "__main__":
    """ Starting 🚩"""
    try:
        train_loader = make_train_loader()
        val_loader = make_val_loader()
            
        for param in model.features.parameters():
            param.requires_grad = False
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr= config["LR"], weight_decay= config["WEIGHT_DECAY"], momentum= 0.9, nesterov= True
            )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= config["EPOCHS"]//2)

        print("===Stage 1: only classifier===\n")
        best, new_train, new_val = train_stage(
                    model,
                    config,
                    config["LOGS_F"],
                    1,
                    optimizer,
                    scheduler,
                    train_loader,
                    val_loader
                    )
        plot_train(config["LOGS_F"])


        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = optim.SGD(
                model.parameters(),
                lr= config["LR"]*0.1, weight_decay= config["WEIGHT_DECAY"], momentum= 0.9, nesterov= True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= config["EPOCHS"]//2)

        print("\n===Stage 2: all model===\n")
        _, _, _ = train_stage(
                model,
                config,
                config["LOGS_S"],
                2,
                optimizer,
                scheduler,
                new_train,
                new_val
                )
        plot_train(config["LOGS_S"])
    except KeyboardInterrupt:
        print("\n🛑 Прервано пользователем. Закрываю DataLoader воркеры...")
        # закрываем воркеров руками
        if getattr(train_loader, "_iterator", None) is not None:
            train_loader._iterator._shutdown_workers()
        if getattr(val_loader, "_iterator") is not None:
            val_loader._iterator._shutdown_workers()
        exit(0)