import os
import json
import matplotlib.pyplot as plt

def plot_train(metrics_path):
    """ Строим графики по стадии"""
    if os.path.exists(metrics_path):
        metrics_log = json.load(open(metrics_path))

        epochs = list(range(1, len(metrics_log["train_loss"]) + 1))
        if len(epochs) < 2:
            print("⚠ Недостаточно точек для графика")
            return
        train_loss = metrics_log["train_loss"]
        val_loss = metrics_log["val_loss"]
        acc = metrics_log["acc"]
        f1 = metrics_log["f1"]
        prec = metrics_log["precision"]


        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
        plt.plot(epochs, val_loss, label="Val Loss", linewidth=2)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("📉 Loss per Epoch")

        plt.subplot(2, 1, 2)
        plt.plot(epochs, acc, label="Accuracy", linewidth=2)
        plt.plot(epochs, f1, label="F1-score", linewidth=2)
        plt.plot(epochs, prec, label="Precision score", linewidth=2)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("📈 Accuracy & F1 & Precision per Epoch")

        plt.tight_layout()
        plt.show()
    else:
        print("Not found logs")