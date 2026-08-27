import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from dataset import get_dataloaders, NUM_CLASSES, ACTION_CLASSES
from model import ActionRecognitionNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_action_net.pt")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(batch_size=16)

    model = ActionRecognitionNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.permute(0, 4, 1, 2, 3).to(device)
            outputs = model(videos)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=[ACTION_CLASSES[i] for i in range(NUM_CLASSES)]))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues", aspect='auto')
    plt.colorbar()
    plt.xticks(range(NUM_CLASSES), [ACTION_CLASSES[i] for i in range(NUM_CLASSES)], rotation=45)
    plt.yticks(range(NUM_CLASSES), [ACTION_CLASSES[i] for i in range(NUM_CLASSES)])
    plt.xlabel("Predicted Action")
    plt.ylabel("True Action")
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    print("Saved confusion matrix")

    # Per-class accuracy
    plt.figure(figsize=(8, 4))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.bar(range(NUM_CLASSES), per_class_acc, color='skyblue', edgecolor='black')
    plt.xticks(range(NUM_CLASSES), [ACTION_CLASSES[i] for i in range(NUM_CLASSES)], rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_class_accuracy.png"), dpi=150)
    print("Saved per-class accuracy")


if __name__ == "__main__":
    evaluate()
