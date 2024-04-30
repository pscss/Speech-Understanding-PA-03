import os
import pandas as pd
import re
import numpy as np
from torch.utils.data import Dataset
import torch
import librosa
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import wandb


def get_sound_file_path_df(path):
    sound_files = []

    # Compile regex patterns for efficiency
    regex_fake_or_real = re.compile(r"/fake|/real", re.IGNORECASE)
    regex_fake = re.compile("/fake", re.IGNORECASE)
    regex_train = re.compile(r"/train[a-z]*", re.IGNORECASE)
    regex_file_ext = re.compile(r".*(\.wav|\.mp3)$", re.IGNORECASE)

    for root, dirs, files in os.walk(path):
        for file in files:
            if regex_file_ext.match(file):
                sound_files.append(os.path.join(root, file))

    df = pd.DataFrame({"path": sound_files})

    # Extracting label and type from folder names
    df["label"] = [
        (
            "fake"
            if regex_fake.search(p)
            else "real" if regex_fake_or_real.search(p) else "unknown"
        )
        for p in df["path"]
    ]
    df["type"] = [
        "train" if regex_train.search(p.rsplit("/", 1)[0]) else "test"
        for p in df["path"]
    ]

    return df


class CustomSoundDataset(Dataset):
    def __init__(self, dataframe, seconds=4, sampling_rate=16000):
        self.dataframe = dataframe
        self.sr = sampling_rate
        self.cut = seconds * sampling_rate  # Maximum length of the audio samples

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        audio_path = row["path"]
        label_text = row["label"]
        X, fs = librosa.load(audio_path, sr=self.sr)
        X_pad = self.pad(X)  # Use self.cut implicitly within the pad method
        x_inp = torch.Tensor(X_pad)

        # Map labels 'real' as 1 and 'fake' as 0, then one-hot encode
        label = 1 if label_text == "real" else 0
        # label_tensor = torch.tensor([1, 0]) if label == 1 else torch.tensor([0, 1])

        return x_inp, label

    def pad(self, x):
        x_len = x.shape[0]
        if x_len >= self.cut:
            return x[: self.cut]
        num_repeats = int(self.cut / x_len) + 1
        padded_x = np.tile(x, (1, num_repeats))[:, : self.cut][0]
        return padded_x


def evaluate_model(data_loader, model, device, wandb_log=False):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            predictions.extend(outputs[:, 1].cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return true_labels, predictions


def get_metrics(true_labels, predictions, wandb_log=False):
    # Calculate the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    auc = roc_auc_score(true_labels, predictions)

    # Calculate EER
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_index]  # FPR at EER
    eer_threshold = thresholds[eer_index]
    return fpr, tpr, thresholds, auc, eer, eer_threshold


def plot_roc_curve(fpr, tpr, auc_score, eer, filename=None):
    plt.figure()
    # Update label to include EER
    label_text = f"ROC curve (AUC = {auc_score:.2f}, EER = {eer:.2f})"
    plt.plot(fpr, tpr, color="blue", lw=2, label=label_text)
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    if filename:
        plt.savefig(filename)  # Save plot to file
    else:
        plt.show()  # Show plot if filename is not provided
