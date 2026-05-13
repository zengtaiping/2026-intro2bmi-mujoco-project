from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
PCA_COMPONENTS = 10
EPOCHS = 120
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "go2_eeg_preprocessed.npz").exists():
            return candidate
    raise FileNotFoundError("Cannot find data/go2_eeg_preprocessed.npz")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def per_class_accuracy_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    support = cm.sum(axis=1)
    per_class_accuracy = []
    for class_idx in range(n_classes):
        tp = cm[class_idx, class_idx]
        fp = cm[:, class_idx].sum() - tp
        fn = cm[class_idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        per_class_accuracy.append((tp + tn) / cm.sum())
    per_class_accuracy = np.asarray(per_class_accuracy)
    macro_accuracy = float(per_class_accuracy.mean())
    weighted_accuracy = float(np.average(per_class_accuracy, weights=support))
    return macro_accuracy, weighted_accuracy


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    macro_acc, weighted_acc = per_class_accuracy_metrics(y_true, y_pred, n_classes)
    return {
        "macro_accuracy": macro_acc,
        "weighted_accuracy": weighted_acc,
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def format_percent(value: float) -> str:
    return f"{value * 100:.4g}%"


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


class MLP(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * n_times, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class RNN(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=8, stride=8)
        self.rnn = nn.RNN(input_size=n_channels, hidden_size=24, batch_first=True)
        self.fc = nn.Linear(24, n_classes)

    def forward(self, x):
        x = self.pool(x).transpose(1, 2)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


class LSTM(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=8, stride=8)
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=24, batch_first=True)
        self.fc = nn.Linear(24, n_classes)

    def forward(self, x):
        x = self.pool(x).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class EEGNetLSTM(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 31), padding=(0, 15), bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
        )
        self.lstm = nn.LSTM(input_size=16, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        z = self.spatial(self.temporal(x.unsqueeze(1))).squeeze(2).transpose(1, 2)
        out, _ = self.lstm(z)
        return self.fc(out[:, -1])


class TransformerDecoder(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=8, stride=8)
        self.proj = nn.Linear(n_channels, 16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.pool(x).transpose(1, 2)
        z = self.proj(x)
        z = self.encoder(z)
        return self.fc(z.mean(dim=1))


def normalize_eeg_by_train(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channel_mean = X_train.mean(axis=(0, 2), keepdims=True)
    channel_std = X_train.std(axis=(0, 2), keepdims=True)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)
    return ((X_train - channel_mean) / channel_std).astype(np.float32), ((X_test - channel_mean) / channel_std).astype(np.float32)


def train_neural_model(model_cls, X_train, y_train, X_test, seed: int, device: torch.device, n_classes: int) -> np.ndarray:
    set_seed(seed)
    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    model = model_cls(n_channels, n_times, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(X_train, y_train, shuffle=True)

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float().to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()
    return y_pred


def main() -> None:
    set_seed(RANDOM_STATE)
    project_root = find_project_root(Path.cwd())
    data_path = project_root / "data" / "go2_eeg_preprocessed.npz"
    output_path = project_root / "data" / "decoding_demo3_results.csv"

    data = np.load(data_path, allow_pickle=True)
    X_train_eeg = data["X_train"].astype(np.float32)
    X_test_eeg = data["X_test"].astype(np.float32)
    X_train_flat = data["X_train_flat"].astype(np.float32)
    X_test_flat = data["X_test_flat"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)
    label_classes = data["label_classes"].astype(str)
    n_classes = len(label_classes)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
    X_test_pca = pca.transform(X_test_scaled).astype(np.float32)
    X_train_eeg_norm, X_test_eeg_norm = normalize_eeg_by_train(X_train_eeg, X_test_eeg)

    print(f"Input: {data_path}")
    print(f"PCA components: {PCA_COMPONENTS}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Neural network input shape: {X_train_eeg_norm.shape[1:]}")
    print(f"Classes: {dict(enumerate(label_classes))}")

    rows = []

    classical_models = [
        ("LDA", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        (
            "Multiclass Logistic Regression",
            LogisticRegression(
                C=0.001,
                penalty="l2",
                solver="lbfgs",
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "SVM",
            SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=False,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    for model_name, model in classical_models:
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        metrics = compute_metrics(y_test, y_pred, n_classes)
        rows.append({"model": model_name, **metrics})
        print(f"{model_name}: weighted_recall={format_percent(metrics['weighted_recall'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_models = [
        ("MLP", MLP, 3),
        ("CNN", CNN, 1),
        ("RNN", RNN, 1),
        ("LSTM", LSTM, 1),
        ("EEGNet-LSTM", EEGNetLSTM, 3),
        ("Transformer", TransformerDecoder, 999),
    ]

    for model_name, model_cls, seed in neural_models:
        y_pred = train_neural_model(model_cls, X_train_eeg_norm, y_train, X_test_eeg_norm, seed, device, n_classes)
        metrics = compute_metrics(y_test, y_pred, n_classes)
        rows.append({"model": model_name, **metrics})
        print(f"{model_name}: weighted_recall={format_percent(metrics['weighted_recall'])}")

    result = pd.DataFrame(rows).set_index("model")
    result = result.loc[
        [
            "LDA",
            "Multiclass Logistic Regression",
            "SVM",
            "MLP",
            "CNN",
            "RNN",
            "LSTM",
            "EEGNet-LSTM",
            "Transformer",
        ]
    ]

    result_percent = result.map(format_percent)
    result_percent.to_csv(output_path, encoding="utf-8-sig")
    print("\nResult table")
    print(result_percent.to_string())
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

