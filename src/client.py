import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from model import get_model
from noise import inject_noise, inject_symmetric_noise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BRFSSClient(fl.client.NumPyClient):
    def __init__(self, X, y, noise_rate=0.0, seed=42, noise_type="asymmetric"):
        self.X = X
        self.y = y
        self.noise_rate = noise_rate
        self.seed = seed
        input_size = X.shape[1] if len(X) > 0 else 4
        self.model = get_model(input_size=input_size).to(DEVICE)

        if noise_rate > 0:
            if noise_type == "symmetric":
                self.y = inject_symmetric_noise(y, noise_rate, seed=seed)
            else:
                self.y = inject_noise(y, noise_rate, seed=seed)

        if len(self.y) > 0:
            neg = max((self.y == 0).sum(), 1)
            pos = max((self.y == 1).sum(), 1)
            total = len(self.y)
            weight = torch.tensor(
                [total / (2 * neg), total / (2 * pos)], dtype=torch.float32
            ).to(DEVICE)
        else:
            weight = torch.tensor([1.0, 1.0]).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        if len(self.X) == 0:
            return self.get_parameters(config={}), 0, {"train_loss": 0.0}

        self.set_parameters(parameters)

        MAX_SAMPLES = 500
        X, y = self.X, self.y
        if len(X) > MAX_SAMPLES:
            round_num = config.get("round", 0)
            rng = np.random.default_rng(self.seed + round_num)
            idx = rng.choice(len(X), MAX_SAMPLES, replace=False)
            X, y = X[idx], y[idx]

        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.int64).to(DEVICE)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                            batch_size=32, shuffle=True)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        total_loss, total_samples = 0.0, 0
        for epoch in range(3):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
                total_samples += len(y_batch)

        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return self.get_parameters(config={}), len(self.X), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        if len(self.X) == 0:
            return 0.0, 0, {"accuracy": 0.0, "auc": 0.5}

        self.set_parameters(parameters)

        MAX_EVAL_SAMPLES = 2000
        X, y = self.X, self.y
        if len(X) > MAX_EVAL_SAMPLES:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(X), MAX_EVAL_SAMPLES, replace=False)
            X, y = X[idx], y[idx]

        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.int64).to(DEVICE)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=256)

        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                output = self.model(X_batch)
                total_loss += self.criterion(output, y_batch).item() * len(y_batch)
                correct += (output.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
                probs = F.softmax(output, dim=1)[:, 1].cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y_batch.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        return float(total_loss / total), total, {
            "accuracy": correct / total,
            "auc": float(auc),
        }


class FedProxClient(BRFSSClient):
    """FedProx (Li et al., 2020): adds proximal term (mu/2)||w - w_global||^2 to local loss."""

    def __init__(self, X, y, noise_rate=0.0, seed=42, noise_type="asymmetric", mu=0.01):
        super().__init__(X, y, noise_rate, seed, noise_type)
        self.mu = mu

    def fit(self, parameters, config):
        if len(self.X) == 0:
            return self.get_parameters(config={}), 0, {"train_loss": 0.0}

        self.set_parameters(parameters)
        global_weights = [p.clone().detach() for p in self.model.parameters()]

        MAX_SAMPLES = 500
        X, y = self.X, self.y
        if len(X) > MAX_SAMPLES:
            round_num = config.get("round", 0)
            rng = np.random.default_rng(self.seed + round_num)
            idx = rng.choice(len(X), MAX_SAMPLES, replace=False)
            X, y = X[idx], y[idx]

        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.int64).to(DEVICE)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                            batch_size=32, shuffle=True)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        total_loss, total_samples = 0.0, 0
        for _ in range(3):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)

                prox = sum(
                    ((p - g) ** 2).sum()
                    for p, g in zip(self.model.parameters(), global_weights)
                )
                loss = loss + (self.mu / 2) * prox

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
                total_samples += len(y_batch)

        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return self.get_parameters(config={}), len(self.X), {"train_loss": train_loss}
