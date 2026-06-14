import torch
import torch.nn as nn


class BinaryMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


def get_model(input_size=4):
    return BinaryMLP(input_size)


if __name__ == "__main__":
    for size in [4, 30]:
        model = get_model(input_size=size)
        dummy = torch.randn(8, size)
        out = model(dummy)
        print(f"input_size={size} | output: {out.shape}")
