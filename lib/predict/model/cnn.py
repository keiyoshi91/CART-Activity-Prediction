import torch
import torch.nn as nn


class LengthMaxPool1D(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())

    def forward(self, x):
        x = self.layer(x)
        x = torch.max(x, dim=1).values
        return x


class CnnModel(nn.Module):
    def __init__(
        self,
        embed_size: int,
        input_size: int = 256,
        kernel_size: int = 5,
        output_size: int = 1,
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Conv1d(embed_size, input_size, kernel_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.embed = LengthMaxPool1D(input_size, input_size * 2)
        self.decoder = nn.Linear(input_size * 2, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embed(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = CnnModel(320)
    x = torch.randn(10, 100, 320)
    x = model(x)
