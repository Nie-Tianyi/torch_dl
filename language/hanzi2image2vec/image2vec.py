import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from hanzi2image import get_common_chars, render_hanzi_to_gray, display_gray_matrix


class ChineseCAE32(nn.Module):
    def __init__(self, latent_dim=128):
        super(ChineseCAE32, self).__init__()

        # Encoder: (1, 32, 32) -> latent_dim
        self.encoder = nn.Sequential(
            # Layer 1: 32x32 -> 16x16
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 2: 16x16 -> 8x8
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 3: 8x8 -> 4x4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),
        )

        # Decoder: latent_dim -> (1, 32, 32)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            # 还原 4x4 -> 8x8
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            # 还原 8x8 -> 16x16
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            # 还原 16x16 -> 32x32
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


# --- Encoder 类 ---
class ChineseEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ChineseEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


# --- Decoder 类 ---
class ChineseDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ChineseDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


if __name__ == "__main__":
    chars = get_common_chars()
    data = np.array([render_hanzi_to_gray(i, size=32) for i in chars])
    data = data.reshape(-1, 1, 32, 32)
    data = torch.from_numpy(data).float() / 255.0

    # hyperparams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 128
    batch_size = 1024
    epochs = 10000

    # create dataset and dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # create model
    model = ChineseCAE32(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    # 只用一个总进度条，总步数 = epochs * len(dataloader)
    total_steps = epochs * len(dataloader)
    pbar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True, unit="batch")

    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_data = batch[0].to(device)
            _, reconstructed = model(input_data)
            loss = criterion(reconstructed, input_data)
            loss.backward()
            optimizer.step()

            # 更新进度条
            pbar.update(1)
            # 在描述里实时显示 Epoch 和 Loss
            pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 验证效果
    model.eval()
    with torch.no_grad():
        test_char = "藏"
        test_img = render_hanzi_to_gray(test_char, size=32)
        test_tensor = (
            torch.from_numpy(test_img).float().view(1, 1, 32, 32).to(device) / 255.0
        )

        latent, reconstructed = model(test_tensor)

        print(f"汉字 '{test_char}' 的 128 维压缩向量 (词向量):")
        print(latent.cpu().numpy())

        # 对比原图和重建图
        display_gray_matrix(reconstructed.cpu().squeeze().numpy() * 255)

    torch.save(model.state_dict(), "chinese_cae_64.pth")
    print("Model saved to chinese_cae_64.pth")
