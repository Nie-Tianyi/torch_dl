from hanzi2image import display_gray_matrix


import torch
import torch.nn as nn
from hanzi2image import render_hanzi_to_gray  # 必须用同一个函数


class ChineseEncoder128(nn.Module):
    def __init__(self, model_path="chinese_cae_128.pth", device=None):
        super(ChineseEncoder128, self).__init__()
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 结构必须和 ChineseCAE32 里的 self.encoder 完全一致
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
        )

        # 修正加载逻辑：不进行任何 replace，直接从 state_dict 提取
        full_dict = torch.load(model_path, map_location=self.device)
        # 只取以 'encoder.' 开头的权重，并保持 key 一致
        new_dict = {k: v for k, v in full_dict.items() if k.startswith("encoder.")}

        # 关键：既然我们在类里定义了 self.encoder，加载时直接用 model.load_state_dict
        # 但我们要加载到 self (这个整体)，此时 key 就是对的
        self.load_state_dict(new_dict, strict=False)

        self.to(self.device)
        self.eval()

    def forward(self, char):
        # 必须使用和你训练时一模一样的预处理
        img = render_hanzi_to_gray(char, size=32)
        tensor = (
            torch.from_numpy(img).float().view(1, 1, 32, 32).to(self.device) / 255.0
        )

        with torch.no_grad():
            latent = self.encoder(tensor)
        return latent.squeeze()


class ChineseDecoder128(nn.Module):
    def __init__(self, model_path="chinese_cae_128.pth", device=None):
        super(ChineseDecoder128, self).__init__()
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 64 * 4 * 4),
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

        full_dict = torch.load(model_path, map_location=self.device)
        new_dict = {k: v for k, v in full_dict.items() if k.startswith("decoder.")}
        self.load_state_dict(new_dict, strict=False)

        self.to(self.device)
        self.eval()

    def forward(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        with torch.no_grad():
            out = self.decoder(z)
        return out.cpu().squeeze().numpy() * 255


if __name__ == "__main__":
    encoder = ChineseEncoder128("chinese_cae_128.pth")
    vector = encoder("藏")

    print(vector)

    decoder = ChineseDecoder128("chinese_cae_128.pth")
    output = decoder(vector)

    display_gray_matrix(output)
