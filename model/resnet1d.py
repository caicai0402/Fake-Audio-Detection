import torch.nn as nn
import torchaudio
from Wav2Vec2 import Wav2Vec2

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=1, padding=0, stride=1, bias = False)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(input_dim, output_dim, kernel_size=1, padding=0, stride=1, bias = False)
        self.bn2 = nn.BatchNorm1d(output_dim)

        if stride != 1 or input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_dim)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)

        return out


class ResNet1D(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 512, kernel_size=5, padding=2, stride=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)

        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(512, 512))
        self.resblocks = nn.Sequential(*layers)

        self.conv2 = nn.Conv1d(512, output_dim, kernel_size=1, padding=0, stride=1)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.resblocks(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out

if __name__ == "__main__":
    audio_file = "/tmp2/b08208032/ADD2023/ADD2023_Track2_train/wav/ADD2023_T2_T_00000000.wav"
    waveform, sample_rate = torchaudio.load(audio_file)

    feature_extractor = Wav2Vec2()
    feature = feature_extractor.get_feature(waveform)
    print("feature shape:", feature.shape, "\n", feature)

    resnet1d = ResNet1D(input_dim=768, output_dim=128, num_blocks=12)
    input = resnet1d.forward(feature)
    print("input shape:", input.shape, "\n", input)