import torch
import torch.nn as nn
from Wav2Vec2 import Wav2Vec2
from ResNet1D import ResNet1D

class BackendClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BackendClassifier, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=1024), num_layers=2)
        self.bilstm = nn.LSTM(input_dim, num_layers=1, hidden_size=128, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(128 * 2, output_dim)

    def forward(self, x):
        # Transformer Encoder
        transformer_out = self.transformer_encoder(x.permute(0, 2, 1))

        # BiLSTM
        bilstm_out, _ = self.bilstm(transformer_out)

        # Linear layer
        out = self.linear(bilstm_out)
        out = torch.reshape(out, (out.size(0), -1, ))
        return out

if __name__ == "__main__":
    feature_extractor = Wav2Vec2()
    audio_file = "/tmp2/b08208032/ADD2023/data/ADD2023_Track2_train/wav/ADD2023_T2_T_00000000.wav"
    feature = feature_extractor.get_feature(audio_file)
    print("feature shape", feature.shape, feature)
    
    resnet1d = ResNet1D(input_dim=768, output_dim=128, num_blocks=12)
    input = resnet1d.forward(feature)
    print(input.shape, input)

    backendclassifier = BackendClassifier(128, 1)
    output = backendclassifier(input)
    print(output.shape)