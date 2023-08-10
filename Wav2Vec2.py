import torch.nn as nn
import torchaudio
import s3prl.hub as hub

class Wav2Vec2(nn.Module):
    def __init__(self):
        super(Wav2Vec2, self).__init__()
        self.model = hub.wav2vec2_base_960(requires_grad=False)

    def get_feature(self, waveform):
        feature = self.model(waveform)["last_hidden_state"].permute(0, 2, 1).contiguous()
        return feature

if __name__ == "__main__":
    audio_file = "/tmp2/b08208032/ADD2023/ADD2023_Track2_train/wav/ADD2023_T2_T_00000000.wav"
    waveform, sample_rate = torchaudio.load(audio_file)

    feature_extractor = Wav2Vec2()
    feature = feature_extractor.get_feature(waveform)
    print("feature shape:", feature.shape, "\n", feature)
