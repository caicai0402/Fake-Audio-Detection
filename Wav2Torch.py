import os
from tqdm import tqdm
import torch
import torchaudio
from Wav2Vec2 import Wav2Vec2
from ResNet1D import ResNet1D

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device:", device, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Device:", device)

    types = {"train" : range(53093), "dev" : range(1, 17824), "test" : range(50000)}
    prefixs = {"train" : "ADD2023_T2_T_", "dev" : "ADD2023_T2_D_", "test" : "ADD2023_T2_E_"}
    directory_paths = ["/tmp2/b08208032/ADD2023/ADD2023_Track2_{type}/wav"]
    all_data = {"train" : [], "dev" : [], "test" : []}

    wav2vec2 = Wav2Vec2().to(device)
    resnet1d = ResNet1D(input_dim=768, output_dim=128, num_blocks=12).to(device)

    for type in types.keys():
        print(f"Processing {type} data...")
        directory_path = f"/tmp2/b08208032/ADD2023/ADD2023_Track2_{type}/wav"
        for num in tqdm(types[type]):
            file_name = prefixs[type] + "%08d"%num + ".wav"
            audio_file = os.path.join(directory_path, file_name)

            with torch.no_grad():
                waveform, sample = torchaudio.load(audio_file)
                waveform = waveform.to(device)
                
                feature = wav2vec2.get_feature(waveform)
                data = resnet1d.forward(feature)
                
                data = data.cpu()
                all_data[type].append(data)
    
        print("Saving...")
        torch.save(all_data, "all_data.pt")

    print("Done!")