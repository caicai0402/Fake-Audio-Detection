import os
from tqdm import tqdm
import torchaudio
import torch
from MyModel import MyModel

def postprocessing(model_output, threshold=0.25):
    idx1, idx2 = 0, 0
    for i in range(model_output.size(1)):
        if model_output[0][i] > model_output[0][idx1]:
            idx1 = i
    
    for i in range(model_output.size(1)):
        if model_output[0][i] > model_output[0][idx2] and i != idx1:
            idx2 = i
    
    final_output = torch.zeros(model_output.shape)
    final_output[0][idx1], final_output[0][idx2] = model_output[0][idx1] > threshold, model_output[0][idx2] > threshold
    return idx1, idx2, final_output


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device:", device, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Device:", device)

    model_pt = "/tmp2/b08208032/CaiCai_FAD/checkpoints/b64/model14.pt"
    model = MyModel().to(device)
    model.load_state_dict(torch.load(model_pt))

    test_data = torch.load("/tmp2/b08208032/ADD2023/pt/test_data.pt")
    
    # with open("scores.txt", "w") as f:
    #     directory_path = f"/tmp2/b08208032/ADD2023/origin/ADD2023_Track2_test/wav"
    #     for idx in tqdm(range(50000)):
    #         file_name = "ADD2023_T2_E_" + "%08d"%idx + ".wav"
    #         audio_file = os.path.join(directory_path, file_name)
    #         waveform, sample_rate = torchaudio.load(audio_file)
    #         duration = float(waveform.shape[1]) / sample_rate

    #         with torch.no_grad():
    #             idx1, idx2, pred = postprocessing(model(test_data[idx].to(device)).cpu())
            
    #         ret = (pred == 1).sum().item() == 0

    #         idx1, idx2 = min(idx1, idx2), max(idx1, idx2)
    #         if ret == 1 or idx1 == idx2:
    #             time = f"0.00-{round(duration, 2)}-T"
    #         elif pred[0][0] == 1:
    #             bound = idx2 / (pred.size(1)-1) * duration
    #             time = f"0.00-{round(bound, 2)}-F/{round(bound, 2)}-{round(duration, 2)}-T"
    #         elif pred[0][pred.size(1)-1] == 1:
    #             bound = idx1 / (pred.size(1)-1) * duration
    #             time = f"0.00-{round(bound, 2)}-T/{round(bound, 2)}-{round(duration, 2)}-F"
    #         else:
    #             bound1 = idx1 / (pred.size(1)-1) * duration
    #             bound2 = idx2 / (pred.size(1)-1) * duration
    #             time = f"0.00-{round(bound1, 2)}-T/{round(bound1, 2)}-{round(bound2, 2)}-F/{round(bound2, 2)}-{round(duration, 2)}-T"

    #         content = "ADD2023_T2_E_" + "%08d"%idx + ' ' + time + ' ' + str(int(ret)) + '\n'
    #         f.write(content)

    idx = 5734
    directory_path = f"/tmp2/b08208032/ADD2023/origin/ADD2023_Track2_test/wav"
    file_name = "ADD2023_T2_E_" + "%08d"%idx + ".wav"
    audio_file = os.path.join(directory_path, file_name)
    waveform, sample_rate = torchaudio.load(audio_file)
    duration = float(waveform.shape[1]) / sample_rate

    with torch.no_grad():
        pred = model(test_data[idx].to(device)).cpu()
    
    pred = list(pred[0])
    amin, amax = int(min(pred)), int(max(pred))
    for i, val in enumerate(pred):
        pred[i] = round((int(val)-amin) / (amax-amin), 2)
    
    print(pred)