from tqdm import tqdm
import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device:", device, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Device:", device)

    
    types = ["train", "dev"]
    all_label = {"train" : [], "dev" : []}

    print("Loading data...")
    all_data = torch.load("/tmp2/b08208032/ADD2023/pt/all_data.pt")

    for type in types:
        label_file = f"/tmp2/b08208032/ADD2023/origin/ADD2023_Track2_{type}/label.txt"
        with open(label_file, 'r') as f:
            for idx, line in tqdm(enumerate(f.readlines())):
                size = all_data[type][idx].size(2)
                tmp = torch.zeros(size)
                line = line.split()
                if line[-1] == '0':
                    parts = line[1].split('/')
                    audio_len = float(parts[-1].split('-')[1])
                    for part in parts:
                        if part[-1] == 'F':
                            time = part.split('-')
                            tmp[int(round(float(time[0]) / audio_len * (size - 1), 0)) : int(round(float(time[1]) / audio_len * (size - 1), 0))] = 1

                all_label[type].append(tmp)

        print("Saving...")
        torch.save(all_label, "all_label.pt")

    torch.save(all_label["train"], "train_label.pt")
    torch.save(all_label["dev"], "dev_label.pt")
    print("Done!")