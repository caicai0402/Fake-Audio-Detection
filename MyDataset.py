import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data_list = data_list
        self.label_list = label_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx][0]
        label = self.label_list[idx]
        return data, label

def collate_fn(batch):
    max_length = max(item[1].size(0) for item in batch)
    padded_data_batch = [F.pad(item[0], (0, max_length - item[0].size(1), 0, 0)) for item in batch]
    padded_label_batch = [F.pad(item[1], (0, max_length - item[1].size(0))) for item in batch]
    return torch.stack(padded_data_batch), torch.stack(padded_label_batch)

if __name__ == "__main__":
    train_data, train_label = torch.load("/tmp2/b08208032/ADD2023/pt/train_data.pt"), torch.load("/tmp2/b08208032/ADD2023/pt/train_label.pt")
    train_dataset = MyDataset(train_data, train_label, collate_fn=collate_fn)