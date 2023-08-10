import torch
from MyModel import MyModel

def calculate_err(predictions, targets, threshold=0.5):
    first, second = 0, 0
    for element in predictions[0]:
        if element >= first:
            first, second = element, first
        elif element > second:
            second = element
    
    binary_predictions = (predictions >= max(second, threshold)).int()
    if (targets == 0).sum().item != 0:
        fpr = ((binary_predictions == 1) & (targets == 0)).sum().item() / (targets == 0).sum().item()
    else:
        return ((binary_predictions == 0) & (targets == 1)).sum().item() / (targets == 1).sum().item()

    if (targets == 1).sum().item() != 0:
        fnr = ((binary_predictions == 0) & (targets == 1)).sum().item() / (targets == 1).sum().item()
    else:
        return ((binary_predictions == 1) & (targets == 0)).sum().item() / (targets == 0).sum().item()
    
    err = (fpr + fnr) / 2
    return err

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device:", device, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Device:", device)
    
    dev_data = torch.load("/tmp2/b08208032/ADD2023/pt/dev_data.pt")
    dev_label = torch.load("/tmp2/b08208032/ADD2023/pt/dev_label.pt")
    test_data = torch.load("/tmp2/b08208032/ADD2023/pt/test_data.pt")

    model_pt = "/tmp2/b08208032/CaiCai_FAD/checkpoints/b64/model14.pt"
    model = MyModel().to(device)
    model.load_state_dict(torch.load(model_pt))

    err = []
    
    with torch.no_grad():
        pred = model(dev_data[1].to(device)).cpu()
        print(calculate_err(pred, dev_label[1], 0.5))



