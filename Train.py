from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from MyModel import MyModel
from MyDataset import MyDataset, collate_fn

def evaluate(model, dev_dataloader, criterion):
    print("Start evaluating!")
    
    model.eval()
    dev_loss, dev_samples = 0, 0
    with torch.no_grad():
        for batch_data, batch_label in dev_dataloader:
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_label)

            dev_loss += loss.item() * batch_data.size(0)
            dev_samples += batch_data.size(0)

    avg_dev_loss = dev_loss / dev_samples
    return avg_dev_loss

def train(model, train_dataloader, dev_dataloader, optimizer, criterion, num_epochs):
    print("Start training!")

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss, train_sample = 0, 0
        for batch_data, batch_label in tqdm(train_dataloader):
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)

            outputs = model(batch_data)
            loss = criterion(outputs, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_data)
            train_sample += batch_data.size(0)

        avg_train_loss = train_loss / train_sample
        avg_dev_loss = evaluate(model, dev_dataloader, criterion)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss}, Dev Loss: {avg_dev_loss}")
        
        writer.add_scalar('Train/Loss', avg_train_loss, epoch+1)
        writer.add_scalar('Dev/Loss', avg_dev_loss, epoch+1)
        if epoch % 5 == 4:
            torch.save(model.state_dict(), f"./checkpoints/b64_segment/model{epoch}.pt")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device:", device, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Device:", device)

    print("Loading data and label...")
    batch_size = 64
    train_data, train_label = torch.load("/tmp2/b08208032/ADD2023/pt/train_data.pt"), torch.load("/tmp2/b08208032/ADD2023/pt/train_label.pt")
    dev_data, dev_label = torch.load("/tmp2/b08208032/ADD2023/pt/dev_data.pt"), torch.load("/tmp2/b08208032/ADD2023/pt/dev_label.pt")
    
    train_dataset = MyDataset(train_data, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    dev_dataset = MyDataset(dev_data, dev_label)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    log_dir = 'logs'
    writer = SummaryWriter(log_dir=log_dir)

    model = MyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 25

    train(model, train_dataloader, dev_dataloader, optimizer, criterion, num_epochs)

    writer.close()
    print("Done!")