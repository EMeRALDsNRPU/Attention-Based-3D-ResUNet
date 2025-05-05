from loadData import load_data
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class My3DDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Example data (replace with actual 3D data and target sequences)
# data shape: (num_samples, channels, depth, height, width)
# targets shape: (num_samples, target_sequence_length)
def data_loader():

    tv1, data, targets = load_data()
    
    # Create the dataset and dataloader
    tv = int(0.75*tv1)
    trainData = My3DDataset(data[0:tv], targets[0:tv])
    valData = My3DDataset(data[tv:], targets[tv:])
    print(len(trainData))
    print(len(valData))
    trainloader = DataLoader(trainData, batch_size=1, pin_memory=False, shuffle=True)
    valloader = DataLoader(valData, batch_size=1, pin_memory=False, shuffle=True)
    return trainloader, valloader
data_loader()