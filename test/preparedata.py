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

    data, targets = load_data()
    
    # Create the dataset and dataloader
    testData = My3DDataset(data, targets)

    print(len(testData))
    testloader = DataLoader(testData, batch_size=1, pin_memory=False, shuffle=False)

    return testloader
data_loader()