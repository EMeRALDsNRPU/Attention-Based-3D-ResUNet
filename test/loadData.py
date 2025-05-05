import torch
import numpy as np

def load_data():
    cropped_images = torch.load('cropped_images.pt')
    cropped_labels = torch.load('cropped_labels.pt')
    cropped_images = torch.tensor(cropped_images)
    cropped_labels = torch.tensor(cropped_labels)
    # print(torch.unique(cropped_labels))
    print(torch._shape_as_tensor(cropped_images))
    train_len = int(0.6*len(cropped_images))
    val_len = int(0.2*len(cropped_images))
    test_len = int(0.2*len(cropped_images))

    cropped_images = cropped_images[train_len+val_len:,:,:,:,:]
    cropped_labels = cropped_labels[train_len+val_len:,:,:,:,:]

    print(cropped_images.shape)
    print(cropped_labels.shape)

    print(train_len)
    print(val_len)
    print(test_len)
    return cropped_images, cropped_labels
load_data()