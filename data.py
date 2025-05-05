import os
import SimpleITK as sitk
import numpy as np
import torch
import tempfile
import cv2
from imgreader import image_reader
def get_data():
    print(sitk.ImageFileReader())
    images = []
    labels = []
    shapes = []
    
    a=1
    input_dir = "D:\luna16\luna_complete"
    label_dir = "D:\luna16\seg-lungs-LUNA16"
    for root, dirs, files in os.walk(input_dir, topdown=False):
        for file in files:
            pathim = os.path.join(root, file)
            if file.endswith(".mhd"):            
                print(a)
                a = a+1
                print(file)
                image = sitk.ReadImage(pathim)
                label_path = os.path.join(label_dir,file)
                label = sitk.ReadImage(label_path)
                image_array = sitk.GetArrayFromImage(image)
                label_array = sitk.GetArrayFromImage(label)
                print(image_array.shape)
                print(label_array.shape)
                if image_array.shape[0]<100:
                    print(image_array.shape[0])
                    image_array = np.pad(image_array,((int((100-image_array.shape[0])/2),int((100-image_array.shape[0])/2+1)),(0,0),(0,0)),'constant')
                if label_array.shape[0]<100:
                    label_array = np.pad(label_array,((int((100-label_array.shape[0])/2),int((100-label_array.shape[0])/2+1)),(0,0),(0,0)),'constant')
                print(int((100-image_array.shape[0])/2))
                print(int((100-label_array.shape[0])/2))
                image = []
                label = []
                for i in range (100):
                    image.append(cv2.resize(image_array[i],(300,300)))
                    label.append(cv2.resize(label_array[i],(300,300)))
                # label_array = np.where(label_array>0.0,1.0,0.0)
                # print("unique",np.unique(label_array))
                # break
                images.append(image)
                labels.append(label)
                shape = np.shape(image)
                print(shape)
                shapes.append(shape)

    print(np.min(shapes,axis = 0))
    print(np.max(shapes,axis = 0))
    

    mins = 95
    cropped_images = []
    cropped_images = torch.tensor(cropped_images, dtype=torch.float32).cuda()
    cropped_labels = []
    cropped_labels = torch.tensor(cropped_labels, dtype=torch.float32).cuda()
    b = 0
    for i in images:
        i = np.array(i)
        image_center_slice = np.floor(shapes[b][0]/2)
        print(image_center_slice,b)
        print(int(image_center_slice-np.floor(mins/8)))
        print(int(image_center_slice+np.floor(mins/8)+1))
        cropped_image = i[int(image_center_slice-np.floor(mins/8)):int(image_center_slice+np.floor(mins/8)+1),:,:]
        print(np.shape(cropped_image))
        image_array = torch.tensor(cropped_image).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
        cropped_images= torch.cat((cropped_images,image_array),dim=0)
        b = b+1
    c = 0 
    for i in labels:
        i = np.array(i)
        label_center_slice = np.floor(shapes[c][0]/2)
        print(label_center_slice,c)
        cropped_label = i[int(label_center_slice-np.floor(mins/8)):int(label_center_slice+np.floor(mins/8)+1),:,:]
        label_array = torch.tensor(cropped_label).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
        cropped_labels = torch.cat((cropped_labels,label_array),dim=0)
        c = c+1

    cropped_labels = torch.where(cropped_labels> 0.0, 1.0, 0.0)
    print(torch.unique(cropped_labels))
    print(torch._shape_as_tensor(cropped_images))
    torch.save(cropped_images,'cropped_images.pt')
    torch.save(cropped_labels,'cropped_labels.pt')
    np.save('shapes.npy',shapes)

    cropped_images = torch.load('cropped_images.pt')
    cropped_labels = torch.load('cropped_labels.pt')
    shapes = np.load('shapes.npy')
    cropped_images = torch.tensor(cropped_images)
    cropped_labels = torch.tensor(cropped_labels)
    print(torch.unique(cropped_labels))
    print(torch._shape_as_tensor(cropped_images))
    return cropped_images, cropped_labels
get_data()