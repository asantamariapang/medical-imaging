import torch.nn as nn
from torch.utils.data import Dataset
import torchio as tio
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

class NiiDataset(Dataset): # For 3d RESNET

        def __init__(self, subjects, transform=None):

            super().__init__() # define base constructor
            self.subjects = subjects # Define image
            self.transform = transform

        def __len__(self):
            return len(self.subjects)

        def __getitem__(self, idx):
            #returns dictionary of tensors
            subject = self.subjects[idx]

            if self.transform is not None:
                subject = self.transform(subject)

            image = subject['volume'][tio.DATA].numpy()
            image = np.squeeze(np.repeat(image[None],3,axis = 0))


            label = subject['label']

            return image, label


class NiiDataset_MLP(Dataset): # For MLP

        def __init__(self, subjects, transform=None):

            super().__init__() # define base constructor
            self.subjects = subjects # Define image
            self.transform = transform

        def __len__(self):
            return len(self.subjects)

        def __getitem__(self, idx):
            #returns dictionary of tensors
            subject = self.subjects[idx]

            if self.transform is not None:
                subject = self.transform(subject)

            image = subject['volume'][tio.DATA].numpy()
            # image = (np.squeeze(np.repeat(image[None],3,axis = 0)))


            label = subject['label']

            return image, label




#
# Create the model, based on MLP paper Sejal suggested
#
class MultilayerPerceptron2(nn.Module):
  # whatever op layer is becomes num ip for next hidden layer
  def __init__(self, input_size=45*54*45, output_size=58):
    super().__init__()
    N = 200
    self.d1 = nn.Linear(input_size, N)
    self.d2 = nn.Linear(N, N)
    self.d3 = nn.Linear(N, N)
    self.d4 = nn.Linear(N, N)
    self.d5 = nn.Linear(N, output_size)
    self.dropout = nn.Dropout(0.66)
    self.flat = nn.Flatten()


  def forward(self,X):

    X = X.view(-1,45*54*45)
    X = F.relu(self.d1(X))
    X = F.relu(self.d3(X))
    X = self.dropout(X)
    X = F.relu(self.d4(X))
    X = self.d5(X)

    return X




    # Create a dataset class for taking in images, labels, etc
class RGBDataset(Dataset): # For 2.5D Dataset

    def __init__(self,image,labels,transform=None):

        super().__init__() # define base constructor
        self.image = image # Define image
        self.labels = labels # Define labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        #returns dictionary of tensors
        image = self.image[idx]
        label = self.labels[idx]

        # Image is normalized / background is sorted: Next step is to crop the image to make is iso
        ex_dim = image.shape[1]
        image = image[:,4:ex_dim-5,:]

        # Image is iso: Project along two axes:
        ax = np.sum(image,axis=2) # projecting along two axes
        cor = np.sum(image,axis=1)
        sag = np.sum(image,axis=0)

        final_im = np.zeros((45, 45, 3))
        final_im[:,:,0] = ax
        final_im[:,:,1] = sag
        final_im[:,:,2] = cor


        # Scale the rgb image:
        mm2, MM2  = np.min(final_im), np.max(final_im)
        rgb = 255*(final_im-mm2)/(MM2-mm2)
        image = rgb.astype(np.uint8)

        # image = np.moveaxis(image, 0, -1)

        trans = transforms.Compose([transforms.ToTensor()])
        image = trans(image)


        if self.transform is not None:
            image = self.transform(image)

        return image, label