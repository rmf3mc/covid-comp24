import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import ast

class ChallengeDataset(Dataset):
    def __init__(self, df, transforms=None, batch_size=16):
        """
        img_paths: List of paths to the images.
        labels: List of labels corresponding to each image.
        transforms: Transformations to apply to the images.
        """
        self.img_paths = df['Path'].tolist()
        self.labels = df['Label'].values
        self.transforms = transforms

        self.ranges = df['Range'].apply(ast.literal_eval).tolist()
        self.batch_size = batch_size

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Retrieve an item by its index."""
        # Get the image path and label for the current index
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        range_list= self.ranges[idx]


        # Check and modify the image path based on condition
        if '/challenge1' in img_path:
            img_path = img_path.replace('preprocessed', 'OriginalDatauncompressed')
        else:
            img_path = img_path.replace('preprocessed', 'dataset')

        if ('neg' in img_path and label==1) or ('pos' in img_path and label==0) or ('/co' in img_path and label==0)  or ('/no' in img_path and label==1):

            print(img_path,label)
        # Generate file paths within the range
        file_paths = list(range(range_list[0], range_list[1]))

        # Randomly sample batch_size images from the generated file paths
        if len(file_paths)>self.batch_size:
            sampled_paths = np.random.choice(file_paths, size=self.batch_size, replace=False)       
        else:
            sampled_paths= file_paths
        sampled_paths= np.sort(sampled_paths)

        # Initialize empty tensors for images and labels
        images = torch.empty((self.batch_size, 3, 384, 384))  
        labels = torch.zeros((self.batch_size,1))    
  
        
        for i, path in enumerate(sampled_paths):
            img = cv2.imread(os.path.join(img_path, f"{path}.jpg") )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            
            img = self.transforms(image=img)['image']
            images[i] = img[:]
            labels[i] = label

        return {
            'image': images,
            'label': labels
        }
