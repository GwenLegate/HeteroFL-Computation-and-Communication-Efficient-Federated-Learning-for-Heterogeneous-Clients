from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch, torchvision

if sys.version_info[0] == 2:
    import _pickle as cPickle
else:
    import pickle
from torchvision.datasets.vision import VisionDataset


class Imagenet32(VisionDataset):
    """
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,cuda=False, sz=32):

        super(Imagenet32, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.base_folder = root
        self.train = train  # training set or test set
        self.cuda = cuda

        self.data = []
        self.target = []
        self.classes_size = 1000

        if self.train:
            # now load the picked numpy arrays
            for i in range(1,11):
                file_name = 'train_data_batch_'+str(i)
                file_path = os.path.join(self.base_folder, file_name)
                with open(file_path, 'rb') as f:
                    if sys.version_info[0] == 2:
                        entry = pickle.load(f)
                    else:
                        entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    if 'labels' in entry:
                        self.target.extend(entry['labels'])
                    else:
                        self.target.extend(entry['fine_labels'])
        else:
            file_name = 'val_data'
            file_path = os.path.join(self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.target.extend(entry['labels'])
                else:
                    self.target.extend(entry['fine_labels'])

        self.target = [t - 1 for t in self.target]
        self.data = np.vstack(self.data).reshape(-1, 3, sz, sz)

        if self.cuda:
            import torch
            self.data = torch.FloatTensor(self.data).half().cuda()#type(torch.cuda.HalfTensor)
        else:
            self.data = self.data.transpose((0,2,3,1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.cuda:
            img = self.transform(img)
            input = {'img': img, 'label': target}
            #return img, target
            return input

        img = Image.fromarray(img)

        if self.transform is not None:
            try:
                img = self.transform(img)
            except TypeError:
                img = torchvision.transforms.functional.pil_to_tensor(img)
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        input = {'img': img, 'label': target}
        #return img, target
        return input

    def __len__(self):
        return len(self.data)