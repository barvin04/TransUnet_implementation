import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
import scipy.io


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

def normalize_image(image, min_val=-125, max_val=275):
    image = (image - min_val) / (max_val - min_val)
    image[image>1] = 1
    image[image<0] = 0
    return image

class Ultrasound_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        files = {'train': 'ultrasound_train_list.txt', 'test_vol': 'ultrasound_val_list.txt'}
        sample_list = open(os.path.join(list_dir, files[self.split])).readlines()
        sample_list = [ sample.strip() for sample in sample_list]
        files_to_exclude = set([ '186_HC.png', '346_HC.png', '628_2HC.png'])
        self.sample_list = list(set(sample_list)-files_to_exclude)
        self.data_dir = base_dir
        self.scat_mat_dir = os.path.join(self.data_dir, 'mat_arrs')

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.data_dir, 'training_set', slice_name)
            label_path = os.path.join(self.data_dir, 'training_set_masks', os.path.splitext(slice_name)[0]+'_mask.png')
            scat_mat_path = os.path.join(self.scat_mat_dir, os.path.splitext(slice_name)[0]+'.mat')
            image = cv2.imread(img_path)
            image = image[:,:,0] 
            image = normalize_image(image, min_val=0, max_val=255)
            label = cv2.imread(label_path)
            label = (label[:,:,0] == 255).astype(np.uint8)
        else:
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.data_dir, 'training_set', slice_name)
            label_path = os.path.join(self.data_dir, 'training_set_masks', os.path.splitext(slice_name)[0]+'_mask.png')
            scat_mat_path = os.path.join(self.scat_mat_dir, os.path.splitext(slice_name)[0]+'.mat')
            image = cv2.imread(img_path)
            image = image[:,:,0] 
            image = normalize_image(image, min_val=0, max_val=255)
            label = cv2.imread(label_path)
            label = (label[:,:,0] == 255).astype(np.uint8)
            image = image[np.newaxis, :, :]
            label = label[np.newaxis, :, :]
        
        scat_mat = scipy.io.loadmat(scat_mat_path)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        sample['scat_mat'] = torch.from_numpy(scat_mat['S'])
        return sample

class LungXray_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        files = {'train': 'covid_lungs_seg_train_list.txt', 'test_vol': 'covid_lungs_seg_val_list.txt', 'test':'covid_lungs_seg_test_list.txt'}
        self.sample_list = open(os.path.join(list_dir, files[self.split])).readlines()
        self.data_dir = base_dir
        self.scat_mat_dir = os.path.join(self.data_dir, 'mat_arrs')

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.data_dir, 'images_bcet', slice_name)
            label_path = os.path.join(self.data_dir, 'masks_resized', slice_name)
            scat_mat_path = os.path.join(self.scat_mat_dir, os.path.splitext(slice_name)[0]+'.mat')
            image = np.load(img_path)
            label = np.load(label_path)
        else:
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.data_dir, 'images_bcet', slice_name)
            label_path = os.path.join(self.data_dir, 'masks_resized', slice_name)
            scat_mat_path = os.path.join(self.scat_mat_dir, os.path.splitext(slice_name)[0]+'.mat')
            image = np.load(img_path)
            label = np.load(label_path)
            image = image[np.newaxis, :, :]
            label = label[np.newaxis, :, :]
        scat_mat = scipy.io.loadmat(scat_mat_path)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        sample['scat_mat'] = torch.from_numpy(scat_mat['S'])
        return sample
