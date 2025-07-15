import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from PIL import Image
import timm
from perlin import rand_perlin_2d_np
from torch.utils.data import DataLoader


import random
import math
from torchvision import transforms
import torch

import random
import math
from torchvision import transforms
import torch
import numpy as np
torch.manual_seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
#np.random.seed(3407)
def cut_paste_collate_fn(batch):
    # cutPaste return 2 tuples of tuples we convert them into a list of tuples
    img_types = list(zip(*batch))
#     print(list(zip(*batch)))
    return [torch.stack(imgs) for imgs in img_types]


class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""
    def __init__(self, transform=None):
        self.transform = transform


    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img

class CutPasteNormal(CutPaste):
    """Randomly copy one patch from the image and paste it somewhere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """
    def __init__(self, area_ratio=[0.02,0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        #TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))
        # 根据insert box做异常gt
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        mask_binary = np.zeros((h,w))
        mask_binary[to_location_h: to_location_h + cut_h, to_location_w: to_location_w + cut_w] = 255

        # print(insert_box)
        augmented = img.copy()
        # print(type(augmented))
        augmented.paste(patch, insert_box)

        return super().__call__(img, augmented), mask_binary

class CutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """
    def __init__(self, width=[2,16], height=[10,25], rotation=[-45,45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]

        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)



        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg,expand=True)
        # print(patch.size)
        #paste
        to_location_h = int(random.uniform(0, h - patch.size[1]))
        to_location_w = int(random.uniform(0, w - patch.size[0]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")

        mask_binary = np.zeros((h,w))
        mask1 = np.array(mask)
        # print("masked1 shape:", mask1.shape)
        # print("mask1.shape[0]:", mask1.shape[0])
        # print("mask1.shape[1]:", mask1.shape[1])
        # print("to_location_h+mask1.shape[0]:", to_location_h+mask1.shape[0])
        # print("to_location_w+mask1.shape[1]:", to_location_w+mask1.shape[1])
        # print("to_location_h:", to_location_h)
        # print("to_location_w:", to_location_w)
        #mask_binary[to_location_h: to_location_h+mask1.shape[1], to_location_w : to_location_w+mask1.shape[0]] = mask1
        mask_binary[to_location_h: to_location_h + mask1.shape[0], to_location_w: to_location_w + mask1.shape[1]] = mask1
        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        return super().__call__(img, augmented), mask_binary

class CutPasteUnion(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)

    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)

class CutPaste3Way(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)

    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)

        return org, cutpaste_normal, cutpaste_scar



train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.Resize((224,224)))
train_transform.transforms.append(CutPasteUnion())
import csv

class all_MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(glob.glob(root_dir + "/*/train/good/*.png"))

        print(self.image_paths)
        print(len(self.image_paths))
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx, 'image_path': self.image_paths[idx]}

        return sample




class MVTecDRAEMTestDataset_visa(Dataset):

    def __init__(self, root_dir, object_name, resize_shape=None):
        self.csv_file_path = '../datasets/VisA_20220922/split_csv/1cls.csv'
        self.obj_name = object_name
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.data=[]
        with open(self.csv_file_path, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if row[0] == object_name and row[1] == 'test':
                    self.data.append(row)

    def __len__(self):
        return len(self.data)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)*255
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask/ 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        base_path='../datasets/VisA_20220922/'
        csv_row = self.data[idx]
        #mask_path = self.mask[idx]
        if csv_row[1]=='test' and csv_row[2]=='normal':
            image, mask = self.transform_image(base_path+csv_row[3], None)
            has_anomaly = np.array([0], dtype=np.float32)
        elif csv_row[1]=='test' and csv_row[2]=='anomaly':
            image, mask = self.transform_image(base_path+csv_row[3], base_path+csv_row[4])
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'idx': idx}

        return sample

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None, dataset=0):
        self.root_dir = root_dir
        if dataset == 0:
            self.images = sorted(glob.glob(root_dir+"/good/*.png"))
        elif dataset == 1:
            self.images = sorted(glob.glob(root_dir+"/*/*.JPG"))
        elif dataset == 2:
            self.images = sorted(glob.glob(root_dir+"/*/*.bmp"))
       
        #print('image_path:',self.images)
        self.resize_shape=resize_shape
        self.dataset = dataset
    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        #print('base_dir:',base_dir)
        if base_dir == 'good' or base_dir== 'ok':
            #print(1)
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            #print(2)
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            #print('mask_path1:', mask_path)
            if self.dataset==0:
                mask_file_name = file_name.split(".")[0]+"_mask.png"
            elif self.dataset==1 or 2:
                mask_file_name = file_name.split(".")[0]+".png"
            mask_path = os.path.join(mask_path, mask_file_name)
            #print('mask_path:', mask_path)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx, 'img_path': img_path}

        return sample

class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, object_name, resize_shape=None, dataset=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        basepath='../datasets/VisA_20220922/'
        self.csv_file_path = '../datasets/VisA_20220922/split_csv/1cls.csv'
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        if dataset == 0:
            self.image_paths = sorted(glob.glob(root_dir+"/*.png"))
        elif dataset==1:
            self.image_paths = []
            with open(self.csv_file_path, "r", newline="") as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    if row[0] == object_name and row[1] == 'train':
                        self.image_paths.append(basepath+row[3])
            self.image_paths = sorted(self.image_paths)        
        elif dataset==2:
            self.image_paths = sorted(glob.glob(root_dir+"/*.bmp"))


        print(self.image_paths)
        print(len(self.image_paths))
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample

class MVTecDRAEMTrainDataset_shape(Dataset):

    def __init__(self, root_dir, anomaly_source_path, shape_path=None, resize_shape=None, dataset=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        if dataset == 0:
            self.shape_paths = sorted(glob.glob(shape_path+"/*.png"))
            self.image_paths = sorted(glob.glob(root_dir+"/*.png"))
        elif dataset ==2:
            self.shape_paths = sorted(glob.glob(shape_path+"/*.png"))
            self.image_paths = sorted(glob.glob(root_dir+"/*.bmp"))
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.4
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        #print(perlin_thr.shape) # (256, 256)
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        #print(perlin_thr.shape) # (256, 256, 1)
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr)  + (1 - beta) * img_thr + beta * image * (perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 1:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            #print(msk.shape) (256, 256, 1)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, float(angle), 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def transform_image(self, image_path, anomaly_source_path, shape_image):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        shape_info = cv2.imread(shape_image, 0)
        shape_info = cv2.resize(shape_info, dsize=(224, 224))


        rot_angle = torch.randint(-90,91,(1,))
        do_aug_orig = torch.rand(1).numpy()[0] > 0.5
        if do_aug_orig:
            image = self.rotate_image(image,rot_angle)
            shape_info = self.rotate_image(shape_info, rot_angle)
        # print(image.shape)
        # print(shape_info.shape)
        shape_info = np.expand_dims(shape_info,2)
        #print(shape_info.shape)
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        shape_info = np.array(shape_info).reshape((shape_info.shape[0], shape_info.shape[1], shape_info.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        #print(shape_info.shape)
        shape_info = np.transpose(shape_info, (2, 0, 1)) # [1,256,256]
        #print(shape_info.shape)
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, shape_info, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()

        image, shape_info, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx],self.shape_paths[idx])
        sample = {'image': image, "shape_info": shape_info, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample

class MVTecDRAEMTrainDataset_visa_shape(Dataset):

    def __init__(self, root_dir, anomaly_source_path, shape_path,object_name, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file_path = '../datasets/VisA_20220922/split_csv/1cls.csv'
        self.obj_name = object_name
        self.basepath= '../datasets/VisA_20220922/'
        self.image_path=[]
        with open(self.csv_file_path, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if row[0] == object_name and row[1] == 'train':
                    self.image_path.append(self.basepath+row[3])
        # print(self.image_path)
        #self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.shape_paths = sorted(glob.glob(shape_path+"/*.png"))

        print(self.image_path)
        #print(shape_path)
        print(self.shape_paths)

        #self.image_paths = sorted(glob.glob(root_dir+"/*.JPG"))
        matching_file_paths = []
        for original_path in self.image_path:
            original_image_name = os.path.basename(original_path)
            original_image_name = os.path.splitext(original_image_name)[0]
            # print(original_image_name)
            #print(self.shape_paths)
            original_image_name = original_image_name.zfill(4)
            for target_path in self.shape_paths:
                target_image_name = os.path.basename(target_path)
                target_image_name = os.path.splitext(target_image_name)[0]
                target_image_name = target_image_name.zfill(4)
                #print(target_image_name)
                if original_image_name == target_image_name: 
                    matching_file_paths.append(target_path)
                    #print(target_path)
        #print(matching_file_paths)
        self.image_paths = sorted(self.image_path)
        self.shape_paths = sorted(matching_file_paths)
        
        #print(self.image_paths)
        #print(self.shape_paths)
        print(len(self.shape_paths))
        print(len(self.image_paths))
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.4
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        #print(perlin_thr.shape) # (256, 256)
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        #print(perlin_thr.shape) # (256, 256, 1)
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr)  + (1 - beta) * img_thr + beta * image * (perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 1:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            #print(msk.shape) (256, 256, 1)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, float(angle), 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def transform_image(self, image_path, anomaly_source_path, shape_image):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        shape_info = cv2.imread(shape_image, 0)
        shape_info = cv2.resize(shape_info, dsize=(224, 224))


        rot_angle = torch.randint(-90,91,(1,))
        do_aug_orig = torch.rand(1).numpy()[0] > 0.5
        if do_aug_orig:
            image = self.rotate_image(image,rot_angle)
            shape_info = self.rotate_image(shape_info, rot_angle)
        # print(image.shape)
        # print(shape_info.shape)
        shape_info = np.expand_dims(shape_info,2)
        #print(shape_info.shape)
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        shape_info = np.array(shape_info).reshape((shape_info.shape[0], shape_info.shape[1], shape_info.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        #print(shape_info.shape)
        shape_info = np.transpose(shape_info, (2, 0, 1)) # [1,256,256]
        #print(shape_info.shape)
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, shape_info, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()

        image, shape_info, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx],self.shape_paths[idx])
        sample = {'image': image, "shape_info": shape_info, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample

class MVTecDRAEMTrainDataset_cutpaste(Dataset):

    def __init__(self, root_dir, anomaly_source_path, shape_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.shape_paths = sorted(glob.glob(shape_path+"/*.png"))
        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        # print(self.image_paths)
        return len(self.image_paths)

    def augment_image(self, image):
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.7:
            image = image.astype(np.float32)

            mask = np.zeros((image.shape[0], image.shape[1]))

            mask = np.expand_dims(mask, 2)

            return image, mask, np.array([0.0], dtype=np.float32)
        else:
            image = Image.fromarray(image)
            image = train_transform(image)


            org = np.array(image[0][0]).astype(np.float32)
            augmented = np.array(image[0][1]).astype(np.float32)
            mask = np.expand_dims(image[1].astype(np.float32), 2)

            return augmented, mask, np.array([1.0], dtype=np.float32)


    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, float(angle), 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def transform_image(self, image_path, anomaly_source_path, shape_image):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        shape_info = cv2.imread(shape_image, 0)
        shape_info = cv2.resize(shape_info, dsize=(self.resize_shape[1], self.resize_shape[0]))


        rot_angle = torch.randint(-90,91,(1,))
        do_aug_orig = torch.rand(1).numpy()[0] > 0.5
        if do_aug_orig:
            image = self.rotate_image(image,rot_angle)
            shape_info = self.rotate_image(shape_info, rot_angle)

        shape_info = np.expand_dims(shape_info,2)

        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image)
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        shape_info = np.array(shape_info).reshape((shape_info.shape[0], shape_info.shape[1], shape_info.shape[2])).astype(np.float32) / 255.0

        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))

        shape_info = np.transpose(shape_info, (2, 0, 1)) # [1,256,256]

        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

        return image, shape_info, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()

        image, shape_info, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx],self.shape_paths[idx])
        sample = {'image': image, "shape_info": shape_info, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample


class MyMVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, ground_truth_paths, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"/*.png")) # 获得所有正常图片的图片路径

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg")) # 获得所有异常图片的图片路径

        self.ground_truth_paths = sorted(glob.glob(ground_truth_paths+"/*.png")) # 获得正常图片的shape图片路径

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True), # 调整图像对比度
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),# 调整亮度
                      iaa.pillike.EnhanceSharpness(), # 改变图像清晰度
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True), # 调整色调和饱和度
                      iaa.Solarize(0.5, threshold=(32,128)), # 图像曝光
                      iaa.Posterize(), # 色调分离
                      iaa.Invert(), # 将像素值从v 变成 255-v
                      iaa.pillike.Autocontrast(), # 调对比度
                      iaa.pillike.Equalize(), # 均衡图像直方图
                      iaa.Affine(rotate=(-45, 45)) # 将图像旋转 -45 到 45 度
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path): # image是正常图片 anomaly_source_path为异常来源路径
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0

        #======= 对随机选中的异常来源图片做随机图像增强
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        #=======

        #======= 生成perlin噪声
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        #=======

        #======= 根据perlin_noise生成异常区域
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
        #=======

        beta = torch.rand(1).numpy()[0] * 0.8 # 将torch转numpy 并取出第一个值 这样就得到一个数值了  tensor([0.6958337]) =》 [0.6958337] =》 0.6958337

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * ( # 加了噪声生成的异常图片
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5: # 有一半几率直接返回未加噪声的图片
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else: # 有另一半几率返回正常图片
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32) # 加了噪声的位置
            augmented_image = msk * augmented_image + (1-msk)*image # 最终的异常图片
            has_anomaly = 1.0
            if np.sum(msk) == 0: # 标记该图片是否为异常图片 即是否在原图上加了噪声
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path): # image_path为正常图片路径  anomaly_source_path为异常来源的路径
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        #======= 一定几率对原图做旋转
        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)
        #=======

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0

        #======= 调用上面的方法对原图加上生成的异常  返回 带有生成异常的异常图片 异常区域的gt flag标记是否为异常图片
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        #=======

        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item() # 随机选一张正常图片
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item() # 随机选一张异常来源图片
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample

class MVTecDRAEMTrainDatasetFeat(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None, backbone="wide_resnet50_2"):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.Vit_feat_extractor = timm.create_model("vit_large_patch16_224", pretrained=True)
        self.cnn_feat_extractor = timm.create_model("wide_resnet50_2", pretrained=True, features_only=True, out_indices=[1, 2, 3])
        for parameter in self.cnn_feat_extractor.parameters():
            parameter.requires_grad = False
        self.cnn_feat_extractor.eval()
    def __len__(self):
        return len(self.image_paths)

    def get_vit_features(self, input_tensor, feature_extractor):

        for parameter in feature_extractor.parameters():
            parameter.requires_grad = False
        feature_extractor.eval()
        #print(input_tensor.shape)
        input_tensor = np.resize(input_tensor, (3, 224, 224))
        feature = feature_extractor.patch_embed(torch.tensor(np.expand_dims(input_tensor, axis=0)))
        #feature = feature.squeeze(0)

        features = []
        for i in range(len(feature_extractor.blocks)):
            feature = feature_extractor.blocks[i](feature)
            features.append(feature)

        for i in range(len(features)):
            #print(features[i].shape)
            features[i] = feature_extractor.norm(features[i]).reshape(1, 224 // 16, 224 // 16,
                                                                      -1).mean(3) # [1, 14, 14]
            #print(features[i].shape)
        for i in range(len(features)):
            #print(features[i].shape)
            features[i] = np.array(Image.fromarray(features[i].numpy().squeeze(0)).resize((self.resize_shape))) # [256, 256]
            #print(features[i].shape)

        # 24个layer 每8个layer取均值 组成一张特征图
        features_vit_1 = np.concatenate([np.expand_dims(features[i], axis=0) for i in range(0,8)],axis=0).transpose(1, 2, 0).mean(2) # [256 ,256]
        features_vit_2 = np.concatenate([np.expand_dims(features[i], axis=0) for i in range(8,16)],axis=0).transpose(1, 2, 0).mean(2)  # [256 ,256]
        features_vit_3 = np.concatenate([np.expand_dims(features[i], axis=0) for i in range(16,24)],axis=0).transpose(1, 2, 0).mean(2)  # [256 ,256]

        # features_vit_avg = features_vit_avg.permute(0,2,3,1).mean(3) # [b, 256 ,256]
        return features_vit_1, features_vit_2, features_vit_3 # 3 * [256, 256]
    def augment_image(self, image): # 传进去的图片是经过正则化的
        perlin_scale = 6
        min_perlin_scale = 0

        anomaly_feat_1, anomaly_feat_2, anomaly_feat_3 = self.get_vit_features(image, self.Vit_feat_extractor)  #3*[256, 256]
        normal_feat = self.cnn_feat_extractor(torch.tensor(image).unsqueeze(0)) # [256, 56, 56]  [512, 28, 28]  [1024, 14, 14]
        #print(normal_feat[0].shape)
        normal_feat_resize=[] # 存储CNN抽出来的3张特征图
        for i in range(len(normal_feat)):
            normal_feat[i] = normal_feat[i].squeeze(0)
            feat_cnn_avg = Image.fromarray(normal_feat[i].permute(1,2,0).mean(axis=2).numpy()).resize((self.resize_shape)) # [256, 256]
            normal_feat_resize.append(feat_cnn_avg)

        # 输入网络的最终正常特征
        final_normal_feat = np.concatenate([np.expand_dims(normal_feat_resize[i], axis=2) for i in range(0, len(normal_feat_resize))], axis=2) #[256, 256, 3]
        # 输入网络的最终异常特征
        final_anomaly_feat = np.concatenate([np.expand_dims(anomaly_feat_1,axis=2), np.expand_dims(anomaly_feat_2,axis=2), np.expand_dims(anomaly_feat_3,axis=2)], axis=2)  # [256, 256, 3]

        # 生成perlin noise
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley)) #
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise)) # [256, 256]

        perlin_thr = np.expand_dims(perlin_thr, axis=2) # [256,256, 1]

        feat_thr = final_anomaly_feat.astype(np.float32) * perlin_thr / 255.0

        # 生成异常特征
        augmented_feat = final_normal_feat * (1 - perlin_thr)  + feat_thr

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            final_normal_feat = final_normal_feat.astype(np.float32)
            return final_normal_feat, final_normal_feat, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_feat = augmented_feat.astype(np.float32)
            final_normal_feat = final_normal_feat.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_feat = msk * augmented_feat + (1-msk)*final_normal_feat
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return final_normal_feat, augmented_feat, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path) # H,W,C
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        # do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
        #     image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        normal_feat, augmented_feat, anomaly_mask, has_anomaly = self.augment_image(image)
        normal_feat = np.transpose(normal_feat, (2, 0, 1))
        augmented_feat = np.transpose(augmented_feat, (2, 0, 1))
        #image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, normal_feat, augmented_feat, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        # Todo
        # 修改这里 让这里返回需要的信息
        # 需要的信息包括： 正常图片 正常图片的特征 异常图片的特征 是否包含异常的01标签 异常mask区域
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, normal_feat, augmented_feat, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,"normal_feat": normal_feat,
                  'augmented_feat': augmented_feat, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample

if __name__=='__main__':
    dataset = MVTecDRAEMTrainDataset("D:\projects\Datasets\Mvtec\\transistor\\train\good", 'D:\projects\Datasets\dtd\images', resize_shape=[256, 256])
    dataset.augment_image("D:\projects\Datasets\Mvtec\\transistor\\train\good\\001.png", 'D:\projects\Datasets\dtd\images')
