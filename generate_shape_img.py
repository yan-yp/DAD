import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset, MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, ViTReconstruction
from shape_recons_model import ShapeRepair_0mask_ratio, ShapeRepair_0mask_ratio_new
from Unet_transformer import TransReconstructiveSubNetwork
import os
import cv2
import PIL
import torchvision.transforms as T
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork, SwinReconstructiveSubNetwork, \
    ShapeReconstructiveSubNetwork
from PIL import Image
import torchvision.transforms as transforms
from loss import GMSLoss
from ACC_UNet import ACC_UNet
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(3407)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test(obj_names, mvtec_path, checkpoint_path, base_model_name, time):

    for obj_name in obj_names:
        img_dim = 256
        k = 7

        model_shape = ShapeRepair_0mask_ratio_new(obj_name, 8, k, True)

        model_shape.cuda()
        model_shape.load_state_dict(torch.load(
            '/data/CH/ChenHao/projects/DRAEM/shape_repair_model/shape_repair_model_abnormalkv_1*l2_1*ssim_layer8_k' + str(
                k) + '_noseed/DRAEM_test_0.0001_700_bs16_' + obj_name + '_epoch699.pckl'))
        model_shape.eval()

        # dataset = MVTecDRAEMTrainDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim],)
        # dataset =
        # MVTecDRAEMTrainDataset_shape("../datasets/mvtec/cable/train/good/", "../datasets/dtd/images",
        #                                  "../datasets/mvtec/cable/train/2Dshape/", resize_shape=[224, 224])
        # dataset = MVTecDRAEMTestDataset("/data/public-dataset/VisA_pytorch/1cls/"+obj_name+"/test", [256, 256], dataset=1)
        dataset = MVTecDRAEMTestDataset("../datasets/mvtec/" + obj_name + "/train", [256, 256], dataset=0)
        # dataset = MVTecDRAEMTrainDataset("../datasets/mvtec/" + obj_name + "/train/good/", "../datasets/dtd/images/",
        #                                  object_name=obj_name, resize_shape=[256, 256], dataset=0)
        # dataset = MVTecDRAEMTestDataset("../datasets/BTech_Dataset_transformed/"+obj_name+"/test", [256, 256], dataset=2)
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)

        count = 0

        for i_batch, sample_batched in enumerate(dataloader):

            gray_batch = sample_batched["image"].cuda()
            # shape_batch = sample_batched["shape_info"].cuda()


            shape_rec = model_shape(gray_batch)
            transf = transforms.Resize((256, 256))
            shape_rec = transf(shape_rec)

            base_path = "./experiment/" + "shape_trainset_ssim/" + obj_name

            if not os.path.exists(base_path):
                os.makedirs(base_path)


            cv2.imwrite(base_path + "/" + str(count) + ".png",
                        np.uint8((gray_batch[0].cpu().permute(1, 2, 0).detach().numpy()) * 255))
            count = count + 1


            image_fixed = Image.fromarray(np.uint8((shape_rec[0].cpu().squeeze(0).detach().numpy()) * 255))
            image_fixed.save(fp=base_path + "/" + str(count) + ".png", format='PNG')
            count = count + 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, action='store', type=int)
    parser.add_argument('--base_model_name', action='store', type=str, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=False)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=False)

    args = parser.parse_args()

    obj_list = ['capsule',
                'bottle',
                # 'carpet',
                # 'leather',
                'pill',
                'transistor',
                # 'tile',
                'cable',
                'zipper',
                'toothbrush',
                'metal_nut',
                'hazelnut',
                'screw',
                # 'grid',
                # 'wood'
                ]
    my_list = [
        'transistor']  # 'fryum', 'candle', 'capsules', 'cashew', 'chewinggum', 'macaroni1', 'macaroni2', 'pcb1','pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    with torch.cuda.device(args.gpu_id):
        # for time in range(1,6):
        test(my_list, args.data_path, args.checkpoint_path, args.base_model_name, 1)
