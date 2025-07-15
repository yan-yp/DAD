import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset, MVTecDRAEMTrainDataset_shape, MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, ViTReconstruction
from shape_recons_model import finetunedmae_ShapeRepair_0mask_ratio, ImageRepair_0mask_ratio
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
from mae_vpt_recons_model import FineTuneMae
from shape_recons_model import MaskedAutoencoderViT
warnings.filterwarnings("ignore")
torch.manual_seed(3407)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def test(obj_names, mvtec_path, checkpoint_path, base_model_name, time):
    # visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

    for obj_name in obj_names:
        img_dim = 256
        k = 7

        # run_name = base_model_name+"_"+obj_name+'_'+'epoch699'
        model = FineTuneMae()
        model.cuda()
        model.load_state_dict(torch.load(
            '/data/CH/ChenHao/projects/DRAEM/image_repair+Draem_model/fine_tune_mae/mvtec_gatedvpt_cnn_finetuned_bs16_mae_epoch49.pth'))
        model.eval()

        # model = MaskedAutoencoderViT()
        # model.cuda()
        # model.load_state_dict(torch.load(
        #     './mae_visualize_vit_large.pth')['model'])
        # model.eval()

        # dataset = MVTecDRAEMTestDataset("../datasets/mvtec/" + obj_name + "/test", [224, 224], dataset=0)

        dataset = MVTecDRAEMTrainDataset("../datasets/mvtec/" + obj_name + "/train/good/", "../datasets/dtd/images",
                                         object_name=obj_name, resize_shape=[224, 224], dataset=0)
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)

        count = 0

        for i_batch, sample_batched in enumerate(dataloader):

            gray_batch = sample_batched["image"].cuda()

            gray_rec, loss = model(gray_batch)
            #_, gray_rec, _ = model(gray_batch)

            print(gray_rec.shape)

            base_path = "./experiment/mae_originalandfinetuned/" + "gatedvpt_cnn_mae_trainset/" + obj_name



            if not os.path.exists(base_path):
                os.makedirs(base_path)


            cv2.imwrite(base_path + "/" + str(count) + ".png",
                        np.uint8((gray_batch[0].cpu().permute(1, 2, 0).detach().numpy()) * 255))
            count = count + 1


            cv2.imwrite(base_path + "/" + str(count) + ".png",
                        np.uint8((gray_rec[0].cpu().permute(1, 2, 0).detach().numpy()) * 255))
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
                'carpet',
                'leather',
                'pill',
                'transistor',
                'tile',
                'cable',
                'zipper',
                'toothbrush',
                'metal_nut',
                'hazelnut',
                'screw',
                'grid',
                'wood'
                ]
    my_list = [
        'transistor']  # 'fryum', 'candle', 'capsules', 'cashew', 'chewinggum', 'macaroni1', 'macaroni2', 'pcb1','pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    with torch.cuda.device(args.gpu_id):
        # for time in range(1,6):
        test(obj_list, args.data_path, args.checkpoint_path, args.base_model_name, 1)
