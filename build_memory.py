import torch
from data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTrainDataset_visa_shape, MVTecDRAEMTrainDatasetFeat, MVTecDRAEMTrainDataset_shape
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork,DiscriminativeSubNetwork, SwinReconstructiveSubNetwork
from shape_recons_model import finetunedmae_ShapeExtractor_0mask_ratio
from loss import FocalLoss, SSIM, GMSLoss
import os
import torch.nn as nn
import sys
torch.manual_seed(3407)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def build_memory():
    objects1 = ['zipper', 'capsule', 'hazelnut', 'pill', 'metal_nut', 'bottle','transistor','screw','toothbrush','cable']
    # objects = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    objects2 = ['carpet',
                'leather',
                'tile',
                'grid',
                'wood']
    objects = objects1+objects2
    for object in objects:
        model = finetunedmae_ShapeExtractor_0mask_ratio(object, True)
        model.cuda()
        model.eval()
        dataset = MVTecDRAEMTrainDataset("../datasets/mvtec/"+object+"/train/good/", "../datasets/dtd/images",
                                          object_name=object, resize_shape=[224, 224] )

        #dataset = MVTecDRAEMTrainDataset_visa_shape("../datasets/VisA_20220922/"+object+"/Data/Images/Normal/", "../datasets/dtd/images",
         #                                  "../datasets/VisA_20220922/"+object+"/Data/Images/2Dshape/", resize_shape=[224, 224])

        dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=True, num_workers=8)
        print('正在处理:', object)
        for epoch in range(3):
            print(epoch)
            for i_batch, sample_batched in enumerate(dataloader):
                model.store_to_memory(sample_batched["image"].cuda())

        model.subsample()
        print(model.memory_bank_normal.shape)
        torch.save(model.state_dict(), os.path.join("./memory_bank/finetuned_mae_mvtec_memory_bank_"+object+"_0mask_ratio_epoch3.pckl"))

if __name__=="__main__":
    build_memory()
# transistor 240,802816
# screw 320,802816
# capsule 219,802816

