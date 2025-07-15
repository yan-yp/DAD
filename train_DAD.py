import torch
from data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTrainDatasetFeat, MVTecDRAEMTrainDataset_visa_shape
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ViTReconstruction, ReconstructiveSubNetwork,DiscriminativeSubNetwork, SwinReconstructiveSubNetwork,ShapeReconstructiveSubNetwork
from Unet_transformer import TransReconstructiveSubNetwork
from loss import FocalLoss, SSIM, GMSLoss
from shape_recons_model import ShapeRepair_0mask_ratio, ShapeRepair_0mask_ratio_new
from ACC_UNet import ACC_UNet
import os
import torch.nn as nn
import torchvision.transforms as transforms
import time
import sys
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import cv2
import numpy as np
import copy

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#torch.manual_seed(3407)
os.environ['CUDA_VISIBLE_DEVICES']='0'
def train_on_device(obj_names, args, k):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'

        model = ShapeReconstructiveSubNetwork(in_channels=4, out_channels=3)
        model.cuda()
        model.apply(weights_init)


        model_seg = DiscriminativeSubNetwork(in_channels=7, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)


        # model_shape = ShapeRepair_0mask_ratio(object_name=obj_name, depth=8, k=7)
        # model_shape.cuda()
        # model_shape.load_state_dict(torch.load(
        #     '/data/CH/ChenHao/projects/DRAEM/shape_repair_model/shape_repair_model_abnormalkv_1.4*l2_0.6*gms_layer8_k7_visa/DRAEM_test_0.0001_700_bs32_' + obj_name + '_epoch699.pckl'))
        # model_shape.eval()
        # for parameter in model_shape.parameters():
        #     parameter.requires_grad = False

        model_shape = ViTReconstruction(depth=8)
        model_shape.cuda()
        model_shape.load_state_dict(torch.load(
            '/data/CH/ChenHao/projects/DRAEM/shape_repair_model/shape_repair_model_abnormalkv_1.4*l2_0.6*gms_layer8_k0' + '_vitshape/DRAEM_test_0.0001_700_bs8_' + obj_name + '_epoch699.pckl'))
        model_shape.eval()
        for parameter in model_shape.parameters():
            parameter.requires_grad = False

        #optimizer = torch.optim.Adam([{"params": model_seg.parameters(), "lr": args.lr}])
        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}, {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)
        #scheduler_swin = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_swin, )
        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()
        # loss_GMS = GMSLoss()

        dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path,object_name=obj_name, resize_shape=[256, 256], dataset=0)
        # dataset = MVTecDRAEMTrainDataset_visa_shape(args.data_path + obj_name + "/Data/Images/Normal/",
        #                                             args.anomaly_source_path,
        #                                             args.data_path + obj_name + "/Data/Images/2Dshape/",
        #                                             object_name=obj_name, resize_shape=[256, 256])
        dataloader = DataLoader(dataset, batch_size=args.bs,
                              shuffle=True, num_workers=8)

        n_iter = 0
        for epoch in range(args.epochs):
            total_loss = 0
            # print("Epoch: "+str(epoch))
            for i_batch, sample_batched in enumerate(dataloader):
                #transf_224 = transforms.Resize((224, 224))
                gray_batch = sample_batched["image"].cuda()
                gray_batch_detach = copy.deepcopy(gray_batch)
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                #aug_gray_batch_224 = transf_224(aug_gray_batch)
                shape_rec = model_shape(aug_gray_batch)

                anomaly_mask = sample_batched["anomaly_mask"].cuda()

                transf = transforms.Resize((256, 256))
                shape_rec = transf(shape_rec)
                #shape_rec = gaussian_filter(shape_rec)
                gray_rec = model(aug_gray_batch, shape_rec)
                #gray_rec = model(aug_gray_batch)
                
                #joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)*shape_rec*model_seg.weight_shape + torch.cat((gray_rec, aug_gray_batch), dim=1)
                joined_in = torch.cat((gray_rec, aug_gray_batch, shape_rec), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                l2_loss = loss_l2(gray_rec, gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                # gms_loss = loss_GMS(gray_rec, gray_batch)

                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss =  segment_loss  +   l2_loss + ssim_loss 
                total_loss += loss
                optimizer.zero_grad()

                loss.backward()
                print(gray_batch_detach == gray_batch)
                optimizer.step()

                sys.stdout.write(
                        "\r[%s Epoch %d/%d] [Batch %d/%d] [S loss:%f] [l2:%f ssim:%f segment:%f]  [Average loss: %f]"
                    % (
                        obj_name,
                        epoch,
                        args.epochs,
                        i_batch,
                        len(dataloader),
                        loss,
                        l2_loss,
                        ssim_loss,
                        segment_loss,
                        total_loss/(i_batch+1),
                    )
                )


                n_iter +=1
            print(" ")
            print("==========")
            scheduler.step()

            if (epoch+1) % 700 == 0:
                #torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+"epoch"+str(epoch)+str(k)+".pckl"))
                #torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"epoch"+str(epoch)+str(k)+"_seg.pckl"))
                torch.save(model.state_dict(),
                           os.path.join(args.checkpoint_path, run_name + "epoch" + str(epoch) + ".pckl"))
                torch.save(model_seg.state_dict(),
                           os.path.join(args.checkpoint_path, run_name + "epoch" + str(epoch) + "_seg.pckl"))

if __name__=="__main__":
    import subprocess
    
    # subprocess.run(['python', './shape_reconstruction.py'])

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--local_rank', default=-1, type=int)
    
    args = parser.parse_args()

    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    if int(args.obj_id) == -1:
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
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[args.obj_id]
    #time.sleep(7200)
    with torch.cuda.device(args.gpu_id):
    
        #time.sleep(13350)
        for k in [7]:
            train_on_device([
                        #'screw',
                        #'transistor',
                        #'toothbrush', 
                        #'cable',
                        #'zipper', 'capsule', 'hazelnut', 'pill', 'metal_nut', 'bottle'
                        # 'fryum', 'candle',  'cashew', 'chewinggum', 'macaroni1',
                        # 'macaroni2',
                        # 'pcb2',
                        # 'pcb3',
                        # 'pcb4', 'pipe_fryum'#'capsules','pcb1',
                        #'zipper', 'capsule', 'hazelnut', 'pill', 'metal_nut', 'bottle',
                        #'toothbrush', 'transistor', 'cable','screw' 
                        #'toothbrush', 'transistor', 'cable','screw'
                       # 'transistor',
                       # 'toothbrush',
                       # 'cable',
                       # 'screw',
                        'zipper',
                       #
                       # 'capsule',
                       # 'hazelnut',
                       # 'pill',
                       # 'metal_nut',
                       # 'bottle'
#                         '01',
#                         '03'
         		], args, k)
