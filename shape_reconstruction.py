import torch
from data_loader import MVTecDRAEMTrainDataset_visa_shape, MVTecDRAEMTrainDatasetFeat, MVTecDRAEMTrainDataset_shape
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork,DiscriminativeSubNetwork, ViTReconstruction
from shape_recons_model import  ShapeRepair_0mask_ratio, ShapeRepair_0mask_ratio_new
from loss import FocalLoss, SSIM, GMSLoss
import os
import torch.nn as nn
import sys

#torch.manual_seed(20)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
os.environ['CUDA_VISIBLE_DEVICES']='0'
def train_on_device(obj_names, args, k ):
    torch.cuda.set_device(args.gpu_id)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'

        # visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

        model = ShapeRepair_0mask_ratio_new(obj_name, depth=8, k=k)
        #model = ViTReconstruction(depth=8)
        model.cuda()
        #model.load_state_dict(torch.load('.\memory_bank.pckl'))

        optimizer = torch.optim.AdamW([{"params": model.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.8, args.epochs * 0.9], gamma=0.2,
                                                   last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        #loss_focal = FocalLoss()
        #loss_GMS = GMSLoss()
        loss_ssim = SSIM()

        dataset = MVTecDRAEMTrainDataset_shape(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.data_path + obj_name + "/train/2Dshape/", resize_shape=[224, 224], dataset=0)
        # print(len(dataset))
        # dataset = MVTecDRAEMTrainDataset_visa_shape(args.data_path + obj_name + "/Data/Images/Normal/", args.anomaly_source_path, args.data_path + obj_name + "/Data/Images/2Dshape/",object_name=obj_name, resize_shape=[224, 224])
        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=8)

        for epoch in range(args.epochs):
            total_loss = 0
            # print("Epoch: "+str(epoch))
            for i_batch, sample_batched in enumerate(dataloader):

                gray_batch = sample_batched["image"].cuda()
                shape_batch = sample_batched["shape_info"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()

                # 抽shape信息
                # shape_img = model(aug_gray_batch, 0.0)
                shape_img = model(aug_gray_batch)
                # 重构图片
                #img = model(aug_gray_batch, 0.5)
                #gray_rec = model(aug_gray_batch)
                # 抽shape信息
                l2_loss = loss_l2(shape_img, shape_batch)
                #gms_loss = loss_GMS(shape_img, shape_batch)
                ssim_loss = loss_ssim(shape_img, shape_batch)
                # 抽取shape
                loss =l2_loss + ssim_loss#(0.7*l2_loss + 0.3*gms_loss)*2
                # 重构图片
                #l2_loss = loss_l2(gray_rec, gray_batch)
                #ssim_loss = loss_ssim(gray_rec, gray_batch)
                #loss = l2_loss + ssim_loss
                total_loss += loss
                optimizer.zero_grad()
                sys.stdout.write(
                        "\r[%s Epoch %d/%d] [Batch %d/%d] [S loss:%f] [l2_loss: %f ssim_loss:%f]  [Average loss: %f]"
                    % (
                        obj_name,
                        epoch,
                        args.epochs,
                        i_batch,
                        len(dataloader),
                        loss,
                        1*l2_loss,
                        1*ssim_loss,
                        total_loss / (i_batch + 1),
                    )
                )
                loss.backward()
                optimizer.step()


            print(" ")
            print("==========")
            scheduler.step()



            if (epoch+1) % 700 == 0:
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+"epoch"+str(epoch)+".pckl"))
                print("model saved! \n")



if __name__=="__main__":
    # python train_on_memory.py --gpu_id 0 --obj_id 5 --lr 0.0001 --bs 8 --epochs 700 --data_path ..\datasets\
    # mvtec\ --anomaly_source_path ..\datasets\dtd\images\ --checkpoint_path .\shape_repair_checkpoints\ --log_path .\logs\
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', default=-1, type=int, required =False)
    parser.add_argument('--bs', action='store', default=16, type=int,required =False)
    parser.add_argument('--lr', action='store', default=0.0001, type=float, required =False)
    parser.add_argument('--epochs', action='store', default=700, type=int, required =False)
    parser.add_argument('--gpu_id', action='store', default=0, type=int, required=False)
    parser.add_argument('--data_path', action='store', default="../datasets/mvtec/" ,type=str, required =False)
    parser.add_argument('--anomaly_source_path', action='store', default="../datasets/dtd/images/", type=str, required =False)
    parser.add_argument('--checkpoint_path', action='store', default='./shape_repair_model/shape_repair_model_abnormalkv_1*l2_1*ssim_layer8_k7_noseed/', type=str, required =False)
    parser.add_argument('--log_path', action='store', default='./logs/', type=str, required=False)
    # parser.add_argument('--visualize', action='store_true')

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
        picked_classes = obj_batch[int(args.obj_id)]
    import time
    with torch.cuda.device(args.gpu_id):
        opt = parser.parse_args()
        print(opt.bs)
        #time.sleep(14400)
        # parser.set_defaults()
        opt = parser.parse_args()
        #train_on_device(picked_classes ,opt)
        # print(opt.dataset_name)
        for k in [7]:
            train_on_device([
                            # '01',
                            # '03'
                            # 'toothbrush',
                            'transistor',
                            # 'bottle',
                            # 'cable',
                            # 'screw',
                            #'zipper',
                            # 'capsule',
                            # 'hazelnut', 'pill', 'metal_nut'
                           #'candle',
                            #'cashew',
                            #'chewinggum', 'fryum', 'macaroni1', 'macaroni2', #'pcb1','capsules',
                           #'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
                           #'zipper', 'capsule', 'hazelnut', 'pill', 'metal_nut', 'bottle',
                           #'toothbrush', 'transistor', 'cable','screw'
                           #'transistor'
                           ], opt, k)# pill, metal_nut, bottle, wood, tile, leather,grid

