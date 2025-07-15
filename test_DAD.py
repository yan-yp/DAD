import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset_visa, MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from model_unet import ViTReconstruction, ReconstructiveSubNetwork, DiscriminativeSubNetwork, ShapeReconstructiveSubNetwork
import os
from shape_recons_model import ShapeRepair_0mask_ratio, ShapeRepair_0mask_ratio_new
import torchvision.transforms as transforms
from bisect import bisect_left
from typing import Any, Callable, List, Optional, Tuple
from torchmetrics import Metric
#from torchmetrics.functional import auc, roc
from torchmetrics.utilities.data import dim_zero_cat
from anomalib.utils.metrics.pro import (
    connected_components_cpu,
    connected_components_gpu,
)
from torch import Tensor
from ACC_UNet import ACC_UNet
import torch.nn as nn
from PIL import Image
from loss import FocalLoss, SSIM, GMSLoss
import warnings
warnings.filterwarnings("ignore")

def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./visa_outputs/'):
        os.makedirs('./visa_outputs/')

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./visa_outputs/visa.txt",'a+') as file:
        file.write(fin_str)
from skimage import measure
from sklearn.metrics import auc

import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,precision_recall_curve
import cv2

from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 300) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    #print(set(masks.flatten()))
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"

    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        #df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame({"pro": [mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

class IAPS(Metric):
    """Implementation of the instance average precision (IAP) score in our paper"""

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        ioi_thresh: float = 0.5,
        recall_thresh: float = 0.9,  # the k% of the metric IAP@k in our paper
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state(
            "preds", default=[], dist_reduce_fx="cat"
        )  # pylint: disable=not-callable
        self.add_state(
            "target", default=[], dist_reduce_fx="cat"
        )  # pylint: disable=not-callable
        self.ioi_thresh = ioi_thresh
        self.recall_thresh = recall_thresh

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with new values.
        Args:
            preds (Tensor): predictions of the model
            target (Tensor): ground truth targets
        """
        self.target.append(target)
        self.preds.append(preds)

    def compute(self):
        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)

        # check and prepare target for labeling via kornia
        if target.min() < 0 or target.max() > 1:
            raise ValueError(
                (
                    f"kornia.contrib.connected_components expects input to lie in the interval [0, 1], but found "
                    f"interval was [{target.min()}, {target.max()}]."
                )
            )
        target = target.type(torch.float)  # kornia expects FloatTensor
        if target.is_cuda:
            cca = connected_components_gpu(target)
            print("On cuda now!")
        else:
            cca = connected_components_cpu(target)
        preds = preds.flatten()
        cca = cca.flatten()# shape = 50176
        target = target.flatten()

        labels = cca.unique()[1:] # 不要0
        ins_scores = []

        for label in labels:
            mask = cca == label
            heatmap_ins, _ = preds[mask].sort(descending=True)
            ind = np.int64(self.ioi_thresh * len(heatmap_ins))
            ins_scores.append(float(heatmap_ins[ind]))

        if len(ins_scores) == 0:
            raise Exception("gt_masks all zeros")

        ins_scores.sort()

        recall = []
        precision = []

        for i, score in enumerate(ins_scores):
            recall.append(1 - i / len(ins_scores))
            tp = torch.sum(preds * target >= score)
            tpfp = torch.sum(preds >= score)
            precision.append(float(tp / tpfp))

        for i in range(0, len(precision) - 1):
            precision[i + 1] = max(precision[i + 1], precision[i])
        ap_score = sum(precision) / len(ins_scores)
        recall = recall[::-1]
        precision = precision[::-1]
        k = bisect_left(recall, self.recall_thresh)
        return ap_score, precision[k]


os.environ['CUDA_VISIBLE_DEVICES']='0'
def test(obj_names, mvtec_path, checkpoint_path, base_model_name):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    #obj_aupro_list = []
    obj_iap_list = []
    obj_iap90_list = []

    loss_ssim = SSIM()

    for obj_name in obj_names:
        img_dim = 256
        run_name = base_model_name+"_"+obj_name+'_'+'epoch699'
        k = 7

        model = ShapeReconstructiveSubNetwork(in_channels=4, out_channels=3)
        #model = nn.DataParallel(model)
        model.load_state_dict(torch.load("/data/projects/DAD/shape_repair/layer8_k7_yyp_part1_visa/test_0.0001_700_bs16_"+obj_name+"_epoch699.pckl", map_location='cuda:0'))
        #model.load_state_dict(torch.load("/data/CH/ChenHao/projects/DRAEM/shape_repair+Draem_model/shape_into_discriminative/mask_0/DRAEM_test_0.0001_700_bs8_" + obj_name + "_epoch699.pckl", map_location='cuda:0'))
        model.cuda()
        model.eval()



        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        #model_seg = ACC_UNet(7, 2)
        #model_seg = nn.DataParallel(model_seg)
        model_seg.load_state_dict(torch.load("/data/projects/DAD/shape_repair/layer8_k7_yyp_part1_visa/test_0.0001_700_bs16_"+obj_name+"_epoch699_seg.pckl", map_location='cuda:0'))
        #model_seg.load_state_dict(torch.load("/data/CH/ChenHao/projects/DRAEM/shape_repair+Draem_model/shape_into_discriminative/mask_0/DRAEM_test_0.0001_700_bs8_" + obj_name + "_epoch699_seg.pckl", map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()

        # model_shape = ViTReconstruction(depth=8)
        # model_shape.cuda()
        # model_shape.load_state_dict(torch.load(
        #     '/data/CH/ChenHao/projects/DRAEM/shape_repair_model/shape_repair_model_abnormalkv_1.4*l2_0.6*gms_layer8_k0'+ '_vitshape/DRAEM_test_0.0001_700_bs8_' + obj_name + '_epoch699.pckl'))
        # model_shape.eval()
        # for parameter in model_shape.parameters():
        #     parameter.requires_grad = False

        model_shape = ShapeRepair_0mask_ratio(obj_name, 8, k, True)
        #model_shape = nn.DataParallel(model_shape)
        model_shape.cuda()
        model_shape.load_state_dict(torch.load(
             '/data/projects/DAD/shape_repair_model/shape_repair_model_abnormalkv_1.4*l2_0.6*gms_layer8_k7_visa/DRAEM_test_0.0001_700_bs32_' + obj_name + '_epoch699.pckl'))
        model_shape.eval()

        dataset = MVTecDRAEMTestDataset_visa(mvtec_path + obj_name + "/Data/", object_name=obj_name, resize_shape=[img_dim, img_dim])
        #dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim], dataset=0)
        #dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim], dataset=0)
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []
        #pixel_aupro=[]
        seg_IAPS = IAPS().cpu()        
        
        # index = 0
        # SAM_test_path='./SAM_test2Dshape/k7opening-'+obj_name
        # test_2D = [file for file in os.listdir(SAM_test_path) if file.endswith('.png')]
        # test_2D = sorted(test_2D)

        for i_batch, sample_batched in enumerate(dataloader):

            gray_batch = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            #img_path = sample_batched["img_path"]



            # print('true_mask:', set(true_mask.flatten()))
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
            #print('true_mask shape:', true_mask.detach().numpy().shape)
            shape_batch = model_shape(gray_batch)
            transf = transforms.Resize((256, 256))
            shape_rec = transf(shape_batch)
            
    

            gray_rec = model(gray_batch, shape_rec)

            # joined_in = torch.cat((gray_rec, gray_batch,shape_rec), dim=1)
            # joined_in = torch.cat((gray_rec, gray_batch), dim=1)
            joined_in = torch.cat((gray_rec, gray_batch), dim=1) * shape_rec * model_seg.weight_shape + torch.cat(
                 (gray_rec, gray_batch), dim=1)
            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)
            #t_mask = out_mask_sm[:, 1:, :, :]

            out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()
            #print("out_mask_sm:", out_mask_sm[:,1:,:,:].shape)
            #print(out_mask_sm[:,1:,:,:].detach().cpu().shape, true_mask.detach().cpu().shape) # torch.Size([1, 1, 256, 256]) torch.Size([1, 1, 256, 256])
            seg_IAPS.update(out_mask_sm[:,1:,:,:].detach().cpu(), true_mask.detach().cpu())

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            
            #Todo: image 就是相乘 out_mask_avaraged*testSAM2Dshape
            #testSAM2Dshape = os.path.join(SAM_test_path,str(f"{index:03d}")+"_result.png")
            # print(img_path)
            #
            # print(testSAM2Dshape)
            #index = index+1
            #image = Image.open(testSAM2Dshape).resize((256,256))


            #result_array = np.expand_dims(np.expand_dims(np.array(image), axis=0), axis=0).astype('float')/255# [1, 1, 256, 256]

            #result_array = abs(testSAM2Dshape-shape_batch.cpu().detach().numpy()*255)
            #np.savetxt('resultarray.txt', result_array[0][0], delimiter=' ', fmt='%d')
            #print(testSAM2Dshape.any() > 1, shape_batch)
            ## kernel = np.ones((5, 5), np.uint8)
            ## result_array = np.expand_dims(np.expand_dims(cv2.morphologyEx(result_array[0][0], cv2.MORPH_OPEN, kernel), axis=0), axis=0)/255

            out_mask_averaged = out_mask_averaged#*result_array

            image_score = np.max(out_mask_averaged)
            
            anomaly_score_prediction.append(image_score)

            # if true_mask_cv.any() == 1:
            #     pixel_pro_score = compute_pro(np.where(true_mask_cv.transpose(2, 0, 1) != 0, np.ones_like(true_mask_cv.transpose(2, 0, 1)), np.zeros_like(true_mask_cv.transpose(2, 0, 1))), np.expand_dims(out_mask_cv,axis=0))
            #     pixel_aupro.append(pixel_pro_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten() #+ result_array.flatten()
            #print(type(out_mask_cv), type(result_array), flat_out_mask)
            # print(flat_out_mask.shape)
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        #print('anomaly_score_prediction', anomaly_score_prediction.shape)
        #print('anomaly_score_gt', anomaly_score_gt.shape)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        #print(img_dim, mask_cnt)
        print('total_gt_pixel_scores',total_gt_pixel_scores.shape)
        print('total_pixel_scores',total_pixel_scores.shape)
        # assert 1 == 2
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        
        #fpr, tpr, thresholds = roc_curve(total_gt_pixel_scores, total_pixel_scores)
        #selected_indices = fpr >= 0.3
        #selected_tpr = tpr[selected_indices]
        #selected_fpr = fpr[selected_indices]
        #aupro = auc(selected_fpr, selected_tpr)
        
        iap_seg, iap90_seg = seg_IAPS.compute()

        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        #obj_aupro_list.append(np.mean(pixel_aupro))
        obj_iap_list.append(iap_seg)
        obj_iap90_list.append(iap90_seg)
        print(obj_name)
        print("AUC Image:  " +str(auroc))
        print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Pixel:  " +str(ap_pixel))
        #print('AUPRO: '+ str(np.mean(pixel_aupro)))
        print("iap:  " +str(iap_seg))
        print("iap90:  " +str(iap90_seg))
        print("==============================")

    print(run_name)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))
    #print("aupro mean: "+ str(np.mean(obj_aupro_list)))
    print("iap mean:  " + str(np.mean(obj_iap_list)))
    print("iap90 mean:  " + str(np.mean(obj_iap90_list)))
    #write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)

if __name__=="__main__":
    import argparse
    # python test_DRAEM.py --gpu_id 0 --base_model_name "DRAEM_test_0.0001_700_bs8" --data_path ..\datasets\mvtec\ --checkpoint_path .\shape_recons_mae_cnn\
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--base_model_name', action='store', type=str,default="mvtec", required=False)
    parser.add_argument('--data_path', action='store', type=str, default="../datasets/VisA_20220922/", required=False) # VisA_20220922
    parser.add_argument('--checkpoint_path', action='store', type=str, default="./shape_repair+Draem_model/", required=False)

    args = parser.parse_args()

    obj_list = [ 'capsule',
                 'bottle',
                 #'carpet',
                 #'leather',
                 'pill',
                 'transistor',
                 #'tile',
                 'cable',
                 'zipper',
                 'toothbrush',
                 'metal_nut',
                 'hazelnut',
                 'screw',
                 #'grid',
                 #'wood'
                #'01',
                #'03'
                ]
    mylist=[#'toothbrush', 'transistor','screw', 'cable',
            #'zipper'#, 'capsule', 'hazelnut', 'pill', 'metal_nut', 'bottle'
            'fryum', 'candle', 'cashew', 'chewinggum', 'macaroni1', 'macaroni2','pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
            #, 'pcb1' , 'capsules'
            #'fryum', 'candle', 'capsules', 'cashew', 'chewinggum'
            #'fryum'

    with torch.cuda.device(args.gpu_id):
        test(mylist,args.data_path, args.checkpoint_path, args.base_model_name)









