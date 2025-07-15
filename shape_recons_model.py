__author__ = 'Hao Chen'

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from torch import Tensor, nn
from util.pos_embed import get_2d_sincos_pos_embed

from functools import partial

import torch
import cv2
import torchvision.transforms as tt
from timm.models.vision_transformer import PatchEmbed, Block, VisionTransformer

from util.pos_embed import get_2d_sincos_pos_embed

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from swin_transformer import SwinTransformer, BasicLayer
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)
#
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.para = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)
#
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
#
#         dots = torch.matmul(q, k.transpose(-1, -2)) * (self.scale * self.para)# -torch.diag(torch.tensor([-100.0] * min(tensor.size(0), tensor.size(1))))
#         # print("dots:", dots.shape) # [8, 12, 197, 197]
#         reshaped_tensor = dots.view(dots.shape[0] * dots.shape[1], 197, 197)
#         diagonal = torch.diag(torch.tensor([100.0] * min(reshaped_tensor.size(-2), reshaped_tensor.size(-1)))).cuda()
#         result_tensor = reshaped_tensor - diagonal.unsqueeze(0)
#         dots = result_tensor.view(dots.shape[0], dots.shape[1], 197, 197)
#
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, num_patches, dim, depth, heads, mlp_dim,  dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        return x
# torch.cuda.set_device(0)

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from uniformer import CBlock, SABlock

class ImageRepair_0mask_ratio(nn.Module):
    def __init__(self, object_name, depth=8, k=7, test=False):
        super().__init__()
        self.k = k
        self.test = test
        # --------------------------------------------------------------------------
        # shape search
        net = ShapeExtractor_0mask_ratio(object_name).to("cpu")
        if object_name in ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1',
                           'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']:
            net.load_state_dict(torch.load("./memory_bank/visa_memory_bank_" + object_name + "_0mask_ratio_epoch1.pckl",
                                           map_location='cpu'))
        else:
            net.load_state_dict(
                torch.load("./memory_bank/mvtec_memory_bank_" + object_name + "_0mask_ratio_epoch3.pckl",
                           map_location='cpu'))
        self.memory_bank = net.state_dict()['memory_bank_normal'].cuda()
        for parameter in net.parameters():
            parameter.requires_grad = False
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # feature extractor
        self.feature_extractor = MaskedAutoencoderViT()
        # weights_dict = torch.load("./mae_visualize_vit_large.pth")

        weights_dict = torch.load("./mae_visualize_vit_large.pth", map_location='cpu')
        self.feature_extractor.load_state_dict(weights_dict['model'], strict=False)
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # shape repair
        self.softmax = nn.Softmax(dim=-1)
        # self.linear_query = nn.Linear(512, 512)
        self.linear_key = nn.Linear(512, 512)
        self.linear_value = nn.Linear(512, 512)

        self.reduce_dim = nn.Linear(512*8, 768)

        # 抽取shape时depth是8 重构图片时depth是12
        # self.swin = BasicLayer(dim=768, input_resolution=[14,14], depth=depth,num_heads=12, window_size=7)
        self.vit = ViT(num_patches=196, dim=768, depth=depth, heads=12, mlp_dim=768 * 4, dropout=0.)
        print('depth:' + str(depth))
        print('k:' + str(k))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)


        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=512, num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=norm_layer)
            for i in range(4)])

        self.blocks2 = nn.ModuleList([
            SABlock(
                dim=512, num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=norm_layer)
            for i in range(8)])

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 )
        imgs: (N, 3, H, W)
        """
        # print(x.shape)
        # x = cv2.resize(x, (224, 224))
        p = 16  # self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)  # 14
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))  # [N, 14, 14, 16, 16, 1]
        x = torch.einsum('nhwpqc->nchpwq', x)  # [N, 1, 14, 16, 14, 16]
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs  # [B, 1, 224, 224]

    def unpatchify_256(self, x):
        """
        x: (N, L, 3*patch_size**2 )
        imgs: (N, 3, H, W)
        """
        # print(x.shape)
        # x = cv2.resize(x, (224, 224))
        p = 16  # self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)  # 14
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))  # [N, 14, 14, 16, 16, 3]
        x = torch.einsum('nhwpqc->nchpwq', x)  # [N, 3, 14, 16, 14, 16]
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs  # [B, 3, 224, 224]

    def search_bank(self, img, mask_ratio=0.0):
        img = tt.Resize((224, 224))(img)

        x1, x2, mask1, mask2, ids_restore = self.feature_extractor.forward_encoder(img, mask_ratio)
        x1, _, multi_feat1, _ = self.feature_extractor.forward_decoder(x1, x2, ids_restore)

        feat0 = []
        for i, layer_feat in enumerate(zip(multi_feat1)):  # [1, 197, 512]

            layer_feat0 = layer_feat[0][:, 1:, :]  # [1, 196, 512]

            feat0.append(layer_feat0)  # 8* [1, 196, 512]
        feats0 = torch.cat(feat0, dim=1)  # [1, 8*196, 512 ]
        layer_feat_a = feats0.reshape(-1, 8 * 196 * 512)  # [B, 8*196*512]

        # 扩展输入向量的形状，使其与 memory bank 的大小匹配
        layer_feat_1 = layer_feat_a.unsqueeze(1)  # [8, 1, 802816]

        # 计算输入向量与 memory bank 中所有向量之间的余弦相似度
        # similarities_a = F.cosine_similarity(layer_feat_1, self.memory_bank, dim=2)
        # test
        if self.test:
            similarities_a = F.cosine_similarity(layer_feat_1[:, :, :], self.memory_bank, dim=2)

        else:
            # mapreduce here when training
            batch = layer_feat_1.shape[0]
            # print('device:', layer_feat_1.get_device())
            # print(layer_feat_1.get_device()== 0)
            if layer_feat_1.get_device() == 8:
                self.memory_bank = self.memory_bank.to('cuda:0')
                similarities_a1 = F.cosine_similarity(layer_feat_1[0:1, :, :], self.memory_bank, dim=2)
                similarities_a2 = F.cosine_similarity(layer_feat_1[1:2, :, :], self.memory_bank, dim=2)
                similarities_a3 = F.cosine_similarity(layer_feat_1[2:3, :, :], self.memory_bank, dim=2)
                similarities_a4 = F.cosine_similarity(layer_feat_1[3:4, :, :], self.memory_bank, dim=2)
                # print(similarities_a1.shape) [4,320]
                similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3, similarities_a4), dim=0)
                # similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3), dim=0)
            if layer_feat_1.get_device() == 9:
                self.memory_bank = self.memory_bank.to('cuda:1')
                similarities_a1 = F.cosine_similarity(layer_feat_1[0:1, :, :], self.memory_bank, dim=2)
                similarities_a2 = F.cosine_similarity(layer_feat_1[1:2, :, :], self.memory_bank, dim=2)
                similarities_a3 = F.cosine_similarity(layer_feat_1[2:3, :, :], self.memory_bank, dim=2)
                similarities_a4 = F.cosine_similarity(layer_feat_1[3:4, :, :], self.memory_bank, dim=2)
                # print(similarities_a1.shape) [4,320]
                # similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3), dim=0)
                similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3, similarities_a4), dim=0)
            print(self.memory_bank.shape, layer_feat_1.shape)
            similarities_a1 = F.cosine_similarity(layer_feat_1[0:batch // 4, :, :], self.memory_bank, dim=2)
            similarities_a2 = F.cosine_similarity(layer_feat_1[batch // 4:batch // 2, :, :], self.memory_bank, dim=2)
            similarities_a3 = F.cosine_similarity(layer_feat_1[batch // 2:batch * 3 // 4, :, :], self.memory_bank,
                                                  dim=2)
            similarities_a4 = F.cosine_similarity(layer_feat_1[batch * 3 // 4:batch, :, :], self.memory_bank, dim=2)
            similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3, similarities_a4), dim=0)
        # 根据余弦相似度降序排列，选择前3个最接近的向量的索引
        _, indices_a = torch.topk(similarities_a, k=self.k, dim=1, largest=True)

        # 根据索引从Memory Bank中获取对应的向量
        # print(indices_a.get_device())

        closest_vectors_a = self.memory_bank[indices_a]  # [8, k, 802816]

        return layer_feat_a, closest_vectors_a  # [8, 802816] [8, k, 802816]

    def forward(self, img, mask_ratio=0.0):
        layer_feat_a, closest_vectors_a = self.search_bank(img, mask_ratio)
        m = layer_feat_a.shape[0]  # 8
        k = self.k
        closest_vectors_a = closest_vectors_a.reshape(m, 8, k * 196, 512)  # feature from memory bank
        layer_feat_a = layer_feat_a.reshape(m, 8, 196, 512)  # feature from input

        closest_vectors_a = torch.cat((closest_vectors_a, layer_feat_a), dim=2)  # [m, 8, (k+1)*196, 512]
        # closest_vectors_a = self.generate_relative_absolute_positional_encoding(closest_vectors_a.cuda(), layer_feat_a.cuda())
        assert closest_vectors_a.shape[2] == (k + 1) * 196
        # 计算 normal-abnormal-attention
        # query_a = self.linear_query(layer_feat_a)
        key_a = self.linear_key(closest_vectors_a)  # generate key according to closest_vectors_a
        value_a = self.linear_value(closest_vectors_a)  # generate value according to closest_vectors_a

        score1 = layer_feat_a @ key_a.permute(0, 1, 3, 2)
        # score1 = query_a @ key_a.permute(0, 1, 3, 2)
        score1 = self.softmax(score1) * 512 ** (-0.5)

        # [n, 8, 196, 512]
        out = score1 @ value_a  # output of first layer, which calculate normal-abnormal attention instead of self-attention

        out = out.reshape(out.shape[0]*out.shape[1], 196, 512).reshape(-1,14,14,512).permute(0, 3, 1, 2) # [8*n, 512, 14, 14]
        for blk in self.blocks1:
            out = blk(out) # [8*n, 512, 14, 14]
        for blk in self.blocks2:
            out = blk(out) # [8*n, 512, 14, 14]
        # 将互补的特征图乘0.5后相加  考虑直接相加？
        out = out.permute(0, 2, 3, 1).reshape(out.shape[0]//8, 14,14,512*8).reshape(out.shape[0]//8, 14*14,512*8)  # 这里的reshape相当于 卷积里面的concat [n, 196, 8*512]
        #print(out.shape)
        out = self.reduce_dim(out)  # [n, 196, 768]

        # ====ViT重构===
        out = self.vit(out)  # [n, 197, 768]

        # ===重构图片===
        out = self.unpatchify_256(out[:, 1:, :])  # [n, 3, 224, 224]


        return out

from mae_vpt_recons_model import FineTuneMae


class finetunedmae_ShapeRepair_0mask_ratio(nn.Module):
    def __init__(self, object_name, depth=8, k=7, test=False):
        super().__init__()
        self.k=k
        self.test=test
        # --------------------------------------------------------------------------
        # shape search
        #net = ShapeExtractor_0mask_ratio(object_name).to("cpu")
        net = finetunedmae_ShapeExtractor_0mask_ratio(object_name).to("cpu")
        if object_name in ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1',
                           'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']:
            net.load_state_dict(torch.load("./memory_bank/visa_memory_bank_"+object_name+"_0mask_ratio_epoch1.pckl", map_location='cpu'))
        else:
            net.load_state_dict(torch.load("./memory_bank/finetuned_mae_mvtec_memory_bank_" + object_name + "_0mask_ratio_epoch3.pckl",
                                           map_location='cpu'))
        self.memory_bank = net.state_dict()['memory_bank_normal'].cuda()
        for parameter in net.parameters():
            parameter.requires_grad = False
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # feature extractor
        #self.feature_extractor = MaskedAutoencoderViT()
        self.feature_extractor = FineTuneMae()

        #weights_dict = torch.load("./mae_visualize_vit_large.pth")
        weights_dict = torch.load("./image_repair+Draem_model/fine_tune_mae/mvtec_finetuned_bs16_mae_epoch49.pth",  map_location='cpu')
        #self.feature_extractor.load_state_dict(weights_dict['model'], strict=True)
        self.feature_extractor.load_state_dict(weights_dict, strict=True)
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        # --------------------------------------------------------------------------


        # --------------------------------------------------------------------------
        # shape repair
        self.softmax = nn.Softmax(dim=-1)
        #self.linear_query = nn.Linear(512, 512)
        self.linear_key = nn.Linear(512, 512)
        self.linear_value = nn.Linear(512, 512)

        self.reduce_dim = nn.Linear(512*8, 768)

        # 抽取shape时depth是8 重构图片时depth是12
        # self.swin = BasicLayer(dim=768, input_resolution=[14,14], depth=depth,num_heads=12, window_size=7)
        #self.vit = ViT(num_patches=196, dim=768, depth=depth, heads=12, mlp_dim=768*4, dropout=0.)
        print('depth:'+str(depth))
        print('k:'+str(k))

        # early cnn
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.vit = ViT(num_patches=196, dim=768, depth=depth, heads=12, mlp_dim=768 * 4, dropout=0.)
        # 最后输出3通道再输出一通道作为shape试试
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

        # 两层cnn输出3通道图片
        #self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # 加BN relu
        #self.bn = nn.BatchNorm2d(3)
        #self.relu = nn.ReLU(inplace=True)
        #expandcnn
        #self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        #self.conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
	#使用更大的卷积核试试
        #self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3)
        #self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, stride=1, padding=3)
        # --------------------------------------------------------------------------
	
	#使用两层重可参数化的大卷积核加小卷积核试试
        #self.conv_big_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3)
        #self.conv_big_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, stride=1, padding=3)
        #self.conv_small_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        #self.conv_small_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)        
        #self.big_bn1 = nn.BatchNorm2d(3)
        #self.small_bn1 = nn.BatchNorm2d(3)
	
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 )
        imgs: (N, 3, H, W)
        """
        # print(x.shape)
        # x = cv2.resize(x, (224, 224))
        p = 16 # self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5) # 14
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1)) # [N, 14, 14, 16, 16, 1]
        x = torch.einsum('nhwpqc->nchpwq', x) # [N, 1, 14, 16, 14, 16]
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs # [B, 1, 224, 224]

    def unpatchify_256(self, x):
        """
        x: (N, L, 3*patch_size**2 )
        imgs: (N, 3, H, W)
        """
        # print(x.shape)
        # x = cv2.resize(x, (224, 224))
        p = 16 # self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5) # 14
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3)) # [N, 14, 14, 16, 16, 3]
        x = torch.einsum('nhwpqc->nchpwq', x) # [N, 3, 14, 16, 14, 16]
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs # [B, 3, 224, 224]
    def search_bank(self, img, mask_ratio=0.0):
        img = tt.Resize((224, 224))(img)

        # x1, x2, mask1, mask2, ids_restore = self.feature_extractor.forward_encoder(img, mask_ratio)
        # x1, _, multi_feat1, _ = self.feature_extractor.forward_decoder(x1, x2, ids_restore)
        latent = self.feature_extractor.forward_encoder(img, mask_ratio)
        multi_feat1 = self.feature_extractor.forward_decoder(latent)

        feat0 = []
        for i, layer_feat in enumerate(zip(multi_feat1)): # [1, 197, 512]

            layer_feat0 = layer_feat[0][:, 1:, :] # [1, 196, 512]


            feat0.append(layer_feat0) # 8* [1, 196, 512]
        feats0 = torch.cat(feat0, dim=1)  # [1, 8*196, 512 ]
        layer_feat_a = feats0.reshape(-1, 8 * 196 * 512)  # [B, 8*196*512]


        # 扩展输入向量的形状，使其与 memory bank 的大小匹配
        layer_feat_1 = layer_feat_a.unsqueeze(1)  # [8, 1, 802816]

        # 计算输入向量与 memory bank 中所有向量之间的余弦相似度
        # similarities_a = F.cosine_similarity(layer_feat_1, self.memory_bank, dim=2)
        #test
        if self.test:
            similarities_a = F.cosine_similarity(layer_feat_1[:, :, :], self.memory_bank, dim=2)

        else:
            # mapreduce here when training
            batch = layer_feat_1.shape[0]
            #print('device:', layer_feat_1.get_device())
            #print(layer_feat_1.get_device()== 0)
            if layer_feat_1.get_device() == 8 :
                self.memory_bank = self.memory_bank.to('cuda:0')
                similarities_a1 = F.cosine_similarity(layer_feat_1[0:1, :, :], self.memory_bank, dim=2)
                similarities_a2 = F.cosine_similarity(layer_feat_1[1:2, :, :], self.memory_bank, dim=2)
                similarities_a3 = F.cosine_similarity(layer_feat_1[2:3, :, :], self.memory_bank, dim=2)
                similarities_a4 = F.cosine_similarity(layer_feat_1[3:4, :, :], self.memory_bank, dim=2)
            #print(similarities_a1.shape) [4,320]
                similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3, similarities_a4), dim=0)
                #similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3), dim=0)
            if layer_feat_1.get_device() == 9 :
                self.memory_bank = self.memory_bank.to('cuda:1')
                similarities_a1 = F.cosine_similarity(layer_feat_1[0:1, :, :], self.memory_bank, dim=2)
                similarities_a2 = F.cosine_similarity(layer_feat_1[1:2, :, :], self.memory_bank, dim=2)
                similarities_a3 = F.cosine_similarity(layer_feat_1[2:3, :, :], self.memory_bank, dim=2)
                similarities_a4 = F.cosine_similarity(layer_feat_1[3:4, :, :], self.memory_bank, dim=2)
             #print(similarities_a1.shape) [4,320]
                #similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3), dim=0)
                similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3, similarities_a4), dim=0)

            #print(output.shape)
            #print(self.memory_bank.shape, layer_feat_1.shape, layer_feat_1[0:batch//4, :, :].shape)
            similarities_a1 = F.cosine_similarity(layer_feat_1[0:batch//4, :, :], self.memory_bank, dim=2)
            similarities_a2 = F.cosine_similarity(layer_feat_1[batch//4:batch//2, :, :], self.memory_bank, dim=2)
            similarities_a3 = F.cosine_similarity(layer_feat_1[batch//2:batch*3//4, :, :], self.memory_bank, dim=2)
            similarities_a4 = F.cosine_similarity(layer_feat_1[batch*3//4:batch, :, :], self.memory_bank, dim=2)
            similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3, similarities_a4), dim=0)
            #print(similarities_a.shape)
         # 根据余弦相似度降序排列，选择前3个最接近的向量的索引
        _, indices_a = torch.topk(similarities_a, k=self.k, dim=1, largest=True)
        #print(indices_a.shape)
        # 根据索引从Memory Bank中获取对应的向量
        #print(indices_a.get_device())
        
        closest_vectors_a = self.memory_bank[indices_a] # [8, k, 802816]

        return layer_feat_a, closest_vectors_a # [8, 802816] [8, k, 802816]
    
    

    def forward(self, img, mask_ratio=0.0):
        layer_feat_a, closest_vectors_a = self.search_bank(img, mask_ratio)
        m = layer_feat_a.shape[0] # 8
        k = self.k
        closest_vectors_a = closest_vectors_a.reshape(m, 8, k*196, 512) # feature from memory bank
        layer_feat_a = layer_feat_a.reshape(m, 8, 196, 512) # feature from input
        # layer_feat_a = layer_feat_a+ torch.full_like(layer_feat_a,1)
        # 做key value计算时 也将异常图片加进去
	#layer_feat_a = layer_feat_a + torch.full_like(layer_feat_a,1)
        closest_vectors_a = torch.cat((closest_vectors_a, layer_feat_a), dim=2) # [m, 8, (k+1)*196, 512]
        #closest_vectors_a = self.generate_relative_absolute_positional_encoding(closest_vectors_a.cuda(), layer_feat_a.cuda())
        assert closest_vectors_a.shape[2] == (k+1)*196
        # 计算 normal-abnormal-attention
        #query_a = self.linear_query(layer_feat_a)
        key_a = self.linear_key(closest_vectors_a) # generate key according to closest_vectors_a
        value_a = self.linear_value(closest_vectors_a) # generate value according to closest_vectors_a

        score1 = layer_feat_a @ key_a.permute(0, 1, 3, 2)
        #score1 = query_a @ key_a.permute(0, 1, 3, 2) 
        score1 = self.softmax(score1) * 512**(-0.5)

        out = score1 @ value_a # output of first layer, which calculate normal-abnormal attention instead of self-attention

        # 将互补的特征图乘0.5后相加  考虑直接相加？
        out = out.permute(0, 2, 1, 3).reshape(-1, 196, 512 * 8) # 这里的reshape相当于 卷积里面的concat [n, 196, 8*512]
        out = self.reduce_dim(out) # [n, 196, 768] [n, 196, 192]

        # early cnn
        out = out.reshape(-1, 14, 14, 768).permute(0, 3, 1, 2)# [8, 768, 14, 14]
        out = self.conv1(out) # [8, 768, 14, 14]
        out = out.reshape(-1, 768, 14*14).permute(0, 2, 1)
        # ====ViT重构===
        out = self.vit(out) # [n, 197, 768] [n, 197, 192]


        # 最终定稿 不直接reduce dim了  最后用3通道图变1通道图来修shape
        # out = self.unpatchify_256(out[:, 1:, : ]) # [n, 3, 224, 224]
        # out = self.conv2(self.conv1(out))
        # 加bn 和 relu
        #out = self.conv1(out)
        #out = self.bn(out)
        #out = self.relu(out)
        #out = self.conv2(out)
        #三个卷积
        #out = self.conv3(self.conv2(self.conv1(out)))
	# ===抽取shape===
        #out = self.reduce_dim2(out[:, 1:, : ]) # [n, 196, 256]

        #out = self.unpatchify(out) # [n, 1, 224, 224]
        #out = self.fin_2(self.fin_1(out))

        # ===重构图片===
        out = self.unpatchify_256(out[:, 1:, : ])# [n, 3, 224, 224]

        # ===swin重构图片===
        #out = self.swin(out)
        #out = self.unpatchify_256(out)

        #可重参数化卷积
        #out = self.big_bn1(self.conv_big_1(out))+ self.small_bn1(self.conv_small_1(out))
        #out = self.conv_big_2(out)+ self.conv_small_2(out) 
        
        return out


class ShapeRepair_0mask_ratio(nn.Module):
    def __init__(self, object_name, depth=8, k=7, test=False):
        super().__init__()
        self.k = k
        self.test = test
        # --------------------------------------------------------------------------
        # shape search
        net = ShapeExtractor_0mask_ratio(object_name).to("cpu")
        if object_name in ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1',
                           'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']:
            net.load_state_dict(torch.load("./memory_bank/visa_memory_bank_" + object_name + "_0mask_ratio_epoch1.pckl",
                                           map_location='cpu'))
        else:
            net.load_state_dict(
                torch.load("./memory_bank/mvtec_memory_bank_" + object_name + "_0mask_ratio_epoch3.pckl",
                           map_location='cpu'))
        self.memory_bank = net.state_dict()['memory_bank_normal'].cuda()
        for parameter in net.parameters():
            parameter.requires_grad = False
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # feature extractor
        self.feature_extractor = MaskedAutoencoderViT()
        # weights_dict = torch.load("./mae_visualize_vit_large.pth")

        weights_dict = torch.load("./mae_visualize_vit_large.pth", map_location='cpu')
        self.feature_extractor.load_state_dict(weights_dict['model'], strict=False)
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # shape repair
        self.softmax = nn.Softmax(dim=-1)
        # self.linear_query = nn.Linear(512, 512)
        self.linear_key = nn.Linear(512, 512)
        self.linear_value = nn.Linear(512, 512)

        self.reduce_dim = nn.Linear(512 * 8, 768)

        # 抽取shape时depth是8 重构图片时depth是12
        # self.swin = BasicLayer(dim=768, input_resolution=[14,14], depth=depth,num_heads=12, window_size=7)
        self.vit = ViT(num_patches=196, dim=768, depth=depth, heads=12, mlp_dim=768*4, dropout=0.)
        print('depth:' + str(depth))
        print('k:' + str(k))

        # early cnn
        # self.conv1 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3, stride=1, padding=1)
        # self.vit = ViT(num_patches=196, dim=768, depth=depth, heads=12, mlp_dim=768 * 4, dropout=0.)
        # 最后输出3通道再输出一通道作为shape试试
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

        # 两层cnn输出3通道图片
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # 加BN relu
        # self.bn = nn.BatchNorm2d(3)
        # self.relu = nn.ReLU(inplace=True)
        # expandcnn
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

    # 使用更大的卷积核试试
    # self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3)
    # self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, stride=1, padding=3)
    # --------------------------------------------------------------------------

    # 使用两层重可参数化的大卷积核加小卷积核试试
    # self.conv_big_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3)
    # self.conv_big_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, stride=1, padding=3)
    # self.conv_small_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    # self.conv_small_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
    # self.big_bn1 = nn.BatchNorm2d(3)
    # self.small_bn1 = nn.BatchNorm2d(3)

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 )
        imgs: (N, 3, H, W)
        """
        # print(x.shape)
        # x = cv2.resize(x, (224, 224))
        p = 16  # self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)  # 14
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))  # [N, 14, 14, 16, 16, 1]
        x = torch.einsum('nhwpqc->nchpwq', x)  # [N, 1, 14, 16, 14, 16]
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs  # [B, 1, 224, 224]

    def unpatchify_256(self, x):
        """
        x: (N, L, 3*patch_size**2 )
        imgs: (N, 3, H, W)
        """
        # print(x.shape)
        # x = cv2.resize(x, (224, 224))
        p = 16  # self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)  # 14
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))  # [N, 14, 14, 16, 16, 3]
        x = torch.einsum('nhwpqc->nchpwq', x)  # [N, 3, 14, 16, 14, 16]
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs  # [B, 3, 224, 224]

    def search_bank(self, img, mask_ratio=0.0):
        img = tt.Resize((224, 224))(img)

        x1, x2, mask1, mask2, ids_restore = self.feature_extractor.forward_encoder(img, mask_ratio)
        x1, _, multi_feat1, _ = self.feature_extractor.forward_decoder(x1, x2, ids_restore)

        feat0 = []
        for i, layer_feat in enumerate(zip(multi_feat1)):  # [1, 197, 512]

            layer_feat0 = layer_feat[0][:, 1:, :]  # [1, 196, 512]

            feat0.append(layer_feat0)  # 8* [1, 196, 512]
        feats0 = torch.cat(feat0, dim=1)  # [1, 8*196, 512 ]
        layer_feat_a = feats0.reshape(-1, 8 * 196 * 512)  # [B, 8*196*512]

        # 扩展输入向量的形状，使其与 memory bank 的大小匹配
        layer_feat_1 = layer_feat_a.unsqueeze(1)  # [8, 1, 802816]

        # 计算输入向量与 memory bank 中所有向量之间的余弦相似度
        # similarities_a = F.cosine_similarity(layer_feat_1, self.memory_bank, dim=2)
        # test
        if self.test:
            similarities_a = F.cosine_similarity(layer_feat_1[:, :, :], self.memory_bank, dim=2)

        else:
            # mapreduce here when training
            batch = layer_feat_1.shape[0]
            # print('device:', layer_feat_1.get_device())
            # print(layer_feat_1.get_device()== 0)
            if layer_feat_1.get_device() == 8:
                self.memory_bank = self.memory_bank.to('cuda:0')
                similarities_a1 = F.cosine_similarity(layer_feat_1[0:1, :, :], self.memory_bank, dim=2)
                similarities_a2 = F.cosine_similarity(layer_feat_1[1:2, :, :], self.memory_bank, dim=2)
                similarities_a3 = F.cosine_similarity(layer_feat_1[2:3, :, :], self.memory_bank, dim=2)
                similarities_a4 = F.cosine_similarity(layer_feat_1[3:4, :, :], self.memory_bank, dim=2)
                # print(similarities_a1.shape) [4,320]
                similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3, similarities_a4), dim=0)
                # similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3), dim=0)
            if layer_feat_1.get_device() == 9:
                self.memory_bank = self.memory_bank.to('cuda:1')
                similarities_a1 = F.cosine_similarity(layer_feat_1[0:1, :, :], self.memory_bank, dim=2)
                similarities_a2 = F.cosine_similarity(layer_feat_1[1:2, :, :], self.memory_bank, dim=2)
                similarities_a3 = F.cosine_similarity(layer_feat_1[2:3, :, :], self.memory_bank, dim=2)
                similarities_a4 = F.cosine_similarity(layer_feat_1[3:4, :, :], self.memory_bank, dim=2)
                # print(similarities_a1.shape) [4,320]
                # similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3), dim=0)
                similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3, similarities_a4), dim=0)
            #print(layer_feat_1.shape, self.memory_bank.shape)
            similarities_a1 = F.cosine_similarity(layer_feat_1[0:batch // 4, :, :], self.memory_bank, dim=2)
            similarities_a2 = F.cosine_similarity(layer_feat_1[batch // 4:batch // 2, :, :], self.memory_bank, dim=2)
            similarities_a3 = F.cosine_similarity(layer_feat_1[batch // 2:batch * 3 // 4, :, :], self.memory_bank,
                                                  dim=2)
            similarities_a4 = F.cosine_similarity(layer_feat_1[batch * 3 // 4:batch, :, :], self.memory_bank, dim=2)
            similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3, similarities_a4), dim=0)
        # 根据余弦相似度降序排列，选择前3个最接近的向量的索引
        _, indices_a = torch.topk(similarities_a, k=self.k, dim=1, largest=True)

        # 根据索引从Memory Bank中获取对应的向量
        # print(indices_a.get_device())

        closest_vectors_a = self.memory_bank[indices_a]  # [8, k, 802816]

        return layer_feat_a, closest_vectors_a  # [8, 802816] [8, k, 802816]

    def forward(self, img, mask_ratio=0.0):
        layer_feat_a, closest_vectors_a = self.search_bank(img, mask_ratio)
        m = layer_feat_a.shape[0]  # 8
        k = self.k
        closest_vectors_a = closest_vectors_a.reshape(m, 8, k * 196, 512)  # feature from memory bank
        layer_feat_a = layer_feat_a.reshape(m, 8, 196, 512)  # feature from input
        # layer_feat_a = layer_feat_a+ torch.full_like(layer_feat_a,1)
        # 做key value计算时 也将异常图片加进去
        # layer_feat_a = layer_feat_a + torch.full_like(layer_feat_a,1)
        closest_vectors_a = torch.cat((closest_vectors_a, layer_feat_a), dim=2)  # [m, 8, (k+1)*196, 512]
        # closest_vectors_a = self.generate_relative_absolute_positional_encoding(closest_vectors_a.cuda(), layer_feat_a.cuda())
        assert closest_vectors_a.shape[2] == (k + 1) * 196
        # 计算 normal-abnormal-attention
        # query_a = self.linear_query(layer_feat_a)
        key_a = self.linear_key(closest_vectors_a)  # generate key according to closest_vectors_a
        value_a = self.linear_value(closest_vectors_a)  # generate value according to closest_vectors_a

        score1 = layer_feat_a @ key_a.permute(0, 1, 3, 2)
        # score1 = query_a @ key_a.permute(0, 1, 3, 2)
        score1 = self.softmax(score1) * 512 ** (-0.5)

        out = score1 @ value_a  # output of first layer, which calculate normal-abnormal attention instead of self-attention

        # 将互补的特征图乘0.5后相加  考虑直接相加？
        out = out.permute(0, 2, 1, 3).reshape(-1, 196, 512 * 8)  # 这里的reshape相当于 卷积里面的concat [n, 196, 8*512]
        out = self.reduce_dim(out)  # [n, 196, 768] [n, 196, 192]

        # early cnn
        # out = out.reshape(-1, 14, 14, 768).permute(0, 3, 1, 2)  # [8, 768, 14, 14]
        # out = self.conv1(out)  # [8, 768, 14, 14]
        # out = out.reshape(-1, 768, 14 * 14).permute(0, 2, 1)
        # ====ViT重构===
        out = self.vit(out)  # [n, 197, 768] [n, 197, 192]

        # 最终定稿 不直接reduce dim了  最后用3通道图变1通道图来修shape
        out = self.unpatchify_256(out[:, 1:, : ]) # [n, 3, 224, 224]
        out = self.conv2(self.conv1(out))
        # 加bn 和 relu
        # out = self.conv1(out)
        # out = self.bn(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # 三个卷积
        # out = self.conv3(self.conv2(self.conv1(out)))
        # ===抽取shape===
        # out = self.reduce_dim2(out[:, 1:, : ]) # [n, 196, 256]

        # out = self.unpatchify(out) # [n, 1, 224, 224]
        # out = self.fin_2(self.fin_1(out))

        # ===重构图片===
        # out = self.unpatchify_256(out[:, 1:, :])  # [n, 3, 224, 224]

        # ===swin重构图片===
        # out = self.swin(out)
        # out = self.unpatchify_256(out)

        # 可重参数化卷积
        # out = self.big_bn1(self.conv_big_1(out))+ self.small_bn1(self.conv_small_1(out))
        # out = self.conv_big_2(out)+ self.conv_small_2(out)

        return out


class ShapeRepair(nn.Module):
    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------------------
        # feature extractor
        self.feature_extractor = MaskedAutoencoderViT()
        weights_dict = torch.load("./mae_visualize_vit_large.pth")
        self.feature_extractor.load_state_dict(weights_dict['model'], strict=False)
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # shape search
        net = ShapeExtractor().to("cpu")
        net.load_state_dict(torch.load("./memory_bank/memory_bank_screw.pckl"))
        self.memory_bank1 = net.state_dict()['memory_bank1'].to("cuda:0")
        self.memory_bank2 = net.state_dict()['memory_bank2'].to("cuda:0")
        for parameter in net.parameters():
            parameter.requires_grad = False
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # shape repair
        self.softmax = nn.Softmax(dim=-1)
        self.linear_key = nn.Linear(512, 512)
        self.linear_value = nn.Linear(512, 512)

        self.reduce_dim = nn.Linear(512*8, 768)

        # 抽取shape时depth是8 重构图片时depth是12
        self.vit = ViT(num_patches=196, dim=768, depth=8, heads=12, mlp_dim=768*4, dropout=0.)

        # self.reduce_dim2 = nn.Linear(768, 256, bias=True)
        ##self.norm = nn.LayerNorm(256)
        # 抽取shape信息
        # self.fin = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
         # 尝试在最后加CNN来smooth修复效果
        # self.fin_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.fin_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        # 最后输出3通道再输出一通道作为shape试试
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        #self.transformer = Transformer(dim=256,depth=1,heads=4,dim_head=64, mlp_dim=256*4)
        # --------------------------------------------------------------------------

        # 直接重构原图片
        # self.fin_img = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 )
        imgs: (N, 3, H, W)
        """
        # print(x.shape)
        # x = cv2.resize(x, (224, 224))
        p = 16 # self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5) # 14
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1)) # [N, 14, 14, 16, 16, 1]
        x = torch.einsum('nhwpqc->nchpwq', x) # [N, 1, 14, 16, 14, 16]
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs # [B, 1, 224, 224]

    def unpatchify_256(self, x):
        """
        x: (N, L, 3*patch_size**2 )
        imgs: (N, 3, H, W)
        """
        # print(x.shape)
        # x = cv2.resize(x, (224, 224))
        p = 16 # self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5) # 14
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3)) # [N, 14, 14, 16, 16, 3]
        x = torch.einsum('nhwpqc->nchpwq', x) # [N, 3, 14, 16, 14, 16]
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs # [B, 3, 224, 224]

    def search_bank(self, img, mask_ratio=0.5):
        img = tt.Resize((224, 224))(img)

        x1, x2, mask1, mask2, ids_restore = self.feature_extractor.forward_encoder(img, mask_ratio)
        x1, x2, multi_feat1, multi_feat2 = self.feature_extractor.forward_decoder(x1, x2, ids_restore)

        feat0 = []
        feat1 = []
        for i, layer_feat in enumerate(zip(multi_feat1, multi_feat2)): # [1, 197, 512]

            layer_feat0 = layer_feat[0][:, 1:, :] # [1, 196, 512]
            layer_feat1 = layer_feat[1][:, 1:, :]  # [1, 196, 512]


            feat0.append(layer_feat0) # 8* [1, 196, 512]
            feat1.append(layer_feat1)
            # layer_feats.append(torch.cat((layer_feat0, layer_feat1), dim=1)) # [1, 2, 224, 224]
        feats0 = torch.cat(feat0, dim=1)  # [1, 8*196, 512 ]
        feats1 = torch.cat(feat1, dim=1)  # [1, 8*196, 512 ]
        layer_feat_a = feats0.reshape(-1, 8 * 196 * 512)  # [1, 8*196*512]
        layer_feat_b = feats1.reshape(-1, 8 * 196 * 512)  # [1, 8*196*512]

        # print("正在查询最相关的三个向量...")
        # print(layer_feat_a.shape) # [8, 802816]

        # 扩展输入向量的形状，使其与 memory bank 的大小匹配
        layer_feat_1 = layer_feat_a.unsqueeze(1)  # [8, 1, 802816]
        layer_feat_2 = layer_feat_b.unsqueeze(1)

        # 计算输入向量与 memory bank 中所有向量之间的余弦相似度
        similarities_a = F.cosine_similarity(layer_feat_1, self.memory_bank1, dim=2)
        similarities_b = F.cosine_similarity(layer_feat_2, self.memory_bank2, dim=2)
        # 根据余弦相似度降序排列，选择前3个最接近的向量的索引
        _, indices_a = torch.topk(similarities_a, k=3, dim=1, largest=True)
        _, indices_b = torch.topk(similarities_b, k=3, dim=1, largest=True)

        # 根据索引从Memory Bank中获取对应的向量
        closest_vectors_a = self.memory_bank1[indices_a] # [8, 3, 802816]
        closest_vectors_b = self.memory_bank2[indices_b]

        return layer_feat_a, layer_feat_b, closest_vectors_a, closest_vectors_b # [8, 802816] [8, 3, 802816]

    def forward(self, img, mask_ratio=0.5):
        layer_feat_a, layer_feat_b, closest_vectors_a, closest_vectors_b = self.search_bank(img.cuda(), mask_ratio)
        m = layer_feat_a.shape[0] # 8
        closest_vectors_a = closest_vectors_a.reshape(m, 8, 3*196, 512) # feature from memory bank
        closest_vectors_b = closest_vectors_b.reshape(m, 8, 3*196, 512) # [n , 8, 3*196, 512]
        layer_feat_a = layer_feat_a.reshape(m, 8, 196, 512) # feature from input
        layer_feat_b = layer_feat_b.reshape(m, 8, 196, 512) # [n, 8, 196, 512]


        # 计算 normal-abnormal-attention
        key_a = self.linear_key(closest_vectors_a) # generate key according to closest_vectors_a
        value_a = self.linear_value(closest_vectors_a) # generate value according to closest_vectors_a
        key_b = self.linear_key(closest_vectors_b)
        value_b = self.linear_value(closest_vectors_b)

        score1 = layer_feat_a @ key_a.permute(0, 1, 3, 2)
        score1 = self.softmax(score1) * 512**(-0.5)
        score2 = layer_feat_b @ key_b.permute(0, 1, 3, 2)
        score2 = self.softmax(score2) * 512**(-0.5)

        out1 = score1 @ value_a # output of first layer, which calculate normal-abnormal attention instead of self-attention
        out2 = score2 @ value_b # [n, 8, 196, 512]

        # 将互补的特征图乘0.5后相加  考虑直接相加？
        out = 0.5*out1 + 0.5*out2 # [n, 8, 196, 512]
        out = out.permute(0, 2, 1, 3).reshape(-1, 196, 512 * 8) # 这里的reshape相当于 卷积里面的concat [n, 196, 8*512]
        out = self.reduce_dim(out) # [n, 196, 768]

        out = self.vit(out) # [n, 196, 768]
        # 不直接reduce dim了  最后用3通道图变1通道图来修shape
        out = self.unpatchify_256(out[:, 1:, : ]) # [n, 3, 224, 224]
        out = self.conv2(self.conv1(out))
        # ===抽取shape===
        #out = self.reduce_dim2(out[:, 1:, : ]) # [n, 196, 256]

        #out = self.unpatchify(out) # [n, 1, 224, 224]
        #out = self.fin_2(self.fin_1(out))

        # ===重构图片===
        # out = self.unpatchify_256(out[:, 1:, : ]) # [n, 3, 224, 224]
        # out = self.fin_img(out)

        return out


class finetunedmae_ShapeExtractor_0mask_ratio(nn.Module):
    def __init__(self, object_name, extract=False):
        super().__init__()

        self.feature_extractor = FineTuneMae()
        weights_dict = torch.load(
            "/data/CH/ChenHao/projects/DRAEM/image_repair+Draem_model/fine_tune_mae/mvtec_finetuned_bs16_mae_epoch49.pth")
        self.feature_extractor.load_state_dict(weights_dict, strict=True)
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        dict_obj = {'01': 400, '03': 1000, 'candle': 1000, 'capsules': 602, 'cashew': 500, 'chewinggum': 503,
                    'fryum': 500, 'macaroni1': 1000, 'macaroni2': 1000, 'pcb1': 1004, 'pcb2': 1001, 'pcb3': 1006,
                    'pcb4': 1005, 'pipe_fryum': 500, 'cable': 224, 'transistor': 213, 'capsule': 219, 'hazelnut': 391,
                    'screw': 320, 'toothbrush': 60, 'zipper': 240, 'pill': 267, 'metal_nut': 220, 'bottle': 209,
                    'carpet': 280, 'wood': 247, 'tile': 230, 'leather': 245, 'grid': 264}
        # 对于下面 Mvtec是0.3
        if object_name in ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1',
                           'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']:
            self.register_buffer("memory_bank_normal", torch.randn(int(dict_obj[object_name] * 0.1), 802816))
        else:
            self.register_buffer("memory_bank_normal", torch.randn(int(dict_obj[object_name] * 0.3), 802816))
        if extract:
            print("抽取shape")
            self.clear_memory()

    def append_to_memory_bank(self, values):
        self.memory_bank_normal = torch.cat((self.memory_bank_normal.detach().cpu(), values.cpu()), dim=0)

    def store_to_memory(self, img, mask_ratio=0):
        img = tt.Resize((224, 224))(img)

        latent = self.feature_extractor.forward_encoder(img, mask_ratio)
        decoder_feature = self.feature_extractor.forward_decoder(latent)

        feat0 = []
        for i, layer_feat in enumerate(zip(decoder_feature)):  # [1, 197, 512
            # print(type(layer_feat), len(layer_feat), layer_feat[0].shape) # [8, 197, 512]
            layer_feat0 = layer_feat[0][:, 1:, :]  # [1, 196, 512]
            feat0.append(layer_feat0)  # 8*[1, 196, 512]

        feats0 = torch.cat(feat0, dim=1)  # [1, 8*196, 512 ]
        layer_feat0 = feats0.reshape(-1, 8 * 196 * 512)  # [1, 8*196*512]

        self.append_to_memory_bank(layer_feat0)

    def clear_memory(self):
        self.memory_bank_normal = Tensor()

    def subsample(self):
        sampler = ApproximateGreedyCoresetSampler(percentage=0.1, device="cuda:0")
        print("shape of memory bank: " + str(self.memory_bank_normal.shape))  # torch.Size([1704, 100352])
        coreset1_idx = sampler._compute_greedy_coreset_indices(self.memory_bank_normal)
        # coreset2_idx = sampler._compute_greedy_coreset_indices(self.memory_bank2)
        self.memory_bank_normal = self.memory_bank_normal[coreset1_idx, :]  # torch.Size([170, 100352])
        # self.memory_bank2 = self.memory_bank2[coreset2_idx, :]

    def forward(self, img):
        self.subsample()

class ShapeExtractor_0mask_ratio(nn.Module):
    def __init__(self, object_name, extract=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # feature extractor
        self.feature_extractor = MaskedAutoencoderViT()
        weights_dict = torch.load("./mae_visualize_vit_large.pth")
        self.feature_extractor.load_state_dict(weights_dict['model'], strict=False)

        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        dict_obj = {'01':400, '03':1000,'candle':1000,'capsules':602,'cashew':500,'chewinggum':503,'fryum':500, 'macaroni1':1000,'macaroni2':1000,'pcb1':1004, 'pcb2':1001, 'pcb3':1006 , 'pcb4':1005 , 'pipe_fryum': 500,  'cable':224, 'transistor':213, 'capsule':219, 'hazelnut':391, 'screw':320,'toothbrush':60,'zipper':240, 'pill':267, 'metal_nut':220, 'bottle':209, 'carpet':280, 'wood':247, 'tile':230, 'leather':245, 'grid':264}
        # 对于下面 Mvtec是0.3
        if object_name in ['candle','capsules', 'cashew','chewinggum','fryum', 'macaroni1','macaroni2','pcb1', 'pcb2', 'pcb3' , 'pcb4' , 'pipe_fryum']:
            self.register_buffer("memory_bank_normal", torch.randn(int(dict_obj[object_name]*0.1), 802816))
        else:
            self.register_buffer("memory_bank_normal", torch.randn(int(dict_obj[object_name] * 0.3), 802816))
        if extract:
            print("抽取shape")
            self.clear_memory()

    def append_to_memory_bank(self, values):
        self.memory_bank_normal = torch.cat((self.memory_bank_normal.detach().cpu(), values.cpu()), dim=0)
        


    def store_to_memory(self, img, mask_ratio=0):
        img = tt.Resize((224, 224))(img)

        x1, x2, mask1, mask2, ids_restore = self.feature_extractor.forward_encoder(img, mask_ratio)
        x1, _, multi_feat1, _ = self.feature_extractor.forward_decoder(x1, x2, ids_restore)

        feat0=[]
        for i, layer_feat in enumerate(zip(multi_feat1)): # [1, 197, 512

            layer_feat0 = layer_feat[0][:, 1:, :] # [1, 196, 512]
            feat0.append(layer_feat0) # 8*[1, 196, 512]
            


        feats0 = torch.cat(feat0, dim=1)  # [1, 8*196, 512 ]
        layer_feat0 = feats0.reshape(-1, 8 * 196 * 512)  # [1, 8*196*512]

        self.append_to_memory_bank(layer_feat0)

    def clear_memory(self):
        self.memory_bank_normal = Tensor()

    def subsample(self):
        sampler = ApproximateGreedyCoresetSampler(percentage=0.1, device="cuda:0")
        print("shape of memory bank: "+ str(self.memory_bank_normal.shape)) # torch.Size([1704, 100352])
        coreset1_idx = sampler._compute_greedy_coreset_indices(self.memory_bank_normal)
        #coreset2_idx = sampler._compute_greedy_coreset_indices(self.memory_bank2)
        self.memory_bank_normal = self.memory_bank_normal[coreset1_idx, :] # torch.Size([170, 100352])
        #self.memory_bank2 = self.memory_bank2[coreset2_idx, :]


    def forward(self, img):
        self.subsample()

class ShapeExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------------------
        # feature extractor
        self.feature_extractor = MaskedAutoencoderViT()
        weights_dict = torch.load("./mae_visualize_vit_large.pth")
        self.feature_extractor.load_state_dict(weights_dict['model'], strict=False)
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        # torch.manual_seed(42)
        # self.reduce_dim = nn.Linear(8*196 * 512, 196*512)
        # self.reduce_dim.requires_grad = False

        self.register_buffer("memory_bank1", torch.randn(320, 802816))
        self.register_buffer("memory_bank2", torch.randn(320, 802816))
        # self.clear_memory()

    def append_to_memory_bank(self, values1, values2):
        self.memory_bank1 = torch.cat((self.memory_bank1.detach().cpu(), values1.cpu()), dim=0)
        self.memory_bank2 = torch.cat((self.memory_bank2.detach().cpu(), values2.cpu()), dim=0)

  
    def store_to_memory(self, img, mask_ratio=0.5):
        img = tt.Resize((224, 224))(img)

        x1, x2, mask1, mask2, ids_restore = self.feature_extractor.forward_encoder(img, mask_ratio)
        x1, x2, multi_feat1, multi_feat2 = self.feature_extractor.forward_decoder(x1, x2, ids_restore)

        feat0=[]
        feat1=[]
        for i, layer_feat in enumerate(zip(multi_feat1, multi_feat2)): # [1, 197, 512]
            #print(layer_feat.shape)

            layer_feat0 = layer_feat[0][:, 1:, :] # [1, 196, 512]
            layer_feat1 = layer_feat[1][:, 1:, :]  # [1, 196, 512]

            feat0.append(layer_feat0) # 8*[1, 196, 512]
            feat1.append(layer_feat1)


        feats0 = torch.cat(feat0, dim=1)  # [1, 8*196, 512 ]
        feats1 = torch.cat(feat1, dim=1)  # [1, 8*196, 512 ]
        layer_feat0 = feats0.reshape(-1, 8 * 196 * 512)  # [1, 8*196*512]
        layer_feat1 = feats1.reshape(-1, 8 * 196 * 512)  # [1, 8*196*512]
        # layer_feat0 = self.reduce_dim(layer_feat0)
        # layer_feat1 = self.reduce_dim(layer_feat1)# [1, 196*512]

        self.append_to_memory_bank(layer_feat0, layer_feat1)
    def clear_memory(self):
        self.memory_bank1 = Tensor()
        self.memory_bank2 = Tensor()
    def subsample(self):
        sampler = ApproximateGreedyCoresetSampler(percentage=0.1, device="cuda:0")
        print("shape of memory bank: "+ str(self.memory_bank1.shape)) # torch.Size([1704, 100352])
        coreset1_idx = sampler._compute_greedy_coreset_indices(self.memory_bank1)
        coreset2_idx = sampler._compute_greedy_coreset_indices(self.memory_bank2)
        self.memory_bank1 = self.memory_bank1[coreset1_idx, :] # torch.Size([170, 100352])
        self.memory_bank2 = self.memory_bank2[coreset2_idx, :]


    def forward(self, img):
        self.subsample()


import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, ratio=4.0, dropout=0.):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, int(in_dim*ratio))
        self.linear2 = nn.Linear(int(in_dim*ratio), in_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.linear1(x)))
        x = self.dropout(self.act(self.linear2(x)))

        return x



class Cross_Attention(nn.Module):
    def __init__(self, patch_num, embed_dim, num_heads, qkv_bias= False, dropout=0., atten_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
        self.scales = self.head_dim**(-0.5)
        self.patch_num = patch_num

        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.kv = nn.Linear(embed_dim, embed_dim*2)

        self.proj1 = nn.Linear(embed_dim*2, embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)

        # self.cls_token = nn.Parameter(torch.randn(1, 1,embed_dim))
        self.position_embed = nn.Parameter(torch.zeros(1, patch_num, embed_dim))

    def forward(self, x1, x2):
        B, N, C = x1.shape # [n, 196, 768]
        _, N2, _ = x2.shape# # [n, k*196, 768]
        x1 = x1 + self.position_embed

        position_embed = torch.cat([self.position_embed] * int(N2/N), dim=1)
        #print(position_embed.shape)
        #print(x2.shape)
        x2 = x2 + position_embed

        qkv = self.qkv(x1).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # [B, num_heads, N, C//self.num_heads]
        k_2v_2 = self.kv(x2).reshape(B, N2, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = k_2v_2.unbind(0) # [B, num_heads, N2, C//self.num_heads]


        # self_attn
        attn = (q @ k.transpose(-1, -2)) * self.scales # [B, num_heads, N, N]
        attn = self.softmax(attn) # [B, num_heads, N, N]
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)


        # cross_attn
        attn2 = (q @ k2.transpose(-1, -2)) * self.scales  # [B, num_heads, N, N]
        attn2 = self.softmax(attn2)  # [B, num_heads, N, N]
        out2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C)

        out = torch.cat((out, out2), dim=-1) # (B, N, 2C)
        out = self.proj2(self.proj1(out))

        return out

class CrossBlock(nn.Module):

    def __init__(self, patch_num=196, embed_dim=768, num_heads=12, qkv_bias=True, mlp_ratio=4.0, act = nn.GELU, norm = nn.LayerNorm, dropout=0., atten_drop=0.):
        super().__init__()
        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.attn = Cross_Attention(patch_num, embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout=dropout, atten_drop=atten_drop)

        # self.norm3 = norm(embed_dim)
        # self.mlp = MLP(in_dim=embed_dim, ratio=mlp_ratio, dropout=dropout)

    def forward(self, x1, x2):
        x = x1 + self.attn(self.norm1(x1), self.norm2(x2))
        # x = x1 + self.mlp(self.norm3(x1))
        return x

class ShapeRepair_0mask_ratio_new(nn.Module):
    def __init__(self, object_name, depth=8, k=7, test=False):
        super().__init__()
        self.k = k
        self.test = test
        # --------------------------------------------------------------------------
        # shape search
        net = ShapeExtractor_0mask_ratio(object_name).to("cpu")
        if object_name in ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1',
                           'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']:
            net.load_state_dict(torch.load("./memory_bank/visa_memory_bank_" + object_name + "_0mask_ratio_epoch1.pckl",
                                           map_location='cpu'))
        else:
            net.load_state_dict(
                torch.load("./memory_bank/mvtec_memory_bank_" + object_name + "_0mask_ratio_epoch3.pckl",
                           map_location='cpu'))
        self.memory_bank = net.state_dict()['memory_bank_normal'].cuda()
        for parameter in net.parameters():
            parameter.requires_grad = False
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # feature extractor
        self.feature_extractor = MaskedAutoencoderViT()
        # weights_dict = torch.load("./mae_visualize_vit_large.pth")

        weights_dict = torch.load("./mae_visualize_vit_large.pth", map_location='cpu')
        self.feature_extractor.load_state_dict(weights_dict['model'], strict=False)
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # shape repair
        self.softmax = nn.Softmax(dim=-1)

        self.up_reduce_dim = nn.Linear(512*8, 768)
        self.down_reduce_dim = nn.Linear(512*8, 768)

        self.blocks = nn.ModuleList([
            CrossBlock() for i in range(depth)
        ])
        print('depth:' + str(depth))
        print('k:' + str(k))

        # 最后输出3通道再输出一通道作为shape试试
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)



    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 )
        imgs: (N, 3, H, W)
        """
        # print(x.shape)
        # x = cv2.resize(x, (224, 224))
        p = 16  # self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)  # 14
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))  # [N, 14, 14, 16, 16, 1]
        x = torch.einsum('nhwpqc->nchpwq', x)  # [N, 1, 14, 16, 14, 16]
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs  # [B, 1, 224, 224]

    def unpatchify_256(self, x):
        """
        x: (N, L, 3*patch_size**2 )
        imgs: (N, 3, H, W)
        """
        # print(x.shape)
        # x = cv2.resize(x, (224, 224))
        p = 16  # self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)  # 14
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))  # [N, 14, 14, 16, 16, 3]
        x = torch.einsum('nhwpqc->nchpwq', x)  # [N, 3, 14, 16, 14, 16]
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs  # [B, 3, 224, 224]

    def search_bank(self, img, mask_ratio=0.0):
        img = tt.Resize((224, 224))(img)

        x1, x2, mask1, mask2, ids_restore = self.feature_extractor.forward_encoder(img, mask_ratio)
        x1, _, multi_feat1, _ = self.feature_extractor.forward_decoder(x1, x2, ids_restore)

        feat0 = []
        for i, layer_feat in enumerate(zip(multi_feat1)):  # [1, 197, 512]

            layer_feat0 = layer_feat[0][:, 1:, :]  # [1, 196, 512]

            feat0.append(layer_feat0)  # 8* [1, 196, 512]
        feats0 = torch.cat(feat0, dim=1)  # [1, 8*196, 512 ]
        layer_feat_a = feats0.reshape(-1, 8 * 196 * 512)  # [B, 8*196*512]

        # 扩展输入向量的形状，使其与 memory bank 的大小匹配
        layer_feat_1 = layer_feat_a.unsqueeze(1)  # [8, 1, 802816]

        # 计算输入向量与 memory bank 中所有向量之间的余弦相似度
        # similarities_a = F.cosine_similarity(layer_feat_1, self.memory_bank, dim=2)
        # test
        if self.test:
            similarities_a = F.cosine_similarity(layer_feat_1[:, :, :], self.memory_bank, dim=2)

        else:
            # mapreduce here when training
            batch = layer_feat_1.shape[0]
            similarities_a1 = F.cosine_similarity(layer_feat_1[0:batch // 4, :, :], self.memory_bank, dim=2)
            similarities_a2 = F.cosine_similarity(layer_feat_1[batch // 4:batch // 2, :, :], self.memory_bank, dim=2)
            similarities_a3 = F.cosine_similarity(layer_feat_1[batch // 2:batch * 3 // 4, :, :], self.memory_bank,
                                                  dim=2)
            similarities_a4 = F.cosine_similarity(layer_feat_1[batch * 3 // 4:batch, :, :], self.memory_bank, dim=2)
            similarities_a = torch.cat((similarities_a1, similarities_a2, similarities_a3, similarities_a4), dim=0)
        # 根据余弦相似度降序排列，选择前3个最接近的向量的索引
        _, indices_a = torch.topk(similarities_a, k=self.k, dim=1, largest=True)


        closest_vectors_a = self.memory_bank[indices_a]  # [8, k, 802816]

        return layer_feat_a, closest_vectors_a  # [8, 802816] [8, k, 802816]

    def forward(self, img, mask_ratio=0.0):
        layer_feat_a, closest_vectors_a = self.search_bank(img, mask_ratio)
        m = layer_feat_a.shape[0]  # 8
        k = self.k
        closest_vectors_a = closest_vectors_a.reshape(m, 8, k * 196, 512)  # feature from memory bank
        layer_feat_a = layer_feat_a.reshape(m, 8, 196, 512).permute(0, 2, 1, 3).flatten(2)  # [n, 196, 8*512]
        #print("layer_feat_a", layer_feat_a.shape)
        input_token = self.up_reduce_dim(layer_feat_a) # [n, 196, 768]

        closest_vectors_a = closest_vectors_a.permute(0, 2, 1, 3).flatten(2)  # [n, k*196, 8*512]

        memo_token = self.down_reduce_dim(closest_vectors_a) # [n, k*196, 768]

        assert closest_vectors_a.shape[1] == k * 196

        for blk in self.blocks:
            input_token = blk(input_token, memo_token) # [n, 196, 768]

        # 最终定稿 不直接reduce dim了  最后用3通道图变1通道图来修shape
        out = self.unpatchify_256(input_token[:, :, : ]) # [n, 3, 224, 224]
        out = self.conv2(self.conv1(out))

        return out


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):  # mask_ration = 0.5
        # 生成两份互补的掩码
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is removed
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep1 = ids_shuffle[:, :len_keep]
        ids_keep2 = ids_shuffle[:, len_keep:]
        x_masked1 = torch.gather(x, dim=1, index=ids_keep1.unsqueeze(-1).repeat(1, 1, D))
        x_masked2 = torch.gather(x, dim=1, index=ids_keep2.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask1 = torch.ones([N, L], device=x.device)
        mask1[:, :len_keep] = 0
        mask2 = torch.ones([N, L], device=x.device)
        mask2[:, len_keep:] = 0
        # unshuffle to get the binary mask
        mask1 = torch.gather(mask1, dim=1, index=ids_restore)
        mask2 = torch.gather(mask2, dim=1, index=ids_restore)


        return x_masked1, x_masked2, mask1, mask2, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x1, x2, mask1, mask2, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x1.shape[0], -1, -1)

        x1 = torch.cat((cls_tokens, x1), dim=1)
        x2 = torch.cat((cls_tokens, x2), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x1 = blk(x1)
            x2 = blk(x2)
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        return x1, x2, mask1, mask2, ids_restore

    def forward_decoder(self, x1, x2, ids_restore):
        # embed tokens
        x1 = self.decoder_embed(x1)
        x2 = self.decoder_embed(x2)

        # append mask tokens to sequence
        mask_tokens1 = self.mask_token.repeat(x1.shape[0], ids_restore.shape[1] + 1 - x1.shape[1], 1)
        x_1 = torch.cat([x1[:, 1:, :], mask_tokens1], dim=1)  # no cls token
        x_1 = torch.gather(x_1, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x1.shape[2]))  # unshuffle
        x1 = torch.cat([x1[:, :1, :], x_1], dim=1)  # append cls token

        mask_tokens2 = self.mask_token.repeat(x2.shape[0], ids_restore.shape[1] + 1 - x2.shape[1], 1)
        x_2 = torch.cat([x2[:, 1:, :], mask_tokens2], dim=1)  # no cls token
        x_2 = torch.gather(x_2, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x2.shape[2]))  # unshuffle
        x2 = torch.cat([x2[:, :1, :], x_2], dim=1)  # append cls token

        # add pos embed
        x1 = x1 + self.decoder_pos_embed
        x2 = x2 + self.decoder_pos_embed

        decoder_feat1 = []
        decoder_feat2 = []

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x1 = blk(x1)
            decoder_feat1.append(self.decoder_norm(x1))
            x2 = blk(x2)
            decoder_feat2.append(self.decoder_norm(x2))
        x1 = self.decoder_norm(x1)
        x2 = self.decoder_norm(x2)
        # predictor projection
        x1 = self.decoder_pred(x1)
        x2 = self.decoder_pred(x2)
        # remove cls token
        x1 = x1[:, 1:, :]
        x2 = x2[:, 1:, :]
        return x1, x2, decoder_feat1, decoder_feat2
    # def random_masking(self, x, mask_ratio):
    #     """
    #     Perform per-sample random masking by per-sample shuffling.
    #     Per-sample shuffling is done by argsort random noise.
    #     x: [N, L, D], sequence
    #     """
    #     N, L, D = x.shape  # batch, length, dim
    #     len_keep = int(L * (1 - mask_ratio))
    #
    #     noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    #
    #     # sort noise for each sample
    #     ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)
    #
    #     # keep the first subset
    #     ids_keep = ids_shuffle[:, :len_keep]
    #     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    #
    #     # generate the binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask, dim=1, index=ids_restore)
    #
    #     return x_masked, mask, ids_restore
    # #
    # def forward_encoder(self, x, mask_ratio):
    #     # embed patches
    #     x = self.patch_embed(x)
    #
    #     # add pos embed w/o cls token
    #     x = x + self.pos_embed[:, 1:, :]
    #
    #     # masking: length -> length * mask_ratio
    #     x, mask, ids_restore = self.random_masking(x, mask_ratio)
    #
    #     # append cls token
    #     cls_token = self.cls_token + self.pos_embed[:, :1, :]
    #     cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #
    #     # apply Transformer blocks
    #     for blk in self.blocks:
    #         x = blk(x)
    #
    #     x = self.norm(x)
    #
    #     return x, mask, ids_restore
    #
    # def forward_decoder(self, x, ids_restore):
    #     # embed tokens
    #     x = self.decoder_embed(x)
    #
    #     # append mask tokens to sequence
    #     mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    #     x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    #     x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    #     x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    #
    #     # add pos embed
    #     x = x + self.decoder_pos_embed
    #
    #     decoder_feat = []
    #     # apply Transformer blocks
    #     for blk in self.decoder_blocks:
    #         x = blk(x)
    #         decoder_feat.append(self.decoder_norm(x))
    #     x = self.decoder_norm(x)
    #
    #     # predictor projection
    #     x = self.decoder_pred(x)
    #
    #     # remove cls token
    #     x = x[:, 1:, :]
    #
    #     return x, decoder_feat

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.0):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred, _ = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, self.unpatchify(pred), mask

import abc
from typing import Union

import numpy as np
import torch
import tqdm


class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)

