from collections import OrderedDict
from math import ceil, floor
from typing import List, Optional, cast
import pdb
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.distributed as dist
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
__all__ = [
    "VisionTransformer",
    "TextTransformer",
    "GatherLayer",
    "ModifiedResNet"
]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

        self.init_params()

    def init_params(self):
        std = self.attnpool.c_proj.in_features ** -0.5
        nn.init.normal_(self.attnpool.q_proj.weight, std=std)
        nn.init.normal_(self.attnpool.k_proj.weight, std=std)
        nn.init.normal_(self.attnpool.v_proj.weight, std=std)
        nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: Optional[torch.Tensor]=None):
        super().__init__()

        self.n_head = n_head
        self.i = 0
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.static_attn_mask = attn_mask

    def attention(self, x: torch.Tensor, runtime_mask: Optional[torch.Tensor]=None):
        if self.static_attn_mask is None and runtime_mask is None:
            attn_mask = None
        elif self.static_attn_mask is None:
            attn_mask = runtime_mask
        elif runtime_mask is None:
            attn_mask = self.static_attn_mask
        else:
            attn_mask = runtime_mask + self.static_attn_mask

        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
        #   pdb.set_trace()
        return self.attn(x, x, x, need_weights=True, attn_mask=attn_mask)

        # q=k=v, L=S=7*7+1, N=bz, E=768


    def forward(self, x: torch.Tensor, runtime_mask: Optional[torch.Tensor]=None):  
        x = x + self.attention(self.ln_1(x), runtime_mask)[0]
        x = x + self.mlp(self.ln_2(x))
        self.i += 1
        if self.i > 10:
            attn_mask = self.attention(self.ln_1(x), runtime_mask)[1]
            ratio = 0.5
            for i in range(8):
                img = Image.open(f"/home/sandwich/myliu/Attribute_clip/vis_result/{i}.jpg", mode='r')
                img_h, img_w = img.size[0], img.size[1]
                plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

                # scale the image
                img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
                img = img.resize((img_h, img_w))
                plt.imshow(img, alpha=1)
                plt.axis('off')
                
                # normalize the attention mask
                mask = cv2.resize(attn_mask[i].cpu().detach().numpy(), (img_h, img_w))
                mask[:, 0] = 0
                normed_mask = mask / mask.max()
                normed_mask = (normed_mask * 255).astype('uint8')
                plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap="jet")
                plt.savefig(f"/home/sandwich/myliu/Attribute_clip/vis_result/{i}_attn.jpg")
            # pdb.set_trace()
        return x



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, 
                attn_mask: torch.Tensor = None, mask_at_layer: int=-1):
        super().__init__()
        self.width = width
        self.layers = layers

        self.mask_at_layer = mask_at_layer if mask_at_layer>=0 else mask_at_layer + layers

        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.init_params()


    def init_params(self):

        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5

        for block in self.resblocks:
            block = cast(ResidualAttentionBlock, block)
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)


    def forward(self, x: torch.Tensor, mask: torch.FloatTensor=None):
        if mask is None:
            return self.resblocks(x)
        else:
            for i in range(0, self.mask_at_layer):
                x = self.resblocks[i](x)

            x = self.resblocks[self.mask_at_layer](x, mask)

            for i in range(self.mask_at_layer+1, self.layers):
                x = self.resblocks[i](x)
            
            return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, 
                    width: int, layers: int, heads: int, 
                    output_dim: int, extra_token: int=1,
                    mask_at_layer: int=-1):
        super().__init__()
        self.extra_token = extra_token
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.n_heads = heads
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.class_embedding = Parameter(scale * torch.randn(extra_token, width))
        self.positional_embedding = Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + extra_token, width))
        
        # pdb.set_trace()
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, mask_at_layer=mask_at_layer)

        self.ln_post = LayerNorm(width)
        self.proj = Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, bboxes: Optional[torch.Tensor]=None):
        """x: (bz, 3, resolution, resolution)
        bbox: (bz, 4) in 0.0~1.0"""
        
        for i in range(x.size(0)):
            image_new = x[i]
            image_new = image_new.permute(1, 2, 0)
            image_new = image_new.cpu().numpy()
            image_new = np.uint8(255 * (image_new - np.min(image_new)) / (np.max(image_new) - np.min(image_new)))
            image_new = Image.fromarray(image_new)
            image_new.save(f"/home/sandwich/myliu/Attribute_clip/vis_result/{i}.jpg")

        x = self.conv1(x)  # shape = [*, width, grid, grid]   [32, 768, 1, 1]
        grid_size = x.size(-1)
        # pdb.set_trace()
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]                         # [768, 14, 14]
        
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1),    # [768, 1, 768]
             x], dim=1)  # shape = [*, grid ** 2 + extra_token, width]
        # pdb.set_trace()        # x [32,2,768]   self.positional_embedding.to(x.dtype).shape [197,768]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        #pdb.set_trace()
        if bboxes is None:
            # x, attn_mask = self.transformer(x)
            x = self.transformer(x)
        else:
            # TODO: one image with multiple boxes
            mask = self.box_to_mask(bboxes, grid_size)  # (*, grid**2, 1)
            x = self.transformer(x, mask)
        x = x.permute(1, 0, 2)  # LND -> NLD   [64, 197, 768]
        feature_map = x[:, 1:, :]
        b = feature_map.size(0)
        feature_map = feature_map.unsqueeze(0).view(b, 14, 14, 768)
        feature_map = feature_map.permute(0, 3, 1, 2)
        draw_feature_map(feature_map)

        # pdb.set_trace()
        x = self.ln_post(x[:, 0: self.extra_token, :])
        
        if self.proj is not None:
            x = x @ self.proj

        return x


    def box_to_mask(self, bboxes, resolution):
        """Φ is a binary mask converted from the bounding boxes. Φi,j equals −∞ if i is the CLS token and j a patch outside the given bounding boxes, and 0 otherwise."""
        mask = torch.zeros(bboxes.size(0), resolution, resolution)

        for i in range(bboxes.size(0)):
            box = bboxes[i].cpu().numpy()
            x1, y1, x2, y2 = (box*resolution).tolist()
            x1, y1 = floor(x1), floor(y1)
            x2, y2 = ceil(x2), ceil(y2)
            mask[i, y1:y2, x1:x2] = float("-inf")

        # bz, width, width
        mask = mask.reshape(mask.shape[0], -1)
        # bz, width^2

        n_patches = mask.size(1)  # = width*width
        
        mask = torch.cat([
            torch.zeros(mask.size(0), self.extra_token),
            mask,
        ], dim=1)  # bz, width^2+1

        mask = mask.unsqueeze(1).expand(-1, self.extra_token, -1)
        # bz, 1, width^2+1
        
        mask = torch.cat([
            mask, 
            torch.zeros(mask.size(0), n_patches, mask.size(2)),
        ], dim=1) # bz, width^2+1, width^2+1

        mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        # bz, head, width^2+1, width^2+1
        mask = mask.reshape(-1, mask.size(2), mask.size(3))
        # bz*head, width^2+1, width^2+1

        return mask
        




class TextTransformer(nn.Module):
    def __init__(self, embed_dim, vocab_size, context_length, width, heads, layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = Parameter(torch.empty(context_length, width))
        self.ln_final = LayerNorm(width)
        self.projection = Parameter(torch.empty(width, embed_dim))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask(context_length)
        )
        self.init()

    def init(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.projection, std=self.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def build_attention_mask(self, context_length) -> torch.Tensor:
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        # with torch.no_grad():
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.projection

        return x


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


class GatherLayer(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    https://github.com/open-mmlab/OpenSelfSup/blob/696d04950e55d504cf33bc83cfadbb4ece10fbae/openselfsup/models/utils/gather_layer.py
    '''
        
    @staticmethod
    def forward(ctx, input) -> List[torch.Tensor]:
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        with torch.no_grad():
            dist.all_gather(output, input)
        output[dist.get_rank()] = input
        return list(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out



def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,title = 'title', save_dir = 'feature_map',img_name = None):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            # pdb.set_trace()
            img_path = f"/home/sandwich/myliu/Attribute_clip/vis_result/{i}.jpg"
            save_dir = f"/home/sandwich/myliu/Attribute_clip/vis_result_attr"

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            img = mmcv.imread(img_path)
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))
            for idx, heatmap in enumerate(heatmaps):
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap) # 将热力图转换为RGB格式
                #pdb.set_trace()
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # # 将热力图应用于原始图像
                # superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
                superimposed_img = heatmap * 0.5 + img*0.3
                # superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray') # need BGR2RGB
                # plt.imshow(superimposed_img,cmap='jet')
                # plt.imshow(img,cmap='jet')
                # plt.title(title)
                # plt.show()
                save_file_name = os.path.join(save_dir, f'{os.path.basename(img_path).split(".")[0]}_{title}_'+str(idx)+'.png')
                # cv2.imwrite(save_file_name, superimposed_img)  # 将图像保存到硬盘
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(save_file_name, superimposed_img)
            i = i + 1

    else:
        for featuremap in features:
            img_path = '/home/wwyu/dataset/mmocr_det_data/ctw1500/imgs/test/1125.jpg'
            save_dir = '/home/wwyu/dataset/mmocr_det_data/ctw1500/vis_feat'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            img = mmcv.imread(img_path)
            heat_maps=heat_maps.unsqueeze(0)

            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for idx, heatmap in enumerate(heatmaps):
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # 将热力图应用于原始图像
                superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
                # superimposed_img = heatmap * 0.5 + img*0.3
                # superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                # plt.imshow(superimposed_img,cmap='jet')
                plt.title(title)
                plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                cv2.imshow("1",superimposed_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                # i=i+1
