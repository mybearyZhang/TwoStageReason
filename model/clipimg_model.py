# modified from https://github.com/openai/CLIP

from typing import Tuple
from model.clip_utils.visual_prompy import Aggregator, AtemporalProbe
from .clip_utils.tokenize import tokenize
from .clip_utils.model import TextTransformer, GatherLayer, VisionTransformer
import pdb
import torch
from torch import nn
from torchvision import transforms
import numpy as np
from PIL import Image


from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class CLIPimg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        visual_config=cfg.VISUAL_NET
        # text_config=config.text

        ############################# visual encoder ###########################
        
        if visual_config.type == "ViT":
            self.visual_net = VisionTransformer(
                input_resolution=visual_config.resolution,
                patch_size=visual_config.patch_size,
                width=visual_config.width,
                layers=visual_config.layers,
                heads=visual_config.heads,
                output_dim=visual_config.dim,
                extra_token=visual_config.extra_token
            )

        if visual_config.aggregation is not None and visual_config.aggregation != "none":
            if visual_config.aggregation == "atp":
                self.aggregator = AtemporalProbe(cfg)
            else:
                self.aggregator = Aggregator(cfg)
        else:
            self.aggregator = None

        ############################# text encoder #############################
        self.text_net = TextTransformer(
            embed_dim=cfg.TEXT_NET.DIM,
            vocab_size=cfg.TEXT_NET.VOCAB_SIZE,
            context_length=cfg.TEXT_NET.CONTEXT_LENGTH,
            width=cfg.TEXT_NET.WIDTH,
            heads=cfg.TEXT_NET.HEADS,
            layers=cfg.TEXT_NET.LAYERS
        )

        ############################## SVSAP ###################################
        if cfg.VISUAL_NET.svsa_layers:
            self.svsa_model = nn.Sequential(
                Aggregator(cfg, mode="Trasf", layers=cfg.VISUAL_NET.svsa_layers),
                nn.Linear(visual_config.dim, 2)
            )
        else:
            self.svsa_model = self.svsa_head = None


        self.logit_scale = nn.parameter.Parameter(
            torch.log(torch.tensor(1.0/cfg.TEMPERATURE))  )
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="sum")

        self.loss_type = cfg.LOSS_TYPE
        assert self.loss_type in ["local", "global"]


    def __encode_image(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        batchsize = image.size(0)                             # image [1,3,224,224]
        #num_frame = image.size(1)
        #image = image.view(-1, image.size(2), image.size(3), image.size(4))
        # (bz, frame, ch, w, w) -> (bz*frame, ch, w, w)
        # [32, 3, 18, 18]


        
        features = self.visual_net(image) # (bz*frame, dim)    [1,1,512]
        # pdb.set_trace()
        # features = features.view(batchsize, num_frame, -1)
        motion_pred = None
        if self.svsa_model and self.svsa_head:
            motion_pred = self.svsa_model(features)

        if self.aggregator is None:

            # bz = 1, no aggregation
            assert features.size(1) == 1
            features = features.squeeze(1)
        else:
            features = self.aggregator(features)
        
        return features, motion_pred


    def __encode_text(self, text_token):
        return self.text_net(text_token)

    def __encode_normed_image(self, image):
        features, motion_pred = self.__encode_image(image)
        # pdb.set_trace()
        return self.__normalize(features), motion_pred

    def __encode_normed_text(self, text_token):
        return self.__normalize(self.__encode_text(text_token))

    def __normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    def forward(self, image=None, text=None):

        if image is None and text is not None:
            # text(token) -> normed text_feature
            return self.__encode_normed_text(text)
        
        elif self.training:
            # image, token -> logit
            image_features, motion_pred = self.__encode_normed_image(image)
            text_features = self.__encode_normed_text(text)
            # pdb.set_trace()
            assert image_features.size(0) == text_features.size(0)

            if self.loss_type == "global":
                image_features = torch.cat(GatherLayer.apply(image_features), 0)
                text_features  = torch.cat(GatherLayer.apply(text_features), 0)
            # pdb.set_trace()
            if motion_pred is not None:
                return self.__scaled_product(image_features, text_features), motion_pred
            else:
                return self.__scaled_product(image_features, text_features)

        else:
            # image,  text_feature -> logit
            # pdb.set_trace()
            image_feat, _ = self.__encode_normed_image(image)
            
            if text is None:
                return image_feat
            else:
                return self.__scaled_product(image_feat, text)


    # 这个函数实现了CLIP模型中，图像特征和文本特征之间余弦相似度的计算，
    # 在函数中，首先将图像特征和文本特征进行L2归一化，然后计算他们的点积，
    # 最后除以一个因子得到余弦相似度
    def __scaled_product(self, image_features, text_features):
        # cosine similarity as logits
        # pdb.set_trace()
        logit_scale = self.logit_scale.exp()
        # self.logit_scale是CLIP模型中一个用于调整输出结果的参数，它是一个标量值。在CLIP模型中，余弦相似度的计算结果被乘以这个参数，以缩放输出结果
        logits_per_image = logit_scale * image_features @ text_features.t()
        # 它将图像特征向量和文本特征向量进行矩阵乘法运算，得到一个1维的张量，
        # 表示它们之间的点积值。接下来，使用.t()函数对文本特征向量进行转置操作，
        # 将其变成一个列向量。然后将点积值与转置后的文本特征向量进行矩阵乘法运算
        # ，得到一个标量值，表示图像特征向量和文本特征向量之间的余弦相似度得分。
        # 最后，将这个得分乘以logit_scale，得到最终的相似度得分logits_per_image。
        # logits_per_text = logits_per_image.t()
        # shape = [global_batch_size, global_batch_size]

        return logits_per_image #, logits_per_text     这里最后返回的是一个矩阵hhhh


    # def info_nce_loss(self, logits_image, logits_text):
    #     """return loss_image, loss_text"""

    #     batchsize = logits_image.size(0)

    #     ground_truth = torch.arange(batchsize, 
    #         dtype=torch.long, device=logits_image.device)
    #     # size = [bz, bz]

    #     loss_image = self.ce_loss(logits_image, ground_truth)
    #     loss_text  = self.ce_loss(logits_text, ground_truth)

    #     return loss_image, loss_text


    


# def initialize_clip_weight(clip_model, config, strict=True):
#     """load pretrain weight"""
#     if config.initial.weight:
#         state_dict = torch.load(config.initial.weight)
#         if isinstance(state_dict, dict) and 'state_dict' in state_dict:
#             state_dict = state_dict['state_dict']
        
#         clip_pref = "clip_model."
#         state_dict = {
#             k[len(clip_pref):] if k.startswith(clip_pref) else k  :  v
#             for k,v in state_dict.items()
#         }


#         # for key in ["input_resolution", "context_length", "vocab_size"]:
#         #     if key in state_dict:
#         #         del state_dict[key]
        
#         if config.initial.extra_token == 'copy':
#             state_dict['visual_net.class_embedding'] = \
#                 state_dict['visual_net.class_embedding'].expand_as(clip_model.visual_net.class_embedding)
#             state_dict['visual_net.positional_embedding'] = torch.cat([
#                 state_dict['visual_net.positional_embedding'][0].unsqueeze(0).repeat(config.visual.extra_token, 1),
#                 state_dict['visual_net.positional_embedding'][1:],
#             ], dim=0)
#         elif config.initial.extra_token == 'rand':
#             state_dict['visual_net.class_embedding'] = torch.randn(*clip_model.visual_net.class_embedding.shape)
#             state_dict['visual_net.positional_embedding'] = torch.cat([
#                 torch.randn(config.visual.extra_token, state_dict['visual_net.positional_embedding'].shape[1]),
#                 state_dict['visual_net.positional_embedding'][1:],
#             ], dim=0)
#         missing_keys, unexpected_keys = clip_model.load_state_dict(state_dict, strict=False)

#         if strict:
#             assert len(unexpected_keys) == 0, str(unexpected_keys)
#             assert len(missing_keys) == 0, str(missing_keys)
#         else:
#             if len(unexpected_keys) > 0:
#                 print("unexpected_keys", str(unexpected_keys))
#             if len(missing_keys) > 0:
#                 print("missing_keys", str(missing_keys))

#     # freeze backbone
#     for p1 in config.freeze_params:
#         used = False
#         for name, param in clip_model.named_parameters():
#             if p1 in name:
#                 param.requires_grad_(False)
#                 used = True
#         assert used, 'Unrecognized parameter name: %s' % p1

#     # convert_weights(clip_model)



# def convert_weights(model: nn.Module):
#     """Convert applicable model parameters to fp16"""

#     def _convert_weights_to_fp16(l):
#         if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#             l.weight.data = l.weight.data.half()
#             if l.bias is not None:
#                 l.bias.data = l.bias.data.half()

#         if isinstance(l, nn.MultiheadAttention):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

#         for name in ["text_projection", "proj"]:
#             if hasattr(l, name):
#                 attr = getattr(l, name)
#                 if attr is not None:
#                     attr.data = attr.data.half()

#     model.apply(_convert_weights_to_fp16)
