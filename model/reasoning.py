import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from base import BaseModel
import re
from PIL import Image
from typing import Tuple
# from .clip_utils.tokenize import tokenize
# from .clip_utils.model import TextTransformer, GatherLayer, VisionTransformer
from torchvision import transforms
import numpy as np
from minigpt4.models.mini_gpt4 import MiniGPT4
# from open_flamingo import create_model_and_transforms
# from lavis.models import load_model_and_preprocess

# from huggingface_hub import hf_hub_download


class MLP(BaseModel):
    def __init__(self, input_size=512, hidden_sizes=[128], output_size=8, freeze=False):
        super(MLP, self).__init__()
        layer_sizes = [input_size] + hidden_sizes
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        self.last_layer = nn.Linear(hidden_sizes[-1], output_size)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers) - 1:
                x = nn.ReLU()(x)
        x = self.last_layer(x)
        return x

class ResNet18Reason(BaseModel):
    def __init__(self, num_channels=3, pretrained=False, freeze=False, dropout=0.0, keep_dim=False, state=0, reshape=None):
        super(ResNet18Reason, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained, progress=False, num_classes=1000)
        
        self.backbone.fc = nn.Identity()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.keep_dim = keep_dim
        self.state = state
        self.reshape = reshape

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.state < 4:
            x = self.dropout1(x)
            x = self.backbone.layer1(x)
        if self.state < 3:
            x = self.backbone.layer2(x)
        if self.state < 2:
            x = self.backbone.layer3(x)
        if self.state < 1:
            x = self.backbone.layer4(x)

        x = self.dropout2(x)

        x = self.backbone.avgpool(x)
        if not self.keep_dim:
            x = torch.flatten(x, 1)

        x = self.backbone.fc(x)
        return x
    

class CNN(BaseModel):
    def __init__(self, num_classes=8, pretrained=False, freeze=False):
        super(CNN, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained, progress=False, num_classes=1000).layer4
        self.linear = nn.Linear(512, num_classes)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

class Transformer(BaseModel):
    def __init__(self, input_size, output_size, nhead=4, num_layers=6):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead),
            num_layers=num_layers
        )
        self.decoder = nn.Linear(input_size, output_size)
    
    def forward(self, src):
        output = self.encoder(src)
        output = self.decoder(output)
        return output
    
class Identity(BaseModel):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

# Neuro Symbolic:垃圾GPT，啥都不会
# class NeuroSymbolicNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(NeuroSymbolicNetwork, self).__init()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # 定义符号推理规则
# def symbolic_inference(input_vector):
#     # 这里简单地将输入向量的每个元素乘以2
#     return input_vector * 2


# #BLIP2
# class lavis:
#     def __init__(self, model_name, model_type, device) -> None:
#         model, vis_processors, txt_processors = load_model_and_preprocess(name = model_name, model_type = model_type, is_eval=True, device=device)
#         self.model_name = model_name
#         self.model = model
#         self.vis_processors = vis_processors
#         self.txt_processors = txt_processors
#         self.device = device
#     def generate(self, image, question, name='resize'):
#         if 'opt' in self.model_name:
#             prompt = f'Question: {question} Answer:'
#         elif 't5' in self.model_name:
#             prompt = f'Question: {question} Short answer:'
#         else:
#             prompt = f'Question: {question} Answer:'
#         image = Image.open(image).convert("RGB")
#         # if name == "pad":
#         #     image = pad_image(image, (224,224))
#         # elif name == "resize":
#         #     image = resize_image(image, (224,224))
#         image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
#         prompt = self.txt_processors["eval"](prompt)
#         answer = self.model.predict_answers(samples={"image": image, "text_input": prompt}, inference_method="generate", max_len=48, min_len=1)[0]
#         return answer
    

# class LLM(BaseModel):
# def postprocess_vqa_generation(predictions):
#     return re.split("Question|Answer", predictions, 1)[0]
# class OpenFlamingo:
#     def __init__(self, llama_path, check_point, device) -> None:
#         model, image_processor, tokenizer = create_model_and_transforms(
#             clip_vision_encoder_path="ViT-L-14",
#             clip_vision_encoder_pretrained="openai",
#             lang_encoder_path = llama_path,
#             tokenizer_path = llama_path,
#             cross_attn_every_n_layers=4
#         )
#         checkpoint = torch.load(check_point, map_location="cpu")
#         model.load_state_dict(checkpoint, strict=False)
#         #checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
#         self.model = model.to(device)
#         self.image_processor=image_processor
#         self.tokenizer = tokenizer
#         self.device = device
#     def generate(self, image, question, name="resize"):
#         self.tokenizer.padding_side = "left"
#         lang_x = self.tokenizer(
#         [f"<image>Question:{question} Answer:"],
#         return_tensors="pt",
#         ).to(self.device)
#         len_input =  len(lang_x['input_ids'][0])
#         image = Image.open(image)
#         if name == "resize":
#             image = resize_image(image, (224,224))
#         vision_x = [self.image_processor(image).unsqueeze(0)]
#         vision_x = torch.cat(vision_x, dim=0)
#         vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(self.device)
#         generated_text = self.model.generate(
#         vision_x=vision_x,
#         lang_x=lang_x["input_ids"],
#         attention_mask=lang_x["attention_mask"],
#         max_new_tokens=48,
#         num_beams=3,
#         )
#         answer = self.tokenizer.decode(generated_text[0][len_input:], skip_special_tokens=True)
#         '''process_function = (
#             postprocess_vqa_generation)
#         new_predictions = [
#             process_function(out)
#             for out in self.tokenizer.batch_decode(generated_text, skip_special_tokens=True)
#         ]'''
#         return answer

# GNN
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adjacency_matrix, features):

        aggregate = torch.matmul(adjacency_matrix, features)
        aggregate = aggregate / (adjacency_matrix.sum(dim=1, keepdim=True) + 1e-8)  # 防止除零错误
        output = self.linear(aggregate)      
        return output

class GNN(BaseModel):
    def __init__(self, num_nodes, num_features=512, hidden_dim=128, num_classes=8):
        super(GNN, self).__init__()
        
        self.gc1 = GraphConvLayer(num_features, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, num_classes)
        self.adjacency_matrix = torch.randn(num_nodes, num_nodes)
        
    def forward(self, features):

        hidden = self.gc1(self.adjacency_matrix, features)
        hidden = torch.relu(hidden)  
        output = self.gc2(self.adjacency_matrix, hidden)
        
        return output
    
    def set_device(self, device):
        self.adjacency_matrix = self.adjacency_matrix.to(device)

class GCN_MLP(nn.Module):
    def __init__(self, in_size, h_size, out_size, n_layers, activation=nn.ReLU, batchnorm=False, n_channel=4):
        super(GCN_MLP, self).__init__()

        layer_list = []
        if batchnorm:
            layer_list.append(nn.BatchNorm1d(n_channel))

        layer_list.append(nn.Linear(in_size, h_size))
        layer_list.append(activation())
        for _ in range(n_layers):
            layer_list.append(nn.Linear(h_size, h_size))
            layer_list.append(activation())

        layer_list.append(nn.Linear(h_size, out_size))

        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)


class GCN(BaseModel):
    def __init__(self, input_size, hidden_size, output_size, n_layers=0, activation=nn.ReLU, batchnorm=False):
        super(GCN, self).__init__()

        self.f = GCN_MLP(2 * input_size, hidden_size, hidden_size, n_layers, activation, batchnorm, n_channel=2 * input_size)
        self.g = GCN_MLP(2 * hidden_size + input_size, hidden_size, output_size, n_layers, activation, batchnorm, n_channel=4)

    def forward(self, x):
        x_size = x.shape 
        if (len(x_size) == 2):
            x = x.unsqueeze(1)
        if (len(x_size) == 4):
            B, C, H, W = x_size
            x = x.view(B, -1, C)
        B, K, S = x.shape
        # print(x.shape)

        x1 = x.unsqueeze(1).repeat(1, K, 1, 1)
        # print(x1.shape)
        x2 = x.unsqueeze(2).repeat(1, 1, K, 1)

        x12 = torch.cat([x1, x2], dim=-1)
        interactions = self.f(x12.view(B * K * K, 2 * S)).view(B, K, K, -1)
        E = interactions * (1 - torch.eye(K).view(1, K, K, 1).repeat(B, 1, 1, interactions.shape[-1]).to(x.device))

        E = E.sum(2)
        E_sum = E.sum(1).unsqueeze(1).repeat(1, K, 1)

        out = self.g(torch.cat([x, E, E_sum], dim=-1))
        out = out.view(out.shape[0], -1)

        return out

class EBM(BaseModel):
    def __init__(self, input_size, hidden_size, repeat_size):
        super(EBM, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, input_size)
        self.repeat_size = repeat_size

    def forward(self, data, target):
        # print(type(target))
        # print("0",target.shape)
        target = target.unsqueeze(1).repeat(1, self.repeat_size)
        # print(target)
        # print(target.shape)
        #print("tar",target.shape)
        # print("dat",data.shape)
        x = torch.cat((data, target), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output_vector = self.energy_layer(x)
        return torch.norm(output_vector, dim=1)  
#clip




# class CLIPimg(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
        
#         visual_config=cfg.VISUAL_NET
#         # text_config=config.text

#         ############################# visual encoder ###########################
        
#         if visual_config.type == "ViT":
#             self.visual_net = VisionTransformer(
#                 input_resolution=visual_config.resolution,
#                 patch_size=visual_config.patch_size,
#                 width=visual_config.width,
#                 layers=visual_config.layers,
#                 heads=visual_config.heads,
#                 output_dim=visual_config.dim,
#                 extra_token=visual_config.extra_token
#             )


#         ############################# text encoder #############################
#         self.text_net = TextTransformer(
#             embed_dim=cfg.TEXT_NET.DIM,
#             vocab_size=cfg.TEXT_NET.VOCAB_SIZE,
#             context_length=cfg.TEXT_NET.CONTEXT_LENGTH,
#             width=cfg.TEXT_NET.WIDTH,
#             heads=cfg.TEXT_NET.HEADS,
#             layers=cfg.TEXT_NET.LAYERS
#         )

#         self.logit_scale = nn.parameter.Parameter(
#             torch.log(torch.tensor(1.0/cfg.TEMPERATURE))  )
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.kl_div = nn.KLDivLoss(reduction="sum")

#         self.loss_type = cfg.LOSS_TYPE
#         assert self.loss_type in ["local", "global"]


#     def __encode_image(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
#         batchsize = image.size(0)                             
#         features = self.visual_net(image) 
#         motion_pred = None
#         if self.svsa_model and self.svsa_head:
#             motion_pred = self.svsa_model(features)

#         if self.aggregator is None:

#             # bz = 1, no aggregation
#             assert features.size(1) == 1
#             features = features.squeeze(1)
#         else:
#             features = self.aggregator(features)
        
#         return features, motion_pred


#     def __encode_text(self, text_token):
#         return self.text_net(text_token)

#     def __encode_normed_image(self, image):
#         features, motion_pred = self.__encode_image(image)
#         # pdb.set_trace()
#         return self.__normalize(features), motion_pred

#     def __encode_normed_text(self, text_token):
#         return self.__normalize(self.__encode_text(text_token))

#     def __normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
#         return embeddings / embeddings.norm(dim=-1, keepdim=True)

#     def forward(self, image=None, text=None):

#         if image is None and text is not None:
#             # text(token) -> normed text_feature
#             return self.__encode_normed_text(text)
        
#         elif self.training:
#             # image, token -> logit
#             image_features, motion_pred = self.__encode_normed_image(image)
#             text_features = self.__encode_normed_text(text)
#             # pdb.set_trace()
#             assert image_features.size(0) == text_features.size(0)

#             if self.loss_type == "global":
#                 image_features = torch.cat(GatherLayer.apply(image_features), 0)
#                 text_features  = torch.cat(GatherLayer.apply(text_features), 0)
#             # pdb.set_trace()
#             if motion_pred is not None:
#                 return self.__scaled_product(image_features, text_features), motion_pred
#             else:
#                 return self.__scaled_product(image_features, text_features)

#         else:
#             # image,  text_feature -> logit
#             # pdb.set_trace()
#             image_feat, _ = self.__encode_normed_image(image)
            
#             if text is None:
#                 return image_feat
#             else:
#                 return self.__scaled_product(image_feat, text)

#     def __scaled_product(self, image_features, text_features):
#         # cosine similarity as logits
#         # pdb.set_trace()
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ text_features.t()

#         return logits_per_image 

# class EnergyModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(EnergyModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

#     def energy(self, x):
#         x = self.forward(x)
#         return torch.norm(x, dim=1)  # 使用L2范数作为能量函数

# class EBM(nn.Module):
#     def __init__(self, in_planes, out_planes):
#         super().__init__()
#         self.relu = nn.LeakyReLU(0.1)
#         self.conv1 = nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_planes)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         self.conv3 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_planes)
#         downsample = nn.Sequential(
#             nn.Conv2d(out_planes, out_planes, 1, padding=1, bias=False),
#             nn.BatchNorm2d(out_planes),
#         )
#         self.downsample = downsample
#         self.maxpool = nn.MaxPool2d(2)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         out = self.maxpool(out)

#         return out
#     def energy(self, x):
#         x = self.forward(x)
#         return torch.norm(x, dim=1)  
# class PCNN(torch.nn.Module):
#     def __init__(self, input_size, output_size, threshold=0.5, max_iterations=10):
#         super(PCNN, self).__init__()

#         self.threshold = threshold
#         self.max_iterations = max_iterations
#         self.fc= nn.Linear(input_size, output_size)
#         self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size=3, padding=1)

#     def forward(self, x):
#         # 将输入信号转换为脉冲信号
#         input_layer = (x > self.threshold).float()

#         # 脉冲耦合层
#         pulse_coupling_layer = torch.zeros_like(input_layer)
#         for _ in range(self.max_iterations):
#             neighbor_sum = F.conv2d(pulse_coupling_layer.unsqueeze(0), 
#                                     torch.ones(1, 1, 3, 3).to(x.device), 
#                                     padding=1)
#             # print("in",input_layer.shape)
#             neighbor_sum = neighbor_sum.squeeze()
#             # print("neighbor_sum",neighbor_sum.shape)
#             pulse_coupling_layer += input_layer * (neighbor_sum > 0).float()
            
#         # print("a",pulse_coupling_layer.shape)
#         # 输出层：根据脉冲耦合层的结果进行边缘检测
#         output_layer = self.fc(pulse_coupling_layer)

#         return output_layer

def gumbel_softmax(logits, temperature):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10))
    logits_with_noise = (logits + gumbel_noise) / temperature
    softmax_output = F.softmax(logits_with_noise, dim=-1)
    return softmax_output

class PCNN(BaseModel):
    def __init__(self, input_size, output_size, hidden_size=128, threshold=0, max_iterations=10):
        super(PCNN, self).__init__()

        self.threshold = threshold
        self.max_iterations = max_iterations
        self.fc_pre = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size=3, padding=1)

    def forward(self, x):
        # 将输入信号转换为脉冲信号
        # 计算每一行的均值和标准差
        x = self.fc_pre(x)
        row_means = torch.mean(x, dim=1, keepdim=True)
        row_stds = torch.std(x, dim=1, keepdim=True)

        # 进行Z-score标准化
        normalized_x = (x - row_means) / row_stds

        input_layer = gumbel_softmax(normalized_x, 0.5)

        # 脉冲耦合层
        # pulse_coupling_layer = torch.zeros_like(input_layer)
        pulse_coupling_layer = input_layer
        # for _ in range(self.max_iterations):
        #     neighbor_sum = F.conv2d(pulse_coupling_layer.unsqueeze(0), 
        #                             torch.ones(1, 1, 3, 3).to(x.device), 
        #                             padding=1)
        #     # print("in",input_layer.shape)
        #     neighbor_sum = neighbor_sum.squeeze()
        #     # print("neighbor_sum",neighbor_sum.shape)
        #     pulse_coupling_layer += input_layer * (neighbor_sum > 0).float()
            
            
        # print("a",pulse_coupling_layer.shape)
        # 输出层：根据脉冲耦合层的结果进行边缘检测
        # print(pulse_coupling_layer)
        output_layer = self.fc(pulse_coupling_layer)

        return output_layer
    
class LLM(BaseModel):
    def __init__(self, input_size, output_size, num_channels=3, prompt_instruction=None, img_size=224):
        super(LLM, self).__init__()
        self.model = MiniGPT4(num_channels=num_channels, prompt_instruction=prompt_instruction, img_size=img_size, output_size=output_size)
    
    def forward(self, x, y=None):
        x = self.model(x, y)
        return x