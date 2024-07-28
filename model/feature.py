import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import random
from base import BaseModel
from model.modules import *
from timm.models.layers import to_2tuple

class ResNet18Encoder(BaseModel):
    def __init__(self, num_channels=3, pretrained=False, freeze=False, dropout=0.0, keep_dim=False, state=0, reshape=None):
        super(ResNet18Encoder, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained, progress=False, num_classes=1000)
        
        self.backbone.fc = nn.Identity()
        if num_channels != 3:
            self.backbone.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.keep_dim = keep_dim
        self.state = state
        self.reshape = reshape

        # extra layers
        if state < 0:
            self.layer5 = self._make_layer(512, 512, num_blocks=2, stride=2)
        if state < -1:
            self.layer6 = self._make_layer(512, 512, num_blocks=2, stride=2)
        if state < -2:
            self.layer7 = self._make_layer(512, 512, num_blocks=2, stride=2)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
    def _make_layer(self, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_planes))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)


    def forward(self, x):
        if self.reshape == 'CVR':
            x_size = x.shape
            x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])
        if self.state < 5:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
        if self.state < 4:
            x = self.dropout1(x)
            x = self.backbone.layer1(x)
        if self.state < 3:
            x = self.backbone.layer2(x)
        if self.state < 2:
            x = self.backbone.layer3(x)
        if self.state < 1:
            x = self.backbone.layer4(x)
        if self.state < 0:
            x = self.layer5(x)
        if self.state < -1:
            x = self.layer6(x)
        if self.state < -2:
            x = self.layer7(x)

        x = self.dropout2(x)

        x = self.backbone.avgpool(x)
        if not self.keep_dim:
            x = torch.flatten(x, 1)

        x = self.backbone.fc(x)
        return x
    
class ResNet18MiniEncoder(BaseModel):
    def __init__(self, num_channels=3, pretrained=False, freeze=False, dropout=0.0, keep_dim=False, state=0, reshape=None):
        super(ResNet18MiniEncoder, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained, progress=False, num_classes=1000)
        
        self.backbone.fc = nn.Identity()
        if num_channels != 3:
            self.backbone.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.reshape = reshape

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.reshape == 'CVR':
            x_size = x.shape
            x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        return x
    
class IdentityEncoder(BaseModel):
    def __init__(self, keep_dim=False, reshape=None, avgpool=True):
        super(IdentityEncoder, self).__init__()
        self.keep_dim = keep_dim
        self.reshape = reshape
        self.avgpool = avgpool

    def forward(self, x):
        if self.reshape == 'CVR':
            x_size = x.shape
            x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])
        if self.avgpool:
            x = torch.mean(x, dim=(2,3))
        if not self.keep_dim:
            x = torch.flatten(x, 1)
        return x
    
class ConvEncoder(BaseModel):
    def __init__(self, img_size=224, patch_size=16, num_channels=3, embed_dim=1408, reshape=None):
        super(ConvEncoder, self).__init__()
        # super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.reshape = reshape
        # self.patch_size = patch_size
        # self.num_patches = num_patches

        self.proj = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        if self.reshape == 'CVR':
            x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ViTEncoder(BaseModel):
    def __init__(self, image_size, num_channels, patch_size=8, dim=512, num_classes=512, **kwargs):
        super(ViTEncoder, self).__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_embed = nn.Conv2d(num_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, nhead=8), num_layers=6)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # (batch_size, dim, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, dim)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embed
        x = self.transformer(x)
        x = self.fc(x[:, 0])
        return x

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class RavenFeature(BaseModel):
    def __init__(
        self,
        mlp_dim: int = 8,
        dropout: float = 0,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """
        super(RavenFeature, self).__init__()

        feature_extract = True
        num_classes = 1000
        self.hidden_size = mlp_dim

        self.backbone = models.resnet18(pretrained=False, progress=False, num_classes=num_classes)
        num_channels = 16  # Update with the actual number of channels in your RAVEN input
        self.backbone.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        
        x_size = x.shape
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.dropout1(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.dropout2(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.backbone.fc(x)

        return x


class CvrFeature(BaseModel):
    def __init__(self, dropout = 0, pretrained=False, keep_shape=False, **kwargs):
        super(CvrFeature, self).__init__()
        num_classes = 1000
        self.hidden_size = 8
        self.backbone = models.resnet18(pretrained=pretrained, progress=False, num_classes=num_classes)
        self.backbone.fc = nn.Identity()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.keep_shape = keep_shape

    def forward(self, x):
        
        x_size = x.shape
        x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.dropout1(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        if not self.keep_shape:
            x = self.backbone.layer4(x)

        x = self.dropout2(x)

        x = self.backbone.avgpool(x)
        if not self.keep_shape:
            x = torch.flatten(x, 1)

        x = self.backbone.fc(x)

        return x

class SvrtFeature(BaseModel):
    def __init__(
        self,
        mlp_dim: int = 8,
        dropout: float = 0.0,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """
        super(SvrtFeature, self).__init__()

        feature_extract = True
        num_classes = 8
        use_pretrained = False
        self.hidden_size = mlp_dim

        self.backbone = models.resnet18(weights=None, progress=False, num_classes=num_classes)
        num_channels = 3  # Update with the actual number of channels in your RAVEN input
        self.backbone.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        
        x_size = x.shape
        # x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.dropout1(x)  # Apply dropout after the first convolutional layer

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout2(x)  # Apply dropout before the final linear layer

        x = self.backbone.fc(x)
        return x

class ClevrerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClevrerEncoder, self).__init__()

        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_size*2, hidden_size*3, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_size*3, output_size, kernel_size=3, stride=1, padding=0)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x_1 = self.relu(self.conv1(x))
        x_2 = self.relu(self.conv2(x))
        x_3 = self.relu(self.conv3(x))
        x_4 = self.relu(self.conv4(x))

        return x_4.view(x.size(0), -1), x_1, x_2, x_3, x_4

class BongardHoiFeature(BaseModel):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        downsample = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out

class BongardLogoFeature(BaseModel):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        downsample = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out

class CFEncoder(BaseModel):
    def __init__(self, state_size, z_size, freeze=False, keep_dim=True, state=0, reshape=None):
        super(CFEncoder, self).__init__()
        self.gn_encoder = GCN(in_size=state_size, out_size=z_size, n_layers=0, hidden_size=32)

        if freeze:
            for param in self.gn_encoder.parameters():
                param.requires_grad = False

        self.keep_dim = keep_dim
        self.state = state

    def forward(self, x):
        B, T, K, S = x.shape
        z = self.gn_encoder(x.view(B * T, K, S)).view(B, T, K, -1)
        if not self.keep_dim:
            z = z.view(-1, z.shape[-1])
        return z

class CFEncoder_com(nn.Module):
    def __init__(self,z_size,state_size,cf_size,n_layers,hidden_size,emb_size):
        super(CFEncoder_com,self).__init__()
        self.gn_encoder = GCN(in_size=state_size, out_size=z_size, n_layers=0, hidden_size=32)
        self.gn_ab = GCN(in_size=state_size, out_size=emb_size, n_layers=n_layers, hidden_size=hidden_size,
                         batchnorm=False)
        self.rnn_ab = nn.GRU(emb_size, cf_size, num_layers=2, batch_first=True)
    
    def forward(self,x):
            cf = x[0]
            z = x[1]
            B1, T1, K1, S1 = cf.shape
            embeddings = self.gn_ab(cf.view(B1 * T1, K1, S1)).view(B1, T1, K1, -1)
            embeddings = embeddings.transpose(1, 2).reshape(B1 * K1, T1, -1)
            output, _ = self.rnn_ab(embeddings)
            U = output.view(B1, K1, T1, -1)[:, :, -1]
            z = z.unsqueeze(1)
            B2, T2, K2, S2 = z.shape
            z = self.gn_encoder(z.view(B2 * T2, K2, S2)).view(B2, T2, K2, -1)
            z = z.squeeze(1)
            list_z = [z]
            print("U")
            print(U.shape)
            print("z")
            print(z.shape)
            inpt = torch.cat([list_z[-1], U], dim=-1)
            return inpt


class VQAFeature(BaseModel):
    def __init__(self, image_feature_extractor=models.resnet50(pretrained=True), question_feature_extractor=nn.Linear(77, 1024)):
        super(VQAFeature, self).__init__()
        self.image_feature_extractor = image_feature_extractor
        
        self.image_feature_extractor.fc = nn.Identity()
        self.question_feature_extractor = question_feature_extractor
        self.fc = nn.Linear(3072, 512)
        
    def forward(self, data):
        image = data[0]
        question = data[1]
        image_features = self.image_feature_extractor(image)
        question = question.float()
        question_features = self.question_feature_extractor(question)
        combined_features = torch.cat((image_features, question_features), dim=1)
        output = self.fc(combined_features)
        return output
    
    
class ImgEncoder(nn.Module):

    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[-1].in_features  # input size of feature vector
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1])    # remove last fc layer

        self.model = model                              # loaded model without last fc layer
        self.fc = nn.Linear(in_features, embed_size)    # feature vector of image

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
        img_feature = self.fc(img_feature)                   # [batch_size, embed_size]

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

        return img_feature


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc2 = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature

class VQAencoder(BaseModel):
    def __init__(self, embed_size,qst_vocab_size, word_embed_size, qst_embed_size, num_layers, hidden_size,freeze=False):
        super(VQAencoder, self).__init__()
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[-1].in_features  # input size of feature vector
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1])    # remove last fc layer

        self.model = model                              
        self.fc = nn.Linear(in_features, embed_size)        # 上面是图片的，下面是问题的网络
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc2 = nn.Linear(2*num_layers*hidden_size, qst_embed_size) 
    def forward(self, x):
        """Extract feature vector from image vector.
        """
        image = x[0]
        question = x[1]
        
        with torch.no_grad():
            img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
            # print(2)
        img_feature = self.fc(img_feature)                   # [batch_size, embed_size]
        # print(3)
        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)    
        # print(4)# l2-normalized feature vector,下面是qst的
        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        # print(5)
        # print(qst_vec.shape)
        qst_vec = self.tanh(qst_vec)
        # print(6)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        # print(5)
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        # print(6)
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        # print(7)
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc2(qst_feature)
        # print(8)
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        # print(9)
        combined_feature = self.tanh(combined_feature)
        # print("shape", combined_feature.shape)
        # print("type",combined_feature.type)
        # print(10)
        return combined_feature
            
           

