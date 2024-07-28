from typing import Optional
import torch
import torch.nn as nn

from .model import LayerNorm, ResidualAttentionBlock



class AtemporalProbe(nn.Module):
    """ATP (Revisiting the “Video” in Video-Language Understanding)"""
    
    def __init__(self, cfg):
        super().__init__()
        
        visual_cfg = cfg.VISUAL_NET
        embed_dim = visual_cfg.dim

        mods = []
        in_dim = embed_dim * visual_cfg.atp_frames
        for d in visual_cfg.aggregation_fc_dim:
            mods.append(nn.Linear(in_dim, d))
            mods.append(nn.ReLU(inplace=True))
            in_dim = d
        mods.append(nn.Linear(in_dim, visual_cfg.atp_frames))
            
        self.mlp = nn.Sequential(*mods)

        if visual_cfg.atp_sampling == "softmax":
            self.softmax = nn.Softmax(1)
        elif visual_cfg.atp_sampling == "gumble_softmax":
            raise NotImplementedError(visual_cfg.atp_sampling)
        else:
            raise NotImplementedError(visual_cfg.atp_sampling)

    def forward(self, x):
        # x: shape=(b, t, c)
        g = x.view(x.size(0), -1)
        g = self.mlp(g)

        if self.training:
            g = self.softmax(g).unsqueeze(2)  # (b, t, 1), attention
            y = (x*g).sum(1)  # (b, c)
        else:
            # hard selection
            idx = g.argmax(1, keepdim=True) # (b, 1)
            #print("1", idx.shape)
            idx = idx.unsqueeze(-1).expand((-1, 1, x.shape[-1])) # (b, 1, c)
            #print("2", idx.shape)
            y = x.gather(1, idx).squeeze(1) # (b, c)
            #print("3", y.shape)

        return y





class Aggregator(nn.Module):
    def __init__(self, cfg, mode=None, layers=None):
        super().__init__()
        visual_cfg = cfg.VISUAL_NET
        text_cfg = cfg.TEXT_NET

        self.mode = mode or visual_cfg.aggregation
        self.aggr_reduce = visual_cfg.aggregation_reduce
        assert self.mode in ["meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls"]

        if self.mode in ["LSTM", "Transf", "Transf_cls", "Conv_1D"]:
            embed_dim = visual_cfg.dim
            context_length = text_cfg.CONTEXT_LENGTH
            vocab_size = text_cfg.CONTEXT_LENGTH
            transformer_width = text_cfg.WIDTH
            transformer_heads = transformer_width // 64
            transformer_layers = layers or text_cfg.LAYERS

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)

            if self.mode == "Transf" :
                self.transformer = TemporalTransformer(width=embed_dim, layers=visual_cfg.aggregation_layers, heads=transformer_heads)
                print(f'layer={visual_cfg.aggregation_layers}')

            if self.mode == "LSTM":
                self.lstm_visual = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim,
                                        batch_first=True, bidirectional=False, num_layers=1)

            self.apply(self.init_weights)

            if self.mode == "Transf_cls":
                self.transformer = TAggregate(clip_length=visual_cfg.T, embed_dim=embed_dim, n_layers=6)

            if self.mode == 'Conv_1D' :
                self.shift = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False)
                weight = torch.zeros(embed_dim, 1, 3)
                weight[:embed_dim // 4, 0, 0] = 1.0
                weight[embed_dim // 4:embed_dim // 4 + embed_dim // 2, 0, 1] = 1.0
                weight[-embed_dim // 4:, 0, 2] = 1.0
                self.shift.weight = nn.parameter.Parameter(weight)

        else:
            # meanP
            self.apply(self.init_weights)



    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        b, t, c = x.size()
        x = x.contiguous()
        if self.mode == "meanP":
            pass
        elif self.mode == 'Conv_1D':
            x_original = x
            x = x.view(-1, c, t)
            x = self.shift(x.float())
            x = x.permute(0, 2, 1)
            x = x.type(x_original.dtype) + x_original

        elif self.mode == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original

        elif self.mode == "LSTM":
            x_original = x
            x, _ = self.lstm_visual(x.float())
            self.lstm_visual.flatten_parameters()
            x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
            x = x.type(x_original.dtype) + x_original
        elif self.mode == "Transf_cls":
            x_original = x
            return self.transformer(x).type(x_original.dtype)

        else:
            raise NotImplementedError('Unknown optimizer: {}'.format(self.mode))
        
#        print("self.aggr_reduce called")
#        print(x.mean(dim=1,keepdim=False).shape)
#        print(x.squeeze(0).shape)
        if self.aggr_reduce:
            return x.mean(dim=1, keepdim=False)   # (bz=1, dim)
        else:
            return x.squeeze(0)   # (n_frame, dim)


class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))



def trunc_normal_(x, mean=0., std=1.):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6):
        super(TAggregate, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
            embed_dim))

        self.cls_token = nn.parameter.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.parameter.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        nvids = x.shape[0]

        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)

        return o[0]
