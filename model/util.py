# Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch
import torch
from torch import nn
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

def contrastive_loss(pos_support, pos_query, neg_support, neg_query, temperature):
    """
    Contrastive loss for training the encoder, described in equation 1 in the paper.
    """
    B, S1, _ = pos_support.shape
    B, S2, _ = neg_support.shape
    B, Q1, _ = pos_query.shape
    B, Q2, _ = neg_query.shape

    feats = torch.cat([pos_support, pos_query, neg_support, neg_query], dim=1)
    norm_feats = F.normalize(feats, dim=-1)

    sims = torch.bmm(norm_feats, torch.transpose(norm_feats, 1, 2)) / temperature
    assert sims.shape == (B, S1 + S2 + Q1 + Q2, S1 + S2 + Q1 + Q2)

    n_pos = S1 + Q1
    pos_to_neg = torch.sum(torch.exp(sims[:, :n_pos, n_pos:]), dim=-1, keepdim=True)
    pos_to_pos = torch.exp(sims[:, :n_pos, :n_pos])
    # Remove the diagonal elements,
    # from https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379
    pos_to_pos = (
        pos_to_pos.flatten(start_dim=1)[:, 1:]
        .view(B, n_pos - 1, n_pos + 1)[:, :, :-1]
        .reshape(B, n_pos, n_pos - 1)
    )
    probs = pos_to_pos / (pos_to_pos + pos_to_neg)

    return -torch.mean(torch.log(probs))