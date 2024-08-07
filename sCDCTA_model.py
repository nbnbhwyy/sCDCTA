from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import copy
from .customized_linear import CustomizedLinear
from einops import rearrange
import random
import numpy as np
import pandas as pd
from utils import SparseGraphConvolution
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch.nn.functional as F
from scipy import sparse
from scipy.special import gamma

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class GCNEncoderWithFeatures(nn.Module):
    def __init__(self, num_features: int,
                 dropout: float = 0.5, bias: bool = False):
        super(GCNEncoderWithFeatures, self).__init__()
        self.input_dim = num_features
        self.dropout = nn.Dropout(dropout)
        gcn_hidden1=200
        gcn_hidden2=200
        GC = SparseGraphConvolution 
        self.gc_input = GC(in_features = num_features, out_features = gcn_hidden1, bias = bias)
        self.gc_hidden1 = GC(in_features = gcn_hidden1, out_features = gcn_hidden2, bias = bias)
        self.trans_h = nn.Linear(gcn_hidden1 + num_features, gcn_hidden1, bias = bias)
        self.trans_h1 = nn.Linear(gcn_hidden2 + num_features, gcn_hidden2, bias = bias)
    def forward(self, features: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        hidden1 = F.relu(self.trans_h(torch.cat([self.gc_input(features, adj), features], dim=1)))
        hidden1 =  self.dropout(hidden1)
        hidden2 = F.relu(self.trans_h1(torch.cat([self.gc_hidden1(hidden1, adj), features], dim=1)))
        return hidden1,hidden2

class HierGlobalGCN(nn.Module):
    def __init__(self, num_features: int,
                 dropout: float = 0.5):
        super(HierGlobalGCN, self).__init__()
        self.num_features = num_features
        self.dropout = nn.Dropout(dropout)
        self.global_enc = GCNEncoderWithFeatures(num_features)
        
    def forward(self,
               Ori, Adj, return_embeddings: bool = False):
        Ori = self.dropout(Ori)
        Ori_h1, Ori_h2 = self.global_enc(Ori.to(torch.float32).cuda(2), Adj.to(torch.float32).cuda(2))
        Ori_combination = self.dropout(Ori_h2)  
        return  Ori_combination
    
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class FeatureEmbed(nn.Module):
    def __init__(self, num_genes, mask, embed_dim=192, fe_bias=True, norm_layer=None):
        super().__init__()
        self.num_genes = num_genes
        self.num_patches = mask.shape[1]
        self.embed_dim = embed_dim
        mask = np.repeat(mask,embed_dim,axis=1) 
        self.mask = mask
        self.fe = CustomizedLinear(self.mask)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        num_cells = x.shape[0]
        x = rearrange(self.fe(x), 'h (w c) -> h c w ', c=self.num_patches)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features 
      #  self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads,
                 mlp_ratio=4., 
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0., 
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
    def forward(self, x):

        hhh, weights = self.attn(x)
        x = x + self.drop_path(hhh)
        return x, weights
    
def get_weight(att_mat):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    att_mat = torch.stack(att_mat).squeeze(1)
    #print(att_mat.size())
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)
    #print(att_mat.size())
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(3))
    aug_att_mat = att_mat.to(device) + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    #print(aug_att_mat.size())
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #print(joint_attentions.size())
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    #print(v.size())
    v = v[:,0,1:]
    #print(v.size())
    return v

def normalize_adj(adj: sp.csr_matrix) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj)
    adj_ = adj
    rowsum = np.array(adj_.sum(0))
    rowsum_power = []
    for i in rowsum:
        for j in i:
            if j !=0 :
                j_power = np.power(j, -0.5)
                rowsum_power.append(j_power)
            else:
                j_power = 0
                rowsum_power.append(j_power)
    rowsum_power = np.array(rowsum_power)
    degree_mat_inv_sqrt = sp.diags(rowsum_power)
    adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_norm

def sparse_mx_to_torch_sparse_tensor(sparse_mx) \
        -> torch.sparse.FloatTensor:
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def kernel_matrix(x: torch.Tensor, sigma):
    x1  = torch.unsqueeze(x, 0)
    x2  = torch.unsqueeze(x, 1)

    return torch.exp( -sigma * torch.sum(torch.pow(x1-x2, 2), axis=2) )


def bandwidth(d):
    """
    in the case of Gaussian random variables and the use of a RBF kernel,
    this can be used to select the bandwidth according to the median heuristic
    """
    gz = 2 * gamma(0.5 * (d+1)) / gamma(0.5 * d)
    return 1. / (2. * gz**2)

def hsic(z, s):
    d_z = z.shape[1]
    d_s = s.shape[1]

    zz = kernel_matrix(z, bandwidth(d_z))
    ss = kernel_matrix(s, bandwidth(d_s))

    h  = (zz * ss).mean() + zz.mean() * ss.mean() - 2 * torch.mean(zz.mean(1) * ss.mean(1))
    return h.sqrt()

class Transformer(nn.Module):
    def __init__(self, num_classes, num_genes, mask, fe_bias=True,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=FeatureEmbed, norm_layer=None,
                 act_layer=None,sample_index=None):
        """
        Args:
            num_classes (int): number of classes for classification head
            num_genes (int): number of feature of input(expData) 
            embed_dim (int): embedding dimension
            depth (int): depth of transformer 
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate 
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): feature embed layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        norm_layer2 = partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.feature_embed = embed_layer(num_genes, mask = mask, embed_dim=embed_dim, fe_bias=fe_bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.GCN = HierGlobalGCN(num_features = 1998)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                          norm_layer=norm_layer, act_layer = act_layer)
            self.blocks.append(copy.deepcopy(layer))
        self.norm = norm_layer(embed_dim)
        self.norm2 = norm_layer2(embed_dim)
  
        filepath = "path_data/Muraro_rela.npz"
        loaded_matrix = sparse.load_npz(filepath).toarray() 
        reconstructed_matrix = sparse.csr_matrix(loaded_matrix)
        self.rela = reconstructed_matrix[sample_index][:,sample_index]
        self.has_logits = False
        self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.embed_dim*2, num_classes,bias=False) if num_classes > 0 else nn.Identity() #
              
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.feature_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None: 
            x = torch.cat((cls_token, x), dim=1) 
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        attn_weights = []
        tem = x
        for layer_block in self.blocks:
            tem, weights = layer_block(tem)
            attn_weights.append(weights)
        x = tem
        attn_weights = get_weight(attn_weights)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]),attn_weights 
        else:
            return x[:, 0], x[:, 1],attn_weights

    def forward(self, x, x2, index):
        latent, attn_weights = self.forward_features(x.to(torch.float32))
        adj = csr_matrix(self.rela[index][:,index])    
        adj = normalize_adj(adj)     
        adj_norm = sparse_mx_to_torch_sparse_tensor(adj)

        benb = self.GCN(x.to(torch.float32), adj_norm)

        benb = self.norm(benb)
        latent = self.norm2(latent)  
        pre = self.head( torch.cat([latent,benb],dim=1).to(torch.float32)) 
        return latent, pre, attn_weights

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)  

def scTrans_model(num_classes, num_genes, mask, embed_dim=48,depth=2,num_heads=4,has_logits: bool = True,sample_index=None):
    model = Transformer(num_classes=num_classes, 
                        num_genes=num_genes, 
                        mask = mask,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads,
                        drop_ratio=0.5, attn_drop_ratio=0.5, drop_path_ratio=0.5,sample_index = sample_index,
                        representation_size=embed_dim if has_logits else None)
    return model

