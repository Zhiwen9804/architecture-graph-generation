import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_GDSS import AttentionLayer, MLP
from utils.graph_utils import mask_x, mask_adjs, pow_tensor

def get_act(act='swish'):
    """Get actiuvation functions from the config file."""

    if act == 'elu':
        return nn.ELU()
    elif act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif act == 'swish':
        return nn.SiLU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation function does not exist!')

# from DDPM
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class time_embedding(nn.Module):
    def __init__(self, temb_size, nonlinearity='swish'):
        super().__init__()    
        
        self.temb_size = temb_size
        self.act = get_act(nonlinearity)
        
        self.layer1 = nn.Linear(self.temb_size, self.temb_size * 2)
        self.layer2 = nn.Linear(self.temb_size * 2, self.temb_size * 2)

    def forward(self, timesteps):
        temb = get_timestep_embedding(timesteps, self.temb_size)
        
        temb = self.layer1(temb)
        temb = self.layer2(self.act(temb))
        
        return temb    

class ScoreNetworkA(torch.nn.Module):

    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                    c_init, c_hid, c_final, adim, num_heads=4, temb_dim=16, conv='GCN', act='swish', is_temb='True'):

        super(ScoreNetworkA, self).__init__()

        self.nfeat = max_feat_num
        self.max_node_num = max_node_num
        self.nhid  = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.is_temb = is_temb

        #self.time_layer = nn.Linear(temb_dim, self.c_hid)
        ### version 2.1
        if self.is_temb:
            self.time_layer = nn.Linear(temb_dim, 1)
            self.time_act = get_act(act)
            
        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _==0:
                self.layers.append(AttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init, 
                                                    self.c_hid, self.num_heads, self.conv))
            elif _==self.num_layers-1:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_final, self.num_heads, self.conv))
            else:
                self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
                                                    self.c_hid, self.num_heads, self.conv))

        self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
                            use_bn=False, activate_func=F.elu)
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask.unsqueeze_(0)  
        
    def forward(self, x, adj, flags, temb):
        if self.is_temb:
            temb = self.time_layer(self.time_act(temb))
        
        adjc = pow_tensor(adj, self.c_init)
        adj_list = [adjc]
        for _ in range(self.num_layers):
            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)
            if self.is_temb:
                adjc = adjc + temb[:, :, None, None]
                x = x + temb[:, None, :]
            
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        
        out_shape = adjs.shape[:-1] # B x N x N
        score = self.final(adjs).view(*out_shape)

        self.mask = self.mask.to(score.device)
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score

class ScoreNetworkX(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, num_linears,
                 c_init, c_hid, c_final, adim, temb_dim=16, num_heads=4, conv='GCN', act='swish', is_temb='True'):
        super().__init__()

        self.depth = depth
        self.c_init = c_init
        self.is_temb = is_temb
        if self.is_temb:
            self.time_layer = nn.Linear(temb_dim, 1)
            
        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(AttentionLayer(num_linears, max_feat_num, nhid, nhid, c_init, 
                                                  c_hid, num_heads, conv))
            elif _ == self.depth - 1:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_final, num_heads, conv))
            else:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_hid, num_heads, conv))

        fdim = max_feat_num + depth * nhid
        self.final = MLP(num_layers=3, input_dim=fdim, hidden_dim=2*fdim, output_dim=max_feat_num, 
                         use_bn=False, activate_func=F.elu)

        self.time_act = get_act(act)
        self.activation = torch.tanh

    def forward(self, x, adj, flags, temb):
        if self.is_temb:
            temb = self.time_layer(self.time_act(temb))
        
        adjc = pow_tensor(adj, self.c_init)
        
        x_list = [x]
        for _ in range(self.depth):
            x, adjc = self.layers[_](x, adjc, flags)
            x_list.append(x)
            
            if self.is_temb:
                x = x + temb[:, None, :]
                adjc = adjc + temb[:, :, None, None]
            
            x = self.activation(x)
            
        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)

        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)

        return x    