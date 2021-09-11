import random

import numpy as np
import torch
import torch.nn as nn

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class PathEncoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 node_num,
                 hid_dim, 
                 dropout):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)

        self.W_level = nn.Embedding(node_num, hid_dim)
        self.W_parent = nn.Embedding(node_num, hid_dim)

        self.linear = nn.Linear(node_num*hid_dim, hid_dim)

        self.layer_norm = nn.LayerNorm(hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, pos_coeff, src_subtoken_mask):
        
        # subtoken summation
        src = self.tok_embedding(src)
        src_subtoken_mask = src_subtoken_mask.unsqueeze(4)
        src = src*src_subtoken_mask
        src = src.sum(axis=3)
        
        # add node position embedding
        l = torch.arange(0, src.shape[2]).unsqueeze(0).unsqueeze(1).to(src.device)
        E_pos  = pos_coeff.unsqueeze(3) * self.W_parent(l) + (1 - pos_coeff).unsqueeze(3) * self.W_level(l)

        src = self.linear(torch.reshape(src + E_pos, (src.shape[0], src.shape[1], -1)))

        src = self.dropout(src)

        src = self.layer_norm(src)

        return src 
        
class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 node_num,
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 max_length = 200):
        super().__init__()
        
        self.pathEncode = PathEncoder(input_dim, node_num, hid_dim, dropout)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
        
    def forward(self, src, pos_coeff, src_mask, src_subtoken_mask):
        src = self.pathEncode(src, pos_coeff, src_subtoken_mask)

        for layer in self.layers:
            src = layer(src, src_mask)
    
        return src

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)

        src = self.self_attn_layer_norm(src + self.dropout(_src))

        _src = self.positionwise_feedforward(src)

        src = self.ff_layer_norm(src + self.dropout(_src))

        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
     
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.FloatTensor([self.head_dim])).to(Q.device)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
       
        x = torch.matmul(self.dropout(attention), V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))

        x = self.fc_2(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout,
                 max_length = 210):
        super().__init__()

        self.hid_dim = hid_dim
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout)
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).to(trg.device)
   
        trg = self.dropout(self.tok_embedding(trg) * torch.sqrt(torch.FloatTensor([self.hid_dim])).to(trg.device) + self.pos_embedding(pos))

        for layer in self.layers:
            output = layer(trg, enc_src, trg_mask, src_mask)
   
        return output

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        _trg, _ = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.positionwise_feedforward(trg)

        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 hidden,
                 pad_idx):
        super().__init__()
        
        self.hidden = hidden
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = pad_idx
        self.trg_pad_idx = pad_idx
        
    def make_src_mask(self, src):
        # Determines if the last dimensional vector is a zero vector and returns the mask matrix (0+0+... +0 = 0)
        src_mask = (src.sum(axis=3).sum(axis=2) != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        src_subtoken_mask = (src != self.src_pad_idx)
        
        return src_mask, src_subtoken_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = trg_pad_mask.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, pos_coeff, trg):
        src_mask, src_subtoken_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, pos_coeff, src_mask, src_subtoken_mask)

        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output
