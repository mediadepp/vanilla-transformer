from tkinter import W
import torch

from torch import nn 
from models.model.encoder import Encoder
from models.model.decoder import Decoder 


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx 
        self.trg_sos_idx = trg_sos_idx 
        self.device = device 
        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src)  
        src_trg_mask = self.make_pad_mask(trg, src)  # What will happen to this at test time? we don't have target. 
        trg_mask = self.make_pad_mask(trg, trg) * self.make_no_peak_mask(trg, trg)  
        enc_src = self.encoder(src, src_mask) 
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        return output 

    def make_pad_mask(self, q, k):
        len_q, len_k = q.shape[1], k.shape[1]
        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        mask = k & q
        return mask 
    
    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.shape[1], k.shape[1] 
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device) 
        return mask 
