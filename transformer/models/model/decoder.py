import torch
from torch import nn
from models.embeddings.transformer_embedding import TransformerEmbedding 
from models.blocks.decoder_layer import DecoderLayer 


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__() 
        self.emb = TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                                     for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.linear(trg)
        return output 
