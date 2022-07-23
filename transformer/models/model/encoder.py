from torch import nn
from models.blocks.encoder_layer import EncoderLayer
from models.embeddings.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(
            vocab_size=enc_voc_size,
            d_model=d_model, 
            max_len=max_len, 
            drop_prob=drop_prob, 
            device=device
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)]
        )
        
    def forward(self, x, s_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x
