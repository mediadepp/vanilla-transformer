from torch import nn
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.layer_norm import LayerNorm
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)  
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model) 
        self.dropout2 = nn.Dropout(p=drop_prob) 

    def forward(self, x, s_mask):
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        x = self.dropout2(x) 

        return x
        