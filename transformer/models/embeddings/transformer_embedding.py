from torch import nn
from models.embeddings.positional_encoding import PositionalEncoding  
from models.embeddings.token_embedding import TokenEmbedding 


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        res = self.tok_emb(x) + self.pos_emb(x)
        res = self.drop_out(res)
        return res 
