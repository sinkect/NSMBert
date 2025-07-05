import torch.nn as nn
import torch


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.attn(x, x, x, key_padding_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        out = self.linear(attn_output)
        out = self.norm(out + x)
        return out


class TransitionBERT(nn.Module):
    def __init__(self, codebook_size, embed_dim, max_length, num_layers, num_heads):
        super(TransitionBERT, self).__init__()
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        self.positioning = nn.Embedding(max_length, embed_dim)
        self.layers = nn.ModuleList([AttentionBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x, attn_mask):
        x = self.embedding(x) + self.positioning(torch.arange(x.size(1)).unsqueeze(0).to(x.device))
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.norm(x)
        cls_token = x[:, 0, :]
        classifier = self.classifier(cls_token) # cls pooling
        return classifier
