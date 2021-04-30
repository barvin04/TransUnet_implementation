import torch
import torch.nn as nn
import math

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.0):
        super(MLP, self).__init__()
        self.activation = nn.GELU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Multi Head Self Attention
class Attention(nn.Module):
    '''
    Taken / Adapted from the implementation present at
    https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py#L52
    '''
    def __init__(self, hidden_size, dropout_rate):
        super(Attention, self).__init__()
        self.num_attention_heads = 12
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class PatchEmbed(nn.Module):
    def __init__(self, input_dim, img_size, patch_size, emb_dim):
        super(PatchEmbed, self).__init__()

        self.proj_layer = nn.Conv2d(input_dim, emb_dim, patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embs= nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.proj_layer(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = x + self.pos_embs
        x = self.dropout(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, emb_dim):
        super(TransformerLayer, self).__init__()

        mlp_size = 3072 # paper 4.4 model scaling 
        
        self.attention_norm = nn.LayerNorm(emb_dim, eps=1e-6) 
        self.attention_layer = Attention(hidden_size=emb_dim, dropout_rate=0.1)
        
        self.feedforward_norm = nn.LayerNorm(emb_dim, eps=1e-6)   

        self.mlp = MLP(emb_dim, mlp_size, emb_dim, 0.1)

    def forward(self, x):
        mhsa_norm_out = self.attention_layer(self.attention_norm(x))
        x = x + mhsa_norm_out
        x_ffn = self.feedforward_norm(x)
        x_mlp = self.mlp(x_ffn)
        x = x + x_mlp
        return x
