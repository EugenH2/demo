import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x



class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=10000):
        super().__init__()

        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
      
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        dropout=0.0
    ):        
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = PositionalEmbedding(embed_dim, 513)

    def forward(self, x):
        # Preprocess input:

        # Split into patches
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
        x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]

        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.pos_embedding(x)

        # Transformer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Classification 
        x = self.mlp_head(x[0])
        return x




class LSTMModel(nn.Module):
    def __init__(self, num_classes):
        super(LSTMModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)


        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding = 1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, 3, 1, padding = 1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, padding = 1)
        self.bn6 = nn.BatchNorm2d(64)

        
        self.conv7 = nn.Conv2d(64, 128, 3, 2, padding = 1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, padding = 1)
        self.bn8 = nn.BatchNorm2d(128)
        
        self.downsample1 = nn.Conv2d(64, 128, 3, 2, padding = 1) 
        self.bnD1 = nn.BatchNorm2d(128)

   
        self.flatten = nn.Flatten()
        self.avg_pool = nn.AvgPool2d((8, 64))
        self.lstm = nn.LSTM(128, 2048, 1, batch_first=True, proj_size=128, bidirectional=True)
        
        self.drop = nn.Dropout(p=0.4)
        
        self.fc1 = nn.Linear(256, num_classes)



    def forward(self, x):
        x = F.relu(self.bn1( self.conv1(x) ))
        x = F.relu(self.bn2( self.conv2(x) ))
        
       
        residual = x.clone() 
        x = F.relu(self.bn3( self.conv3(x) ))
        x = F.relu(self.bn4( self.conv4(x) ) + residual)

        residual = x.clone()
        x = F.relu(self.bn5( self.conv5(x) ))
        x = F.relu(self.bn6( self.conv6(x) ) + residual)


        residual = x.clone()
        x = F.relu(self.bn7( self.conv7(x) ))
        x = F.relu( self.bn8( self.conv8(x) ) + self.bnD1( self.downsample1(residual) ) )
       

        x = self.avg_pool(x)
        x = self.flatten(x)

        x = self.lstm(x)[0]
        x = self.drop(x)

        x = self.fc1(x)



        return x



