import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

# from transformers import Wav2Vec2Model
# from transformers import Wav2Vec2Config

import math  # FurtherImprovedMEGClassifierのために追加


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)



###以下に新しく追加していった

# ## カスタムモデル1 MEGClassifier
# class MEGClassifier(nn.Module):
#     def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2))
#         self.dropout = nn.Dropout(0.5)
        
#         # 畳み込み層を通した後のサイズを計算
#         conv_out_size = self._get_conv_output_size((1, in_channels, seq_len))
        
#         self.fc1 = nn.Linear(conv_out_size, hid_dim)
#         self.fc2 = nn.Linear(hid_dim, num_classes)

#     def _get_conv_output_size(self, shape):
#         input = torch.rand(1, *shape)
#         output = self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(input))))))
#         return int(torch.prod(torch.tensor(output.size())))

#     def forward(self, x):
        
#         x = x.unsqueeze(1)  # Add channel dimension
        
#         x = self.pool(F.relu(self.conv1(x)))
        
#         x = self.pool(F.relu(self.conv2(x)))
        
#         x = x.view(x.size(0), -1)  # Flatten
        
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
        
#         return x

#  ## カスタムモデル2 ImprovedMEGClassifier
# class ImprovedMEGClassifier(nn.Module):
#     def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128, num_heads: int = 8, num_layers: int = 2):
#         super().__init__()
        
#         # Initial convolutional layers
#         self.conv_blocks = nn.Sequential(
#             ConvBlock(in_channels, hid_dim),
#             ConvBlock(hid_dim, hid_dim),
#         )
        
#         # Transformer layers
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=hid_dim, nhead=num_heads, dim_feedforward=hid_dim*4),
#             num_layers=num_layers
#         )
        
#         # Final classification layers
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             Rearrange("b d 1 -> b d"),
#             nn.Linear(hid_dim, num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Initial convolutional processing
#         x = self.conv_blocks(x)
        
#         # Prepare for Transformer (b, d, t) -> (t, b, d)
#         x = x.permute(2, 0, 1)
        
#         # Apply Transformer
#         x = self.transformer(x)
        
#         # Prepare for final classification (t, b, d) -> (b, d, t)
#         x = x.permute(1, 2, 0)
        
#         # Final classification
#         return self.head(x)


# #カスタムモデル3 FurtherImprovedMEGClassifier,そのためにconvblockも変更
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops.layers.torch import Rearrange
# from einops import rearrange
# import math

# class FurtherImprovedMEGClassifier(nn.Module):
#     def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
#         super().__init__()
        
#         # Initial convolutional layers with skip connections
#         self.conv_blocks = nn.ModuleList([
#             ConvBlockWithSkip(in_channels, hid_dim // 2, kernel_size=7, stride=2),
#             ConvBlockWithSkip(hid_dim // 2, hid_dim, kernel_size=3, stride=1),
#             ConvBlockWithSkip(hid_dim, hid_dim, kernel_size=3, stride=1),
#         ])
        
#         # Positional encoding
#         self.pos_encoding = PositionalEncoding(hid_dim, dropout)
        
#         # Transformer layers
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=num_heads, 
#                                                    dim_feedforward=hid_dim*4, dropout=dropout, 
#                                                    batch_first=True, norm_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Final classification layers
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             Rearrange("b d 1 -> b d"),
#             nn.Linear(hid_dim, hid_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hid_dim // 2, num_classes),
#         )
        
#         # Layer normalization
#         self.layer_norm = nn.LayerNorm(hid_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Initial convolutional processing with skip connections
#         for conv_block in self.conv_blocks:
#             x = conv_block(x)
        
#         # Prepare for Transformer (b, d, t) -> (b, t, d)
#         x = x.permute(0, 2, 1)
        
#         # Apply positional encoding
#         x = self.pos_encoding(x)
        
#         # Apply Transformer
#         x = self.transformer(x)
        
#         # Apply layer normalization
#         x = self.layer_norm(x)
        
#         # Prepare for final classification (b, t, d) -> (b, d, t)
#         x = x.permute(0, 2, 1)
        
#         # Final classification
#         return self.head(x)

# class ConvBlockWithSkip(nn.Module):
#     def __init__(self, in_dim, out_dim, kernel_size: int = 3, stride: int = 1, p_drop: float = 0.1):
#         super().__init__()
        
#         self.conv_path = nn.Sequential(
#             nn.Conv1d(in_dim, out_dim, kernel_size, stride=stride, padding=kernel_size//2),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU(),
#             nn.Conv1d(out_dim, out_dim, kernel_size, stride=1, padding=kernel_size//2),
#             nn.BatchNorm1d(out_dim)
#         )
        
#         self.skip_connection = nn.Sequential(
#             nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=stride),
#             nn.BatchNorm1d(out_dim)
#         ) if in_dim != out_dim or stride != 1 else nn.Identity()
        
#         self.activation = nn.ReLU()
#         self.dropout = nn.Dropout(p_drop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.dropout(self.activation(self.conv_path(x) + self.skip_connection(x)))

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


# ##Wav2Vec2
# class Wav2Vec2ConvClassifier(nn.Module):
#     def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128):
#         super().__init__()

#         # Wav2Vec2の設定を変更してマスキングを無効化
#         config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
#         config.mask_time_prob = 0.0
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", config=config)
#         self.wav2vec2.feature_extractor._freeze_parameters()

#         wav2vec2_dim = self.wav2vec2.config.hidden_size

#         self.blocks = nn.Sequential(
#             ConvBlock(wav2vec2_dim, hid_dim),
#             ConvBlock(hid_dim, hid_dim),
#         )

#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             Rearrange("b d 1 -> b d"),
#             nn.Linear(hid_dim, num_classes),
#         )

#         # 入力シーケンス長とチャンネル数を保存
#         self.seq_len = seq_len
#         self.in_channels = in_channels

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         print(f"Initial input shape: {X.shape}")
        
#         # (b, c, t) -> (b, t)
#         X = X.mean(dim=1)  # チャンネル方向に平均を取る
#         print(f"After channel mean: {X.shape}")
        
#         # パディングを追加（必要に応じて調整）
#         target_length = max(400, self.seq_len)  # Wav2Vec2が期待する最小長さと元の長さの大きい方
#         if X.shape[1] < target_length:
#             padding = target_length - X.shape[1]
#             X = F.pad(X, (0, padding))
#         print(f"After padding: {X.shape}")
        
#         # 入力が2D (batch_size, sequence_length) であることを確認
#         if X.dim() == 2:
#             X = X.unsqueeze(1)  # (batch_size, 1, sequence_length)
#         print(f"Before wav2vec2: {X.shape}")
        
#         # Wav2Vec2モデルを適用
#         X = self.wav2vec2(X).last_hidden_state
#         print(f"After wav2vec2: {X.shape}")
        
#         # 残りの処理...
#         X = X.permute(0, 2, 1)  # (batch, time, features) -> (batch, features, time)
#         X = self.blocks(X)
#         X = self.head(X)
#         return X


# ##Wav2Vec2+3D畳み込みの導入
# class Wav2Vec2MEGClassifier(nn.Module):
#     def __init__(self, num_classes, num_channels=271):
#         super().__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
#         self.wav2vec2.feature_extractor._freeze_parameters()

#         # 3D畳み込み層
#         self.conv3d = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
#         # Global Average Pooling
#         self.gap = nn.AdaptiveAvgPool3d(1)
        
#         # 全結合層
#         self.fc = nn.Linear(32, num_classes)

#     def forward(self, x):
#         # x shape: (batch_size, channels, time_steps)
#         batch_size, channels, time_steps = x.shape
        
#         # Wav2Vec2の入力形状に合わせる
#         x = x.permute(0, 2, 1)  # (batch_size, time_steps, channels)
        
#         # Wav2Vec2を適用
#         x = self.wav2vec2(x).last_hidden_state
        
#         # 3D畳み込みのための形状変更
#         x = x.permute(0, 2, 1).unsqueeze(1)  # (batch_size, 1, channels, time_steps, features)
        
#         # 3D畳み込みを適用
#         x = self.conv3d(x)
        
#         # Global Average Pooling
#         x = self.gap(x).squeeze()
        
#         # 分類
#         x = self.fc(x)
        
#         return x


# ##Wav2Vec2+空間的注意機構の追加
# class SpatialAttention(nn.Module):
#     def __init__(self, in_features):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(in_features, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         # x shape: (batch_size, channels, time_steps, features)
#         attn_weights = self.attention(x.permute(0, 1, 3, 2)).squeeze(-1)
#         attn_weights = F.softmax(attn_weights, dim=1)
#         return (x * attn_weights.unsqueeze(-1)).sum(dim=1)

# class Wav2Vec2MEGClassifierWithAttention(nn.Module):
#     def __init__(self, num_classes, num_channels=271):
#         super().__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
#         self.wav2vec2.feature_extractor._freeze_parameters()

#         wav2vec2_dim = self.wav2vec2.config.hidden_size
#         self.spatial_attention = SpatialAttention(wav2vec2_dim)
        
#         self.fc = nn.Sequential(
#             nn.Linear(wav2vec2_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         # x shape: (batch_size, channels, time_steps)
#         batch_size, channels, time_steps = x.shape
        
#         # Wav2Vec2の入力形状に合わせる
#         x = x.permute(0, 2, 1)  # (batch_size, time_steps, channels)
        
#         # Wav2Vec2を適用
#         x = self.wav2vec2(x).last_hidden_state
        
#         # 空間的注意機構を適用
#         x = x.permute(0, 2, 1)  # (batch_size, channels, time_steps, features)
#         x = self.spatial_attention(x)
        
#         # 分類
#         x = self.fc(x)
        
#         return x


## EEG Conformer
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (271, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 8), (1, 8)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Rearrange('b n e -> b (n e)'),
            nn.LayerNorm(emb_size * 32),  # 32は出力シーケンス長に応じて調整
            nn.Linear(emb_size * 32, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        return self.clshead(x)

    def forward(self, x):
        return self.clshead(x)

class EEGConformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=1854, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )