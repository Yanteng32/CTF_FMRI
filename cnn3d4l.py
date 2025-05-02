import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

'''
# 自定义数据集类，用于加载数据
class BrainMatrixDataset(Dataset):
    def __init__(self, csv_files):
        # 加载所有CSV文件
        self.data = []
        for csv_file in csv_files:
            matrix = pd.read_csv(csv_file, header=None).values
            matrix = matrix.astype(np.float32)  # 确保是浮点型
            self.data.append(matrix)
        self.data = np.array(self.data)
        self.data = self.data[:, np.newaxis, :, :]  # 增加通道维度 (batch_size, 1, 105, 105)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])
'''

class SELayer(nn.Module):
	def __init__(self, channel, reduction=1):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool3d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid(),
		)

	def forward(self, x):
		b, c, _, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1, 1)
		return x * y.expand_as(x)


class BasicConv(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
	             bn=True, bias=False):
		super(BasicConv, self).__init__()
		self.out_channels = out_planes
		self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
		                      dilation=dilation, groups=groups, bias=bias)
		self.bn = nn.BatchNorm3d(out_planes, momentum=0.01, affine=True) if bn else None
		self.relu = nn.ReLU() if relu else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x


class ChannelPool(nn.Module):
	def forward(self, x):
		return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
	def __init__(self):
		super(SpatialGate, self).__init__()
		kernel_size = 3
		self.compress = ChannelPool()
		self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

	def forward(self, x):
		x_compress = self.compress(x)
		x_out = self.spatial(x_compress)
		scale = torch.sigmoid(x_out)
		self.attention = scale
		return x * self.attention



class ChannelAttention2D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class MultiHeadAttention2D(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(MultiHeadAttention2D, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear transformations for Q, K, and V
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)

        # Linear transformation for output
        self.W_out = nn.Linear(input_dim, input_dim)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        # Linear transformation
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # Splitting into multiple heads
        Q = Q.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and apply final linear transformation
        attention_output = attention_output.transpose(0, 1).contiguous().view(batch_size, -1)
        output = self.W_out(attention_output)
        return output


class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads=8):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.head_dim = in_channels // heads
        self.heads = heads
        assert self.head_dim * heads == in_channels, "Incompatible number of heads and in_channels"

        self.values = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.keys = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.queries = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0,
                                 bias=False)
        self.fc_out = nn.Conv3d(self.heads * self.head_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.norm = nn.BatchNorm3d(in_channels)  # 归一化层
        self.activation = nn.ReLU()  # 激活函数

    def forward(self, x):
        N, C, D, H, W = x.size()
        residual = x  # 保存输入以用于残差连接
        # 添加空间位置向量嵌入的部分

        # Apply convolutions to values, keys, and queries
        values = self.values(x).view(N, self.heads, self.head_dim, D, H, W)
        keys = self.keys(x).view(N, self.heads, self.head_dim, D, H, W)
        queries = self.queries(x).view(N, self.heads, self.head_dim, D, H, W)

        # Permute dimensions for matrix multiplication
        values = values.permute(0, 1, 3, 4, 5, 2).contiguous()
        keys = keys.permute(0, 1, 3, 4, 5, 2).contiguous()
        queries = queries.permute(0, 1, 3, 4, 5, 2).contiguous()

        # Calculate attention scores
        energy = torch.einsum("nhdxyz,nhexyz->nhedxy", [queries, keys])
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.einsum("nhedxy,nhdxyz->nhexyz", [attention, values]).reshape(N, self.heads * self.head_dim, D, H, W)

        # Reshape and apply final convolution
        out = self.fc_out(out)
        out = self.norm(out)
        out = self.activation(out)
        out = out + residual

        return out


class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        if x.dim() == 4:
            # 2D positional encoding
            pe = torch.zeros(x.size(0), x.size(2), x.size(3), self.d_model)
            pe.requires_grad = False
            pos = torch.arange(0, x.size(2), dtype=torch.float).unsqueeze(0).unsqueeze(0)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            pe[:, :, :, 0::2] = torch.sin(pos * div_term)
            pe[:, :, :, 1::2] = torch.cos(pos * div_term)
        elif x.dim() == 5:
            # 3D positional encoding
            pe = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4), self.d_model)
            pe.requires_grad = False
            # print(pe.shape)
            pos = torch.arange(0, self.d_model / 2, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            # print(pos.shape)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            pe[:, :, :, :, 0::2] = torch.sin(pos * div_term)
            pe[:, :, :, :, 1::2] = torch.cos(pos * div_term)
        else:
            raise ValueError("Positional encoding input must have 4 or 5 dimensions")

        # 将 pe 移动到与 x 相同的设备上
        pe = pe.permute(0, 4, 1, 2, 3)
        pe = pe.to(x.device)
        return x + pe


class DecoderSelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(DecoderSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.head_dim = in_channels // heads
        self.heads = heads
        assert self.head_dim * heads == in_channels, "Incompatible number of heads and in_channels"

        self.values = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.keys = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.queries = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0,
                                 bias=False)
        self.fc_out = nn.Conv3d(self.heads * self.head_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.norm = nn.BatchNorm3d(in_channels)  # 归一化层
        self.activation = nn.ReLU()  # 激活函数

    def forward(self, x, encoder_out):
        N, C, D, H, W = x.size()
        residual = x  # 保存输入以用于残差连接
        # 添加空间位置向量嵌入的部分

        # Apply convolutions to values, keys, and queries
        values = self.values(encoder_out).view(N, self.heads, self.head_dim, D, H, W)
        keys = self.keys(encoder_out).view(N, self.heads, self.head_dim, D, H, W)
        queries = self.queries(x).view(N, self.heads, self.head_dim, D, H, W)

        # Permute dimensions for matrix multiplication
        values = values.permute(0, 1, 3, 4, 5, 2).contiguous()
        keys = keys.permute(0, 1, 3, 4, 5, 2).contiguous()
        queries = queries.permute(0, 1, 3, 4, 5, 2).contiguous()

        # Calculate attention scores
        energy = torch.einsum("nhdxyz,nhexyz->nhedxy", [queries, keys])
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.einsum("nhedxy,nhdxyz->nhexyz", [attention, values]).reshape(N, self.heads * self.head_dim, D, H, W)

        # Reshape and apply final convolution
        out = self.fc_out(out)
        out = self.norm(out)
        out = self.activation(out)
        out = out + residual

        return out


class CognitiveAttentionModule(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(CognitiveAttentionModule, self).__init__()
        self.multihead_self_attention = MultiHeadAttention2D(input_dim, num_heads)
        self.multihead_cross_attention = MultiHeadAttention2D(input_dim, num_heads)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, input_q, input_k, input_v):
        # Self-attention
        input = self.layer_norm(input_q)
        self_attention_output = self.multihead_self_attention(input, input, input)
        self_attention_output = self_attention_output + input
        # Cross-attention
        cross_attention_output = self.multihead_cross_attention(self_attention_output, input_k, input_v)
        output = self.layer_norm(cross_attention_output + input_v)
        return output



class CNN4D_Net4l(nn.Module):
	def __init__(self, dropout=0):
		nn.Module.__init__(self)
		# self.SpatialGate = SpatialGate()
		self.feature_extractor1 = nn.Sequential(
			nn.Conv3d(1, 16, 3),
			nn.BatchNorm3d(16),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
			nn.BatchNorm3d(16),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			#nn.Conv3d(128, 128, 3,padding=1), #尺寸太小需要补零保持维度
            nn.Conv3d(128, 128, 1),
			nn.BatchNorm3d(128),
			nn.ReLU(),
		)
		#self.ca4 = ChannelAttention2D(128)
		self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)
		self.QL = nn.Parameter(torch.randn(1, 128) * nn.init.xavier_normal_(torch.empty(1, 128)))
		self.cam = CognitiveAttentionModule(128, 8)
	def forward(self, x):
		batch_size, _, d1, d2, d3, time_steps = x.shape
		Q = self.QL.expand(batch_size, -1)  # 扩展到 batch_size
		for t in range(time_steps):
			x_t = x[..., t]  # 提取时间片段，形状 [batch_size, 1, 53, 63, 52]
			features_x = self.feature_extractor1(x_t)  # 特征提取
            #features_x = self.ca4(features_x)
			features_x = self.pool(features_x)  # 全局池化，形状 [batch_size, 128, 1, 1, 1]
			features_x = features_x.view(batch_size, -1)  # 变为 [batch_size, 128]
			fx = self.cam(Q, features_x, features_x)  # CAM 融合

		logits = self.classifier(fx)  # 分类
		return logits, fx


class CNN4D_Net3l(nn.Module):
	def __init__(self, dropout=0):
		nn.Module.__init__(self)
		# self.SpatialGate = SpatialGate()
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
			nn.BatchNorm3d(128),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
			nn.BatchNorm3d(128),
			nn.ReLU(),
		)
		#self.ca4 = ChannelAttention2D(128)
		self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)
		self.QL = nn.Parameter(torch.randn(1, 128) * nn.init.xavier_normal_(torch.empty(1, 128)))
		self.cam = CognitiveAttentionModule(128, 8)
	def forward(self, x):
		batch_size, _, d1, d2, d3, time_steps = x.shape
		Q = self.QL.expand(batch_size, -1)  # 扩展到 batch_size
		for t in range(time_steps):
			x_t = x[..., t]  # 提取时间片段，形状 [batch_size, 1, 53, 63, 52]
			features_x = self.feature_extractor(x_t)  # 特征提取
			features_x = self.pool(features_x)  # 全局池化，形状 [batch_size, 128, 1, 1, 1]
			features_x = features_x.view(batch_size, -1)  # 变为 [batch_size, 128]
			fx = self.cam(Q, features_x, features_x)  # CAM 融合

		logits = self.classifier(fx)  # 分类
		return logits, fx


class CNN4D_Net3l_64tp4(nn.Module):
	def __init__(self, dropout=0):
		nn.Module.__init__(self)
		# self.SpatialGate = SpatialGate()
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
			nn.BatchNorm3d(16),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
			nn.BatchNorm3d(16),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),            
			nn.Conv3d(16, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.Conv3d(64, 64, 1),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			#nn.Conv3d(64, 64, 1),
			#nn.BatchNorm3d(64),
			#nn.ReLU(),          
		)
		self.locat = PositionalEncoding3D(64)
		self.ca = SelfAttention(64)
		self.de = DecoderSelfAttention(64)
		self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 2),
		)
		self.QL = nn.Parameter(torch.randn(1, 64) * nn.init.xavier_normal_(torch.empty(1, 64)))
		self.cam = CognitiveAttentionModule(64, 8)
	def forward(self, x):
		batch_size, _, d1, d2, d3, time_steps = x.shape
		Q = self.QL.expand(batch_size, -1)  # 扩展到 batch_size
		for t in range(time_steps):
			x_t = x[..., t]  # 提取时间片段，形状 [batch_size, 1, 53, 63, 52]
			features_x = self.feature_extractor(x_t)  # 特征提取
			features_x = self.locat(features_x)
			map = self.ca(features_x)
			features_x = self.de(features_x, map) * features_x
			features_x = self.pool(features_x)  # 全局池化，形状 [batch_size, 128, 1, 1, 1]
			features_x = features_x.view(batch_size, -1)  # 变为 [batch_size, 128]
			fx = self.cam(Q, features_x, features_x)  # CAM 融合

		logits = self.classifier(fx)  # 分类
		return logits, fx


class CNN4D_Net3l_128tp4(nn.Module):
	def __init__(self, dropout=0):
		nn.Module.__init__(self)
		# self.SpatialGate = SpatialGate()
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.Conv3d(128, 128, 1),
			nn.BatchNorm3d(128),
			nn.ReLU(),
            #nn.Conv3d(128, 128, 1),
			#nn.BatchNorm3d(128),
			#nn.ReLU(),          
		)
		self.locat = PositionalEncoding3D(128)
		self.ca = SelfAttention(128)
		self.de = DecoderSelfAttention(128)
		self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)
		self.QL = nn.Parameter(torch.randn(1, 128) * nn.init.xavier_normal_(torch.empty(1, 128)))
		self.cam = CognitiveAttentionModule(128, 8)
	def forward(self, x):
		batch_size, _, d1, d2, d3, time_steps = x.shape
		Q = self.QL.expand(batch_size, -1)  # 扩展到 batch_size
		for t in range(time_steps):
			x_t = x[..., t]  # 提取时间片段，形状 [batch_size, 1, 53, 63, 52]
			features_x = self.feature_extractor(x_t)  # 特征提取
			features_x = self.locat(features_x)
			map = self.ca(features_x)
			features_x = self.de(features_x, map) * features_x
			features_x = self.pool(features_x)  # 全局池化，形状 [batch_size, 128, 1, 1, 1]
			features_x = features_x.view(batch_size, -1)  # 变为 [batch_size, 128]
			fx = self.cam(Q, features_x, features_x)  # CAM 融合

		logits = self.classifier(fx)  # 分类
		return logits, fx

        
class CNN4D_Net4l_64tp4(nn.Module):
	def __init__(self, dropout=0):
		nn.Module.__init__(self)
		# self.SpatialGate = SpatialGate()
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 8, 3),
			nn.BatchNorm3d(8),
			nn.ReLU(),
			nn.Conv3d(8, 8, 3),
			nn.BatchNorm3d(8),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(8, 16, 3),
			nn.BatchNorm3d(16),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
			nn.BatchNorm3d(16),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),            
			nn.Conv3d(16, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.Conv3d(64, 64, 1),
			nn.BatchNorm3d(64),
			nn.ReLU(),          
		)
		self.locat = PositionalEncoding3D(64)
		self.ca = SelfAttention(64)
		self.de = DecoderSelfAttention(64)
		self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 2),
		)
		self.QL = nn.Parameter(torch.randn(1, 64) * nn.init.xavier_normal_(torch.empty(1, 64)))
		self.cam = CognitiveAttentionModule(64, 8)
	def forward(self, x):
		batch_size, _, d1, d2, d3, time_steps = x.shape
		Q = self.QL.expand(batch_size, -1)  # 扩展到 batch_size
		for t in range(time_steps):
			x_t = x[..., t]  # 提取时间片段，形状 [batch_size, 1, 53, 63, 52]
			features_x = self.feature_extractor(x_t)  # 特征提取
			features_x = self.locat(features_x)
			map = self.ca(features_x)
			features_x = self.de(features_x, map) * features_x
			features_x = self.pool(features_x)  # 全局池化，形状 [batch_size, 128, 1, 1, 1]
			features_x = features_x.view(batch_size, -1)  # 变为 [batch_size, 128]
			fx = self.cam(Q, features_x, features_x)  # CAM 融合

		logits = self.classifier(fx)  # 分类
		return logits, fx        
    
    
class CNN4D_Net4l_128tp4(nn.Module):
	def __init__(self, dropout=0):
		nn.Module.__init__(self)
		# self.SpatialGate = SpatialGate()
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
			nn.BatchNorm3d(16),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
			nn.BatchNorm3d(16),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),            
			nn.Conv3d(16, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.Conv3d(128, 128, 1),
			nn.BatchNorm3d(128),
			nn.ReLU(),          
		)
		self.locat = PositionalEncoding3D(128)
		self.ca = SelfAttention(128)
		self.de = DecoderSelfAttention(128)
		self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)
		self.QL = nn.Parameter(torch.randn(1, 128) * nn.init.xavier_normal_(torch.empty(1, 128)))
		self.cam = CognitiveAttentionModule(128, 8)
	def forward(self, x):
		batch_size, _, d1, d2, d3, time_steps = x.shape
		Q = self.QL.expand(batch_size, -1)  # 扩展到 batch_size
		for t in range(time_steps):
			x_t = x[..., t]  # 提取时间片段，形状 [batch_size, 1, 53, 63, 52]
			features_x = self.feature_extractor(x_t)  # 特征提取
			features_x = self.locat(features_x)
			map = self.ca(features_x)
			features_x = self.de(features_x, map) * features_x
			features_x = self.pool(features_x)  # 全局池化，形状 [batch_size, 128, 1, 1, 1]
			features_x = features_x.view(batch_size, -1)  # 变为 [batch_size, 128]
			fx = self.cam(Q, features_x, features_x)  # CAM 融合

		logits = self.classifier(fx)  # 分类
		return logits, fx



class CNN4D_Net4l_256tp4(nn.Module):
	def __init__(self, dropout=0):
		nn.Module.__init__(self)
		# self.SpatialGate = SpatialGate()
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
			nn.BatchNorm3d(32),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
			nn.BatchNorm3d(64),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
			nn.BatchNorm3d(128),
			nn.ReLU(),
            nn.MaxPool3d(2, stride=2),           
			nn.Conv3d(128, 256, 3),
			nn.BatchNorm3d(256),
			nn.ReLU(),
			nn.Conv3d(256, 256, 1),
			nn.BatchNorm3d(256),
			nn.ReLU(),
		)
		self.locat = PositionalEncoding3D(256)
		self.ca = SelfAttention(256)
		self.de = DecoderSelfAttention(256)
		self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 2),
		)
		self.QL = nn.Parameter(torch.randn(1, 256) * nn.init.xavier_normal_(torch.empty(1, 256)))
		self.cam = CognitiveAttentionModule(256, 8)
	def forward(self, x):
		batch_size, _, d1, d2, d3, time_steps = x.shape
		Q = self.QL.expand(batch_size, -1)  # 扩展到 batch_size
		for t in range(time_steps):
			x_t = x[..., t]  # 提取时间片段，形状 [batch_size, 1, 53, 63, 52]
			features_x = self.feature_extractor(x_t)  # 特征提取
			features_x = self.locat(features_x)
			map = self.ca(features_x)
			features_x = self.de(features_x, map) * features_x
			features_x = self.pool(features_x)  # 全局池化，形状 [batch_size, 128, 1, 1, 1]
			features_x = features_x.view(batch_size, -1)  # 变为 [batch_size, 128]
			fx = self.cam(Q, features_x, features_x)  # CAM 融合

		logits = self.classifier(fx)  # 分类
		return logits, fx
        
        
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x1 = torch.randn(2, 1, 121, 145, 121).to(device) 
x2 = torch.randn(2, 1, 105, 105).to(device) 
model2 = CNN3D_Net4l().to(device) 
output, feature = model2(x1)
#output = model(x1, x2)
yy = feature.shape
print(yy)
'''