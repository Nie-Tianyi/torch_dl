import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建一个足够长的 PE 矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算公式中的分母部分：10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度，变成 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer(
            "pe", pe
        )  # register_buffer 保证该变量不被视为模型参数进行更新

    def forward(self, x):
        # 将位置编码与词嵌入相加
        # x 维度: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 第一层升维，第二层降维回 d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# 编码器的一层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. 自注意力 + 残差连接 + LayerNorm
        # 注意：这里采用了 Pre-Norm 结构，即先 Norm 再进 Sublayer，这在现代模型中更常用且稳定
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, mask)[0])
        # 2. 前馈网络 + 残差连接 + LayerNorm
        x2 = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x


# 解码器的一层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn  # 处理解码器内部已生成的词
        self.src_attn = src_attn  # 交叉注意力：Q来自解码器，K,V来自编码器
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 1. 遮罩自注意力 (Masked Self-Attention)
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, tgt_mask)[0])
        # 2. 编码器-解码器交叉注意力 (Cross-Attention)
        x2 = self.norm2(x)
        x = x + self.dropout(self.src_attn(x2, memory, memory, src_mask)[0])
        # 3. 前馈网络
        x2 = self.norm3(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 定义 WQ, WK, WV 权重矩阵
        # 这里直接定义三个全连接层，一次性投影出所有头的信息
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 最后输出的投影矩阵 WO
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # batch_size, seq_len, d_model
        batch_size = q.size(0)

        # 1. 线性变换得到 Q, K, V
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # 2. “拆分”成多头 (Split into heads)
        # 维度变换: (batch, seq, d_model) -> (batch, seq, num_heads, d_k) -> (batch, num_heads, seq, d_k)
        # 转置是为了让 batch 和 head 在前，方便后续进行并行矩阵运算
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 计算 Scaled Dot-Product Attention
        # scores 维度: (batch, num_heads, seq_q, seq_k)
        # 这里 K.transpose(-2, -1) 是为了让最后两维从 (seq, d_k) 变成 (d_k, seq) 进行点积
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 如果有 mask（比如 Decoder 中的顺序掩码），将对应位置设为极小值
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax 归一化得到权重
        attn_weights = F.softmax(scores, dim=-1)

        # 4. 加权求和得到 context
        # 维度: (batch, num_heads, seq, d_k)
        context = torch.matmul(attn_weights, V)

        # 5. 合并多头 (Concatenate)
        # 先转置回 (batch, seq, num_heads, d_k)，再合并最后两维
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # 6. 最后的线性输出
        output = self.W_o(context)

        return output, attn_weights


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
    ):
        super().__init__()
        # 1. 词向量层
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # 2. 编码器堆叠 (Deep Copying N layers)
        from copy import deepcopy

        self.mha = MultiHeadAttention(d_model, n_heads)  # 假设已定义之前的 MHA 代码
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, deepcopy(self.mha), deepcopy(self.ffn), dropout)
                for _ in range(n_layers)
            ]
        )

        # 3. 解码器堆叠
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    deepcopy(self.mha),
                    deepcopy(self.mha),
                    deepcopy(self.ffn),
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # 4. 输出投影层
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def encode(self, src, src_mask):
        x = self.pos_encoding(self.src_embed(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.pos_encoding(self.tgt_embed(tgt))
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 编码阶段
        memory = self.encode(src, src_mask)
        # 解码阶段
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        # 映射到词表
        return F.log_softmax(self.fc_out(output), dim=-1)
