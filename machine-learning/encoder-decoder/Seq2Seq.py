"""
Encoder-Decoder 架构概念演示脚本 (序列反转任务)
================================================

本脚本实现了一个最基础的“序列到序列”(Seq2Seq) 模型，用于演示深度学习中的
“理解-压缩-生成”思维框架。

核心任务:
    模型接收一个随机生成的浮点数序列 (如 [1, 2, 3])，目标是学会输出其反转序列 ([3, 2, 1])。

模型组件:
    1. Encoder (编码器): 使用单层 GRU，将变长输入压缩为固定维度的隐藏状态 (Context Vector)。
    2. Decoder (解码器): 从 Context Vector 出发，利用自回归 (Autoregressive) 机制逐步生成预测。
    3. Seq2Seq (容器): 协调编码与解码的逻辑流。

运行环境需求:
    - PyTorch
    - CUDA (可选，脚本会自动检测 NVIDIA GPU)

使用方法:
    直接运行脚本即可开始训练：
    $ uv run Seq2Seq.py

作者: octo (octo-concept-lab)
日期: 2026-03-12
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random


class Encoder(nn.Module):
    """
    编码器模块：负责将输入序列压缩为隐含状态（Context Vector）。

    Attributes:
        rnn (nn.Module): 内部使用的 GRU 单元。
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        初始化编码器。

        Args:
            input_dim: 每个时间步输入的特征维度（例如：1代表单个数值，512代表Embedding维度）。
            hidden_dim: 隐含层（Context Vector）的维度。
        """
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将输入序列编码为最后一个时序的隐藏状态。

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)。

        Returns:
            torch.Tensor: 最后时刻的隐藏状态，形状为 (1, batch_size, hidden_dim)。
                         这也就是我们所说的 Context Vector。
        """
        _, hidden = self.rnn(x)
        return hidden


class Decoder(nn.Module):
    """
    解码器模块：根据 Context Vector 逐步生成目标序列。

    Attributes:
        rnn (nn.Module): 内部使用的 GRU 单元。
        fc_out (nn.Linear): 将隐藏状态映射到输出维度的全连接层。
    """

    def __init__(self, output_dim: int, hidden_dim: int):
        """
        初始化解码器。

        Args:
            output_dim: 每个时间步输出的特征维度。
            hidden_dim: 隐含层的维度。
        """
        super().__init__()
        self.rnn = nn.GRU(output_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        """
        单步前向传播：基于当前输入和上一步的隐藏状态预测下一步。

        Args:
            x: 当前步的输入张量，形状为 (batch_size, 1, output_dim)。
            hidden: 上一步的隐藏状态，形状为 (1, batch_size, hidden_dim)。

        Returns:
            tuple: (prediction, hidden)，其中 prediction 是当前步预测值。
        """
        output, hidden = self.rnn(x, hidden)
        prediction = self.fc_out(output)
        return prediction, hidden


class Seq2Seq(nn.Module):
    """
    序列到序列模型（容器）：协调编码器和解码器的工作流。

    该模型体现了典型的“理解-压缩-生成”思维框架。
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        完整工作流：首先编码整个源序列，然后递归生成目标序列。

        Args:
            source: 源序列张量 (batch_size, seq_len, input_dim)。
            target_len: 期望生成的序列长度。

        Returns:
            torch.Tensor: 生成的目标序列 (batch_size, target_len, output_dim)。
        """
        # 1. 编码阶段 (Encoding phase)
        # 将整个序列压缩成一个 Context Vector
        context_vector = self.encoder(source)

        # 2. 解码阶段 (Decoding phase)
        outputs = []
        # 初始化第一个输入（通常为0向量或特定的起始标记）
        curr_input = torch.zeros(source.size(0), 1, 1).to(source.device)
        curr_hidden = context_vector

        for _ in range(target_len):
            # 这里的自回归逻辑：当前步的预测将成为下一步的输入
            prediction, curr_hidden = self.decoder(curr_input, curr_hidden)
            outputs.append(prediction)
            curr_input = prediction

        return torch.cat(outputs, dim=1)


# --- 2. 准备数据：序列反转任务 ---
def generate_data(batch_size, seq_len):
    # 随机生成 batch_size 组长度为 seq_len 的序列
    x = torch.randn(batch_size, seq_len, 1)
    y = torch.flip(x, dims=[1])  # 目标是反转序列
    return x, y


# --- 3. 运行配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_dim = 1
hidden_dim = 64
model = Seq2Seq(Encoder(input_dim, hidden_dim), Decoder(input_dim, hidden_dim)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# --- 4. 训练循环 ---
print("Starting training...")
for epoch in range(1001):
    src, trg = generate_data(32, 5)  # 序列长度为5
    src, trg = src.to(device), trg.to(device)

    optimizer.zero_grad()
    output = model(src, 5)
    loss = criterion(output, trg)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- 5. 测试效果 ---
model.eval()
with torch.no_grad():
    test_src, test_trg = generate_data(1, 5)
    test_src = test_src.to(device)
    prediction = model(test_src, 5)

    print("\n--- 测试结果 ---")
    print(f"输入序列: {test_src.cpu().flatten().tolist()}")
    print(f"目标序列: {test_trg.flatten().tolist()}")
    print(f"模型预测: {prediction.cpu().flatten().tolist()}")