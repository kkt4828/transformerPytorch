import torch
import torch.nn as nn
import numpy as np

class ScaledDotProudct(nn.Module):
    def __init__(self, masking = False):
        super().__init__()
        self.masking = masking
    def forward(self, x):
        qm, km, vm = x
        qkm = torch.matmul(qm, torch.transpose(km, 1, 2)) / torch.sqrt(torch.tensor(qm.shape[-1]))
        if self.masking:
            maskSize = qkm.shape[-1]
            mask = torch.tensor([[0 if j <= i else -np.inf for j in range(maskSize)] for i in range(maskSize)])
            qkm = torch.add(qkm, mask)
        soft = nn.functional.softmax(torch.matmul(qkm, vm), dim = -1)
        return soft

class HeadAttetion(nn.Module):
    def __init__(self, embDim, kDim, vDim):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn((embDim, kDim)))
        self.k = torch.nn.Parameter(torch.randn((embDim, kDim)))
        self.v = torch.nn.Parameter(torch.randn((embDim, vDim)))
        self.qLinear = nn.Linear(embDim, embDim)
        self.kLinear = nn.Linear(embDim, embDim)
        self.vLinear = nn.Linear(embDim, embDim)
        self.mhaLinear = nn.Linear(vDim, vDim)

    def forward(self, x):
        qw = self.qLinear(x) @ self.q
        kw = self.kLinear(x) @ self.k
        vw = self.vLinear(x) @ self.v

        mhaResult = ScaledDotProudct()((qw, kw, vw))
        mhaResult = self.mhaLinear(mhaResult)
        return mhaResult

class MultiHeadAttention(nn.Module):
    def __init__(self, N, embDim, keyDim, valueDim):
        super().__init__()
        self.N = N
        if valueDim % N != 0:
            raise Exception
        self.mha = nn.ModuleList([
            HeadAttetion(embDim=embDim, kDim=keyDim, vDim=(valueDim // N)) for _ in range(N)
        ])

    def forward(self, x):
        mhaResult = [self.mha[i](x) for i in range(self.N)]
        mhaResult = torch.concat(mhaResult, dim = -1)
        return mhaResult


class FeedForward(nn.Module):
    def __init__(self, vDim, hDim):
        super().__init__()
        self.linear1 = nn.Linear(vDim, hDim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hDim, vDim)
        self.layerNorm = nn.LayerNorm(vDim)

    def forward(self, x):
        return self.layerNorm(x + self.linear2(self.relu(self.linear1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, embDim, kDim, hDim, vDim, N):
        super().__init__()
        self.layerNorm = nn.LayerNorm(vDim)
        self.linear = nn.Linear(embDim, vDim)
        self.head = MultiHeadAttention(N, embDim, kDim, vDim)
        self.feedForwardNet = FeedForward(vDim, hDim)

    def forward(self, x):
        x = torch.add(self.linear(x), self.head(x))
        x = self.layerNorm(x)
        x = self.feedForwardNet(x)
        return x

class Encoder(nn.Module):
    def __init__(self, embDim, kDim, hDim, vDim, N = 8, blockN = 6):
        super().__init__()
        encoderBlockList = [EncoderBlock(embDim, kDim, hDim, vDim, N)]
        encoderBlockList += [EncoderBlock(vDim, kDim, hDim, vDim, N) for _ in range(blockN - 1)]
        self.encoderBlocks = nn.Sequential(
            *encoderBlockList
        )

    def forward(self, x):
        x = self.encoderBlocks(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        pass


if __name__ == "__main__":
    embDim = 512
    keyDim = 512
    valueDim = 512
    testInput = torch.randn(3, 4, embDim)
    transformerEncoder = Encoder(embDim=embDim, kDim=keyDim, hDim = 2048, vDim=valueDim, N = 8)
    print(transformerEncoder(testInput).shape)
    # print(transformerEncoder(testInput))