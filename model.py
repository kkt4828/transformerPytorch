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
    def __init__(self, embDim, kDim, vDim, masking):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn((embDim, kDim)))
        self.k = torch.nn.Parameter(torch.randn((embDim, kDim)))
        self.v = torch.nn.Parameter(torch.randn((embDim, vDim)))
        self.masking = masking

        self.qLinear = nn.Linear(embDim, embDim)
        self.kLinear = nn.Linear(embDim, embDim)
        self.vLinear = nn.Linear(embDim, embDim)
        self.mhaLinear = nn.Linear(vDim, vDim)

    def forward(self, x):
        qw = self.qLinear(x) @ self.q
        kw = self.kLinear(x) @ self.k
        vw = self.vLinear(x) @ self.v

        mhaResult = ScaledDotProudct(masking=self.masking)((qw, kw, vw))
        mhaResult = self.mhaLinear(mhaResult)
        return mhaResult

class MultiHeadAttention(nn.Module):
    def __init__(self, N, embDim, keyDim, valueDim, masking = False):
        super().__init__()
        self.N = N
        self.masking = masking
        if valueDim % N != 0:
            raise Exception
        self.mha = nn.ModuleList([
            HeadAttetion(embDim=embDim, kDim=keyDim, vDim=(valueDim // N), masking = self.masking) for _ in range(N)
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

class DecoderHeadAttention(HeadAttetion):
    def __init__(self, embDim, kDim, vDim, masking, encoderOup):
        super().__init__(embDim, kDim, vDim, masking)
        self.encoderOup = encoderOup
    def forward(self, x):
        qw = self.qLinear(self.encoderOup) @ self.q
        kw = self.kLinear(self.encoderOup) @ self.k
        vw = self.vLinear(x) @ self.v

        mhaResult = ScaledDotProudct(masking=self.masking)((qw, kw, vw))
        mhaResult = self.mhaLinear(mhaResult)
        return mhaResult

class DecoderMha(MultiHeadAttention):
    def __init__(self, N, embDim, keyDim, valueDim, encoderOup, masking = False):
        super().__init__(N, embDim, keyDim, valueDim, masking)
        self.mha = nn.ModuleList([
            DecoderHeadAttention(embDim=embDim, kDim=keyDim, vDim=(valueDim // N), masking = self.masking, encoderOup=encoderOup) for _ in range(N)
        ])

    def forward(self, x):
        mhaResult = [self.mha[i](x) for i in range(self.N)]
        mhaResult = torch.concat(mhaResult, dim=-1)
        return mhaResult

class DecoderBlock(nn.Module):
    def __init__(self, embDim, kDim, hDim, vDim, N, encoderOup):
        super().__init__()
        self.maskHead = MultiHeadAttention(N, embDim, kDim, vDim, masking=True)
        self.head = DecoderMha(N, vDim, kDim, vDim, encoderOup)
        self.layerNorm1 = nn.LayerNorm(vDim)
        self.layerNorm2 = nn.LayerNorm(vDim)
        self.linear1 = nn.Linear(embDim, vDim)
        self.linear2 = nn.Linear(vDim, vDim)
        self.feedForwardNet = FeedForward(vDim, hDim)

    def forward(self, x):
        x = torch.add(self.linear1(x), self.maskHead(x))
        x = self.layerNorm1(x)
        x = torch.add(self.linear2(x), self.head(x))
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
    def __init__(self, embDim, kDim, hDim, vDim, encoderOup, N = 8, blockN = 6):
        super().__init__()
        decoderBlockList = [DecoderBlock(embDim, kDim, hDim, vDim, N, encoderOup)]
        decoderBlockList += [DecoderBlock(vDim, kDim, hDim, vDim, N, encoderOup) for _ in range(blockN - 1)]
        self.decoderBlocks = nn.Sequential(
            *decoderBlockList
        )
    def forward(self, x):
        x = self.decoderBlocks(x)
        return x



if __name__ == "__main__":
    embDim = 512
    keyDim = 512
    valueDim = 512
    testInput = torch.randn(3, 4, embDim)
    testInput2 = torch.randn(3, 4, embDim)
    transformerEncoder = Encoder(embDim=embDim, kDim=keyDim, hDim = 2048, vDim=valueDim, N = 8)
    encoderOup = transformerEncoder(testInput)
    print(encoderOup.shape)
    transformerDecoder = Decoder(embDim=embDim, kDim=keyDim, hDim=2048, vDim=valueDim, encoderOup=encoderOup, N=8)
    print(transformerDecoder(testInput2).shape)
    # print(transformerEncoder(testInput))