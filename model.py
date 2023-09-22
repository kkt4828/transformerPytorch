import torch
import torch.nn as nn

class ScaledDotProudct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        qm, km, vm = x
        qkm = torch.matmul(qm, km.T) / torch.sqrt(torch.tensor(qm.shape[0]))
        soft = nn.functional.softmax(torch.matmul(qkm, vm), dim = 1)
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
        mhaResult = torch.concat(mhaResult, dim = 1)
        return mhaResult


class FeedForward(nn.Module):
    def __init__(self, vDim):
        super().__init__()
        self.linear = nn.Linear(vDim, vDim)
        self.layerNorm = nn.LayerNorm(vDim)

    def forward(self, x):
        return self.layerNorm(x + self.linear(x))

class Encoder(nn.Module):
    def __init__(self, embDim, kDim, vDim, N = 4):
        super().__init__()
        self.embDim = embDim
        self.kDim = kDim
        self.vDim = vDim
        self.layerNorm = nn.LayerNorm(vDim)
        self.linear = nn.Linear(embDim, vDim)
        self.head = MultiHeadAttention(N, embDim, kDim, vDim)
        self.feedForwardNet = FeedForward(vDim)

    def forward(self, x):
        x = torch.add(self.linear(x), self.head(x))
        x = self.layerNorm(x)
        x = self.feedForwardNet(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        pass


if __name__ == "__main__":
    embDim = 256
    keyDim = 64
    valueDim = 128
    testInput = torch.randn(3, embDim)
    transformerEncoder = Encoder(embDim=embDim, kDim=keyDim, vDim=valueDim, N = 4)
    print(transformerEncoder(testInput).shape)
    # print(transformerEncoder(testInput))