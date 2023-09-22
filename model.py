import torch
import torch.nn as nn

class scaledDotProudct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        qm, km, vm = x
        qkm = torch.matmul(qm, km.T) / torch.sqrt(qm.shape[0])
        soft = nn.functional.softmax(torch.matmul(qkm, vm))
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
        qw = self.qLinear(x)
        kw = self.kLinear(x)
        vw = self.vLinear(x)

        mhaResult = scaledDotProudct((qw, kw, vw))
        return mhaResult


class feedForward(nn.Module):
    def __init__(self, vDim):
        super().__init__()
        self.linear = nn.Linear(vDim, vDim)
        self.layerNorm = nn.LayerNorm(vDim)

    def forward(self, x):
        return self.layerNorm(x + self.linear(x))

class encoder(nn.Module):
    def __init__(self):
        pass

class decoder(nn.Module):
    def __init__(self):
        pass