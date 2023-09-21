import torch
import torch.nn as nn

class scaledDotProudct(nn.Module):
    def __init__(self, embDim, kDim, vDim):
        super().__init__()
        self.embDim = embDim
        self.q = torch.nn.Parameter(torch.randn((embDim, kDim)))
        self.k = torch.nn.Parameter(torch.randn((embDim, kDim)))
        self.v = torch.nn.Parameter(torch.randn((embDim, vDim)))

    def forward(self, x):
        qm = torch.matmul(x, self.q)
        km = torch.matmul(x, self.k)
        vm = torch.matmul(x, self.v)
        qkm = torch.matmul(qm, km.T) / torch.sqrt(self.q.shape[0])
        soft = nn.functional.softmax(torch.matmul(qkm, vm))

class multiHeadAttetion(nn.Module):
    def __init__(self):
        super().__init__()

class feedForward(nn.Module):
    def __init__(self):
        super().__init__()