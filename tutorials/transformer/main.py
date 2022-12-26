from model import *
import torch 

if __name__ == "__main__":
    mlp = MLP(3, 1, 3, 32)
    data = torch.randn(3, 1, 3)
    y = mlp(data)
    print(mlp)
    print(y.shape)