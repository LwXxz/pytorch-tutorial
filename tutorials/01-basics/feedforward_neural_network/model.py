import torch 
import torch.nn as nn

# conv neural network with two hidden layers
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim,  hidden_dim, output_dim) -> None:
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 7, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 5, 1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16*16*hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return self.fc(self.flatten(x))

if __name__ == "__main__":
    model = NeuralNetwork(1, 64, 10)
    input = torch.rand(1, 1, 28, 28)
    output = model(input)
    print(output.shape)