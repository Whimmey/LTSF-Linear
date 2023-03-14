import torch.nn as nn
import torch.nn.functional as F

# class of multilayer perceptron By: Copilot
class MLP(nn.Module):
    """
    Multilayer perceptron with dropout and activation function
    Args:
        d_model: dimension of input
        d_ff: dimension of hidden layer
        d_out: dimension of output
        dropout: dropout rate
        activation: activation function
    """
    def __init__(self, d_model, d_ff, d_out, dropout=0.1, activation='relu'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff) # fc1为全连接层 nn.Linear两个参数分别为输入和输出的维度
        self.fc2 = nn.Linear(d_ff, d_out) # fc2为全连接层
        self.dropout = nn.Dropout(dropout) # dropout层
        self.activation = activation # 激活函数

    def forward(self, x):
        if self.activation == 'relu':
            x = F.relu(self.fc1(x))
        elif self.activation == 'gelu':
            x = F.gelu(self.fc1(x))
        elif self.activation == 'sigmoid':
            x = F.sigmoid(self.fc1(x))
        # elif self.activation == 'switch':
            # x = swish(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# example of MLP
# mlp = MLP(512, 2048, 512, dropout=0.1, activation='relu')
# x = torch.randn(1, 512)
# y = mlp(x)
# print(y.shape) 

