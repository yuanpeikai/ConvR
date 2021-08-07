import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.weight = nn.Parameter(torch.randn(10,16, 1, 5, 5))  # 自定义的权值
        self.bias = nn.Parameter(torch.randn(16))  # 自定义的偏置

    def forward(self, x):

        # out = F.conv2d(x[0].unsqueeze(0), self.weight,stride=1,padding=0)
        out_all=torch.zeros([10,16,6,6])
        for i in range(x.shape[0]):
            out = F.conv2d(x[i].unsqueeze(0), self.weight[i], stride=1, padding=0)
            out_all[i]=out
        return out_all

model=CNN()
x=torch.ones([10,1,10,10])
y=model(x)
print(y)
print(y.size())

list=[1,2,3,4]
a=torch.LongTensor(list)
b=torch.nn.Embedding(6,6)
c=b(a)
print(c)
print(c.size())
print(b.weight.size())
