import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_

import lib


class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, lib.eneity_dim, padding_idx=0)  # # batch_size,100
        self.emb_rel = torch.nn.Embedding(num_relations, lib.rel_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(lib.input_drop)  # 0.3
        self.feature_map_drop = torch.nn.Dropout2d(lib.feat_drop)  # 0.2
        self.hidden_drop = torch.nn.Dropout(lib.hidden_drop)  # 0.3

        self.emb_dim1 = lib.embedding_shape1  # 10
        self.emb_dim2 = lib.eneity_dim // self.emb_dim1  # 100//10=10

        # self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=lib.use_bias)  # true
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(100)
        self.bn_rel=torch.nn.BatchNorm1d(2500)
        self.bn2 = torch.nn.BatchNorm1d(lib.eneity_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))  # num_entities
        self.fc = torch.nn.Linear(100 * 6 * 6, lib.eneity_dim)
        self.init()
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)  # batch_size,1,10,10
        rel_embedded = self.emb_rel(rel)  # batch_size,2500
        rel_embedded=self.bn_rel(rel_embedded)
        e1 = self.bn0(e1_embedded)

        e1 = self.inp_drop(e1)
        rel_embedded = self.inp_drop(rel_embedded)


        filters = rel_embedded.view(-1,100, 1, 5, 5) #batch_size_100,1,5,5


        x=torch.zeros([e1.shape[0],100,6,6]).cuda()

        for i in range(e1.shape[0]):
            xi = F.conv2d(e1[i].unsqueeze(0), filters[i], stride=1, padding=0)
            x[i]=xi# batch_size,100,6,6

        # x = self.conv1(x)#batch_size,32,38,8  #他的卷积核是[32, 1, 3, 3] bias:[32]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)  # batch_size,100*6*6
        x = self.fc(x)  # batch_size,100
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)  # batch_size,100
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))  # batch_size,num_entities
        x += self.b.expand_as(x)  # batch_size,num_entities
        pred = torch.sigmoid(x)  # batch_size,num_entities

        return pred


if __name__ == '__main__':
    model=ConvE(13421,225)

    model()