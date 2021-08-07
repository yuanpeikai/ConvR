import json
import torch
from MydataSet import *
import lib
import numpy as np
from model import *
from evaluation import ranking_and_hits
import torch.utils.data as Data

# 处理数据集
eneity_dict = {'no_eneity': 0}
rel_dict = {'no_relative': 0}

train_data = []
test_data = []
# 获取train中数据
with open("./data/FB15k-237/e1rel_to_e2_train.json", "r", encoding='utf8') as loader:
    for line in loader.readlines():
        data = json.loads(line)
        e1 = data['e1']
        rel = data['rel']
        e2_list = data['e2_multi1'].split()
        train_data.append((e1, rel, e2_list))

        if e1 not in eneity_dict.keys():
            eneity_dict[e1] = len(eneity_dict)
        if rel not in rel_dict.keys():
            rel_dict[rel] = len(rel_dict)
        for e2 in e2_list:
            if e2 not in eneity_dict.keys():
                eneity_dict[e2] = len(eneity_dict)

num_eneity = []
num_label = []
print("处理训练集数据")
e2_multi = torch.zeros(len(train_data), len(eneity_dict)).cuda()
i = 0
for data in train_data:

    e1, rel, e2_list = data
    num_e1 = eneity_dict[e1]
    num_rel = rel_dict[rel]
    num_e2_list = [eneity_dict[i] for i in e2_list]
    num_eneity.append((num_e1, num_rel))
    tensor_data = torch.LongTensor(num_eneity).cuda()
    for j in num_e2_list:
        e2_multi[i][j] = 1
    i = i + 1


# 训练集数据处理完成
train_data_batch = Data.DataLoader(dataset=My_Train_DataSet(tensor_data, e2_multi), batch_size=128, shuffle=True)
print("处理测试数据集")
# 获取test中数据
test_eneity = []
test_rel = []
test_multi1 = []
test_multi2 = []
with open("./data/FB15k-237/e1rel_to_e2_ranking_test.json", "r", encoding='utf8') as loader:
    for line in loader.readlines():
        data = json.loads(line)
        e1 = data['e1']

        rel = data['rel']
        e2 = data['e2']
        rel_eval = data['rel_eval']
        e2_multi1 = data['e2_multi1'].split()
        e2_multi2 = data['e2_multi2'].split()

        if e1 not in eneity_dict.keys():
            e1 = 0
        else:
            e1 = eneity_dict[e1]
        if e2 not in eneity_dict.keys():
            e2 = 0
        else:
            e2 = eneity_dict[e2]
        test_eneity.append((e1, e2))
        if rel not in rel_dict.keys():
            rel = 0
        else:
            rel = rel_dict[rel]
        if rel_eval not in rel_dict.keys():
            rel_eval = 0
        else:
            rel_eval = rel_dict[rel_eval]
        test_rel.append((rel, rel_eval))
        e2_list1 = []
        for i in e2_multi1:
            if i not in eneity_dict.keys():
                e2_list1.append(0)
            else:
                e2_list1.append(eneity_dict[i])
        test_multi1.append(e2_list1)
        e2_list2 = []
        for i in e2_multi2:
            if i not in eneity_dict.keys():
                e2_list2.append(0)
            else:
                e2_list2.append(eneity_dict[i])
        test_multi2.append(e2_list2)

test_eneity = torch.LongTensor(test_eneity).cuda()
test_rel = torch.LongTensor(test_rel).cuda()

max1 = 0
for i in test_multi1:
    if len(i) > max1:
        max1 = len(i)

max2 = 0
for i in test_multi2:
    if len(i) > max2:
        max2 = len(i)

test_multi_tensor1 = torch.zeros([len(test_multi1), max1]).cuda()
test_multi_tensor2 = torch.zeros([len(test_multi1), max2]).cuda()

for i, list in enumerate(test_multi1):
    for j, data in enumerate(list):
        test_multi_tensor1[i][j] = data

for i, list in enumerate(test_multi2):
    for j, data in enumerate(list):
        test_multi_tensor2[i][j] = data

test_multi_tensor1 = test_multi_tensor1.int()
test_multi_tensor2 = test_multi_tensor2.int()

# 测试集数据处理完成
test_data_batch = Data.DataLoader(
    dataset=My_test_DataSet(test_eneity, test_rel, test_multi_tensor1, test_multi_tensor2), batch_size=128,
    shuffle=True)
print("数据处理完成")

# 准备模型
model = ConvE(len(eneity_dict), len(rel_dict)).cuda()
loss_fn = torch.nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0)
# # 开始训练
for epoch in range(1000):
    model.train()
    loss_all = []
    for i, (tensor_data, e2_multi) in enumerate(train_data_batch):
        e1 = tensor_data[:, 0]
        rel = tensor_data[:, 1]
        e2_multi = e2_multi.float()
        e2_multi = ((1.0 - lib.label_smoothing) * e2_multi) + (1.0 / e2_multi.size(1))
        output = model(e1, rel)
        loss = loss_fn(output, e2_multi)
        loss_all.append(loss.cpu().item())
        opt.zero_grad()
        loss.backward()
        opt.step()

    print("第{}训练，训练集loss:{}".format(epoch + 1, np.mean(loss_all)))

    model.eval()
    with torch.no_grad():
        if epoch % 5 == 0 and epoch != 0:
            print("第{}训练的测试结果".format(epoch+1))
            ranking_and_hits(model, test_data_batch)
