import torch
import numpy as np


def ranking_and_hits(model, test_data_batch):
    ranks = []
    ranks_left = []
    ranks_right = []
    hits_one = []
    hits_three = []
    hits_ten = []
    hits_left_one = []
    hits_left_three = []
    hits_left_ten = []
    hits_right_one = []
    hits_right_three = []
    hits_right_ten = []

    for index, (test_eneity, test_rel, test_multi_tensor1, test_multi_tensor2) in enumerate(test_data_batch):
        e1 = test_eneity[:, 0]
        e2 = test_eneity[:, 1]
        rel = test_rel[:, 0]
        rel_eval = test_rel[:, 1]
        e2_multi = test_multi_tensor1
        e1_multi = test_multi_tensor2
        pred1 = model(e1, rel)
        pred2 = model(e2, rel_eval)

        for i in range(e1.shape[0]):
            filter1 = e2_multi[i].long()
            filter2 = e1_multi[i].long()
            target_value1 = pred1[i, e2[i].item()].item()
            target_value2 = pred2[i, e1[i].item()].item()
            # 将e2集合所有对应元素置为0
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # 还原测试集中e2原值
            pred1[i, e2[i]] = target_value1
            pred2[i, e1[i]] = target_value2

        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)
        # 将元素对应位置的下标从大到小排序
        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()

        for i in range(e1.shape[0]):
            # 检查e2在argsort中的位置
            rank1 = np.where(argsort1[i] == e2[i].item())[0][0]
            rank2 = np.where(argsort2[i] == e1[i].item())[0][0]
            ranks.append(rank1 + 1)
            ranks_left.append(rank1 + 1)
            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)

            # @1命中率
            if rank1 + 1 <= 1:
                hits_one.append(1.0)
                hits_left_one.append(1.0)
            else:
                hits_one.append(0.0)
                hits_left_one.append(0.0)

            # @3命中率
            if rank1 + 1 <= 3:
                hits_three.append(1.0)
                hits_left_three.append(1.0)
            else:
                hits_three.append(0.0)
                hits_left_three.append(0.0)
            # @10命中率
            if rank1 + 1 <= 10:
                hits_ten.append(1.0)
                hits_left_ten.append(1.0)
            else:
                hits_ten.append(0.0)
                hits_left_ten.append(0.0)

            # @1命中率
            if rank2 + 1 <= 1:
                hits_one.append(1.0)
                hits_right_one.append(1.0)
            else:
                hits_one.append(0.0)
                hits_right_one.append(0.0)

            # @3命中率
            if rank2 + 1 <= 3:
                hits_three.append(1.0)
                hits_right_three.append(1.0)
            else:
                hits_three.append(0.0)
                hits_right_three.append(0.0)
            # @10命中率
            if rank2 + 1 <= 10:
                hits_ten.append(1.0)
                hits_right_ten.append(1.0)
            else:
                hits_ten.append(0.0)
                hits_right_ten.append(0.0)
    print("@1 总命中率：{}".format(np.mean(hits_one)))
    print("@1 left命中率：{}".format(np.mean(hits_left_one)))
    print("@1 right命中率：{}".format(np.mean(hits_right_one)))

    print("@3 总命中率：{}".format(np.mean(hits_three)))
    print("@3 left命中率：{}".format(np.mean(hits_left_three)))
    print("@3 right命中率：{}".format(np.mean(hits_right_three)))

    print("@10 总命中率：{}".format(np.mean(hits_ten)))
    print("@10 left命中率：{}".format(np.mean(hits_left_ten)))
    print("@10 right命中率：{}".format(np.mean(hits_right_ten)))

    print("MR_all:{}".format(np.mean(ranks)))
    print("MR_left:{}".format(np.mean(ranks_left)))
    print("MR_right:{}".format(np.mean(ranks_right)))

    print("MRR_all:{}".format(np.mean(1.0 / np.array(ranks))))
    print("MRR_left:{}".format(np.mean(1.0 / np.array(ranks_left))))
    print("MRR_right:{}".format(np.mean(1.0 / np.array(ranks_right))))
