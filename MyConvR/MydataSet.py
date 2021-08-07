import torch
import torch.utils.data as Data


class My_Train_DataSet(Data.Dataset):
    def __init__(self, tensor_data, e2_multi):
        super(My_Train_DataSet, self).__init__()
        self.tensor_data = tensor_data
        self.e2_multi = e2_multi

    def __len__(self):
        return self.tensor_data.shape[0]

    def __getitem__(self, idx):
        return self.tensor_data[idx], self.e2_multi[idx]


class My_test_DataSet(Data.Dataset):
    def __init__(self, test_eneity, test_rel, test_multi1, test_multi2):
        super(My_test_DataSet, self).__init__()
        self.test_eneity = test_eneity
        self.test_rel = test_rel
        self.test_multi1 = test_multi1
        self.test_multi2 = test_multi2

    def __len__(self):
        return self.test_eneity.shape[0]

    def __getitem__(self, idx):
        return self.test_eneity[idx], self.test_rel[idx], self.test_multi1[idx], self.test_multi2[idx],
