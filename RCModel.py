import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,SubsetRandomSampler
import reservoirpy as rpy
import pandas as pd
from reservoirpy.nodes import Reservoir, Input
from sklearn.model_selection import KFold
import benchMarks

rpy.verbosity(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(3407)
# 建立模型：单层reservoir
class RCModel(nn.Module):
    def __init__(self, res_unit, ler, sr):
        super(RCModel, self).__init__()
        self.res_unit = res_unit
        self.RCmodel = self.create_RC_model(res_unit, ler, sr)
        self.readout = nn.Linear(res_unit, 10)

    def create_RC_model(self, res_unit, ler, sr):
        data = Input()
        reservoir = Reservoir(res_unit, lr=ler, sr=sr, name='res1-1')
        model = data >> reservoir
        return model

    def forward(self, x):
        """
        x: input data with shape: [batch_size,channel, time_step(frame_size), feature_size]
        """
        # running reservoir and print output
        batch_size = x.shape[0]  # 获取当前批次数
        data = x.squeeze(1).cpu().detach().numpy()
        # 这里每一个sample只保留最后一个时间步的state
        curr_state = self.RCmodel.run(data)  # 返回每个时间步的step, 是一个列表（len=batchsize）, 列表的元素是np.array(shape=(timestep,units))
        if type(curr_state) is not list: # batch size=1, 还要重新包装成list
            curr_state = [curr_state]
        curr_states = torch.empty((batch_size, self.res_unit)).to(device)
        for i in range(batch_size):
            curr_states[i] = torch.tensor(curr_state[i][-1], dtype=torch.float32)
        x = self.readout(curr_states)
        return x
def train(model, train_loader, optimizer, loss_func, epochs,device,step=4,batch_size=64,k_fold=3):
    import copy
    """
    define training process of upon model, 使用K折验证
    return:
    model: the model after training
    train_losses: loss for each epoch in training(最优一折的数据)
    CV_losses: loss for each epoch in validation(最优一折的数据)
    """
    best_val_acc_list = [0.0 for i in range(k_fold)]  # 跟踪每个折最佳的损失
    best_models = []  # 记录每一折训练得到的最优model，然后根据正确率得到一个全局最优model
    best_model_params = copy.deepcopy(model.state_dict())  # 创建当前模型参数的深拷贝
    kf = KFold(n_splits=k_fold) # K折交叉验证
    # 记录每次验证的损失
    train_loss_folds = []
    CV_loss_folds = []
    # 记录每次验证的正确率
    train_acc_folds = []
    CV_acc_folds = []

    for fold, (train_indexes, val_indexes) in enumerate(kf.split(train_loader.dataset)): # enumerate: 将loader中的每个sample和索引配对
        train_sampler = SubsetRandomSampler(train_indexes)  # 告诉dataloader应该加载与len(train_indexes)数量相同，与train_indexes对应的样本
        val_sampler = SubsetRandomSampler(val_indexes)
        curr_train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=val_sampler)
        # 记录训练损失和正确率，用于画图：
        train_loss_epochs = []
        train_acc_epochs = []
        # 记录每次验证的损失和正确率，用于画图：
        CV_loss_epochs = []
        CV_acc_epochs = []

        for epoch in range(epochs):
            print(f" fold{fold+1}, epoch{epoch + 1}:...")
            model.train()  # 设置为训练模式
            loss_val = 0.0
            corrects = 0.0
            for datas,labels in curr_train_loader:
                # datas: (batch_size,1,28,28)
                # labels: (batch_size,10)
                datas = datas.to(device)
                labels = labels.to(device)
                datas = torch.cat((torch.split(datas,step,dim=-1)), dim=2) # 将28x28 image重组为196x4 image
                datas = torch.transpose(datas,2,3)
                preds = model(datas)  # 前向传播

                loss = loss_func(preds, labels)  # 计算损失
                optimizer.zero_grad()  # 清除优化器梯度（来自于上一次反向传播）
                loss.backward()  # 反向传播, 计算模型参数梯度
                optimizer.step()  # 根据计算得到的梯度，使用优化器更新模型的参数。

                # 检查准确率
                preds = nn.Softmax(dim=1)(preds)
                preds = torch.argmax(preds, dim=1)
                corrects += torch.sum(preds == labels).item() # item: 将torch张量转为对应的python基本数据类型

                loss_val += loss.item() * datas.size(0)  # 获取loss，并乘以当前批次大小

            train_loss = loss_val / len(train_sampler) # 计算整个模型的总损失
            train_acc = corrects / len(train_sampler) # 计算本次epoch的总正确率
            print(f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.4f}")
            train_loss_epochs.append(train_loss)
            train_acc_epochs.append(train_acc)
            # 进行validation之前, 替换模型权重
            model_info = copy.deepcopy(model.state_dict())
            model_info['readout.weight'] = (change_weights(model_info.get('readout.weight'),device)).reshape(10,-1) # 从(dims,1) tensor变为(10,dims) tensor
            model.load_state_dict(model_info)
            # 每个epoch都进行评估：
            val_loss,val_acc = validation(model, val_loader, loss_func, device,data_size=len(val_sampler),step=step)
            if (best_val_acc_list[fold] < val_acc):  # 出现最优模型时(损失最小的模型)，保存最优模型
                best_val_acc_list[fold] = val_acc
                best_model_params = copy.deepcopy(model.state_dict())
            # 更新平均loss指标
            CV_loss_epochs.append(val_loss)
            CV_acc_epochs.append(val_acc)


        # 更新每个fold的模型和训练及测试的loss记录
        model.load_state_dict(best_model_params)
        best_models.append(model)
        train_loss_folds.append(np.array(train_loss_epochs))
        CV_loss_folds.append(np.array(CV_loss_epochs))
        train_acc_folds.append(np.array(train_acc_epochs))
        CV_acc_folds.append(np.array(CV_acc_epochs))

    best_val_acc_index = torch.argmax(torch.tensor(best_val_acc_list))
    print(best_val_acc_index)
    model = best_models[best_val_acc_index]
    train_losses = np.concatenate(train_loss_folds)
    train_accs = np.concatenate(train_acc_folds)
    CV_losses = np.concatenate(CV_loss_folds)
    CV_accs = np.concatenate(CV_acc_folds)
    return model, train_losses, train_accs,CV_losses,CV_accs


def validation(model, val_loader, loss_func, device,data_size,step=4):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for datas,labels in val_loader:
        # datas: (batch_size,3,64,64)
        # labels: (batch_size,10)
        datas = datas.to(device)
        labels = labels.to(device)
        datas = torch.cat((torch.split(datas, step, dim=-1)), dim=2)  # 将28x28 image重组为196x4 image
        datas = torch.transpose(datas, 2, 3)
        preds = model(datas)
        loss = loss_func(preds, labels.long())
        loss_val += loss.item() * datas.size(0)

        # 检查准确率
        preds = nn.Softmax(dim=1)(preds)
        preds = torch.argmax(preds, dim=1)
        corrects += torch.sum(preds == labels).item()  # item: 将torch张量转为对应的python基本数据类型

    validation_loss = loss_val / data_size  # 计算整个测试集的总损失
    validation_acc = corrects / data_size
    print(f"validation Loss: {validation_loss:.4f}; validation Accuracy: {validation_acc:.4f}")
    return validation_loss,validation_acc

def change_weights(model_params,device):
    """
    按照器件的反馈值改变权重，每个epoch训练时改变
    :param model_params
    :return: replaced_weights
    """
    weights = (pd.read_csv(".\\weights\\LTP.csv")).to_numpy()  # change to numpy array
    weights = torch.tensor(weights.reshape(-1,1).flatten()).to(device) # flatten to 1D array
    model_params = model_params.view(-1) # 展平原本权重
    indice_of_nearest = torch.argmin((model_params.unsqueeze(-1) - weights).abs(),dim=-1)
    weights = weights[indice_of_nearest]

    return weights

def predict_on_test(model,test_loader,step=4):
    """
    return predicted values of test_loader, written in ndarray
    :param model:
    :param test_loader:
    :return:
    """
    model.eval()
    predictions = []
    labels = []
    for data,label in test_loader:
        data = data.to(device)
        label = label.to(device)
        data = torch.cat((torch.split(data, step, dim=-1)), dim=2)  # 将28x28 image重组为196x4 image
        data = torch.transpose(data, 2, 3)
        preds = model(data)
        preds = nn.Softmax(dim=1)(preds)
        preds = torch.argmax(preds, dim=1)
        predictions.append(preds)
        labels.append(label)
    return torch.cat(predictions,dim=-1).cpu().detach().numpy(),torch.cat(labels,dim=-1).cpu().detach().numpy()

def main():
    # load data
    image_dataset_train = datasets.MNIST('.\\MNIST_dataset', train=True, transform=transforms.ToTensor(), download=True)

    # wrap to dataloader
    batch_size = 64
    image_dataloader_train = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True,
                                        num_workers=4)  # num_workers: 调用多少进程来加载数据集,
    image_dataset_test = datasets.MNIST('.\\MNIST_dataset', train=False, transform=transforms.ToTensor())
    image_dataloader_test = DataLoader(image_dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型建立
    # some hyperparameters:
    epoches = 20
    folds = 3
    steps = 4 # length of each split image

    ler = 0.5  # leaky rate
    sr = 0.9  # spectual radius
    lr = 1e-3  # learning rate
    res_unit = 196  # neuron number in reservoir(14x14)

    model = RCModel(res_unit, ler, sr)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()

    model, train_loss, train_acc, CV_loss, CV_acc = train(model,
                                                          train_loader=image_dataloader_train, loss_func=loss_func,
                                                          optimizer=optimizer, step=steps,epochs=epoches, device=device,
                                                          k_fold=folds)

    loss_result_path = ".\\results\\RC_model_loss.jpg"
    acc_result_path = ".\\results\\RC_model_accuracy.jpg"
    confMatrix_result_path = ".\\results\\RC_model_confusion_Matrix.jpg"
    benchMarks.plot_performance_all_loss(epoches * folds, train_loss, CV_loss, loss_result_path)
    benchMarks.plot_performance_all_acc(epoches * folds, train_acc, CV_acc, acc_result_path)
    predictions, labels = predict_on_test(model, image_dataloader_test,steps)
    print(
        f"macro accuracy:{benchMarks.accuracy(predictions,labels):.2f}\n"+
        f"macro precision:{benchMarks.precision(predictions,labels):.2f}\n"+
        f"macro recall:{benchMarks.recall(predictions,labels):.2f}\n"+
        f"macro F1:{benchMarks.F1(predictions,labels):.2f}\n"
    )
    benchMarks.plot_conf_matrix(predictions, labels, path=confMatrix_result_path)

if __name__=='__main__':
    main()
