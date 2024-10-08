import torch
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as T
import benchMarks
import json

torch.manual_seed(3407)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FCModel(nn.Module):
    def __init__(self,input_size,output_size,hidden_size=0):
        super(FCModel, self).__init__()
        self.hidden_size = hidden_size
        if hidden_size:
            self.fc1 = nn.Linear(input_size,hidden_size)
            self.fc2 = nn.Linear(hidden_size,output_size)
        else:
            self.fc = nn.Linear(input_size,output_size)

    def forward(self,x):
        # 合并除第一个维度外其它所有维度
        x = x.view(x.size(0),-1)
        if self.hidden_size:
            x = self.fc1(x)
            x = self.fc2(x)
        else:
            x = self.fc(x)
        return x

def train(model, train_loader, optimizer, loss_func, epochs,device,batch_size=64,k_fold=3,print_every=200):
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

                preds = model(datas)  # 前向传播

                loss = loss_func(preds, labels)  # 计算损失
                optimizer.zero_grad()  # 清除优化器梯度（来自于上一次反向传播）
                loss.backward()  # 反向传播, 计算模型参数梯度
                optimizer.step()  # 根据计算得到的梯度，使用优化器更新模型的参数。

                # 检查准确率
                preds = nn.functional.softmax(preds, dim=1)
                preds = torch.argmax(preds, dim=1)
                corrects += torch.sum(preds == labels).item() # item: 将torch张量转为对应的python基本数据类型

                loss_val += loss.item() * datas.size(0)  # 获取loss，并乘以当前批次大小

            train_loss = loss_val / len(train_sampler) # 计算整个模型的总损失
            train_acc = corrects / len(train_sampler) # 计算本次epoch的总正确率
            print(f"Train Loss: {train_loss:.4f}")
            train_loss_epochs.append(train_loss)
            train_acc_epochs.append(train_acc)
            # 进行validation之前, 替换模型权重
            model_info = copy.deepcopy(model.state_dict())
            original_size = model_info.get('fc.weight').shape
            model_info['fc.weight'] = (change_weights(model_info.get('fc.weight'),device)).reshape(original_size) # 转为原来的parameter size
            # original_size = model_info.get('fc2.weight').shape
            # model_info['fc2.weight'] = (change_weights(model_info.get('fc2.weight'),device)).reshape(original_size)
            model.load_state_dict(model_info)
            # 每个epoch都进行评估：
            val_loss,val_acc = validation(model, val_loader, loss_func, device,data_size=len(val_sampler))
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


def validation(model, val_loader, loss_func, device,data_size):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for datas,labels in val_loader:
        # datas: (batch_size,3,64,64)
        # labels: (batch_size,10)
        datas = datas.to(device)
        labels = labels.to(device)

        preds = model(datas)
        loss = loss_func(preds, labels.long())
        loss_val += loss.item() * datas.size(0)

        # 检查准确率
        preds = nn.functional.softmax(preds, dim=1)
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
    weights = (pd.read_csv("./weights/merge.csv")).to_numpy()  # change to numpy array
    weights = torch.tensor(weights.reshape(-1,1).flatten()).to(device) # flatten to 1D array
    model_params = model_params.view(-1) # 展平原本权重
    indice_of_nearest = torch.argmin((model_params.unsqueeze(-1) - weights).abs(),dim=-1)
    weights = weights[indice_of_nearest]
    return weights

def add_gaussian_noise(image,mean=0.0,std=0.1):
    """
    给图像image添加高斯噪声
    """
    # 确保std是非负数
    if std < 0:
        raise ValueError("Standard deviation must be non-negative")
    # 生成与图像形状相同的正态分布随机数
    noise = torch.randn(image.size()) * std + mean
    # 将噪声添加到图像上
    noisy_image = image + noise
    # 确保像素值在[0, 1]范围内
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image

def predict_on_test(model,test_loader):
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
        preds = model(data)
        preds = nn.functional.softmax(preds,dim=1)
        preds = torch.argmax(preds, dim=1)
        predictions.append(preds)
        labels.append(label)
    return torch.cat(predictions,dim=-1).cpu().detach().numpy(),torch.cat(labels,dim=-1).cpu().detach().numpy()

def init_prediction(model,test_loader,loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for data,label in test_loader:
        data = data.to(device)
        label = label.to(device)
        preds = model(data)
        # loss
        loss = loss_func(preds, label.long())
        loss_val += loss.item() * data.size(0)
        # accuracy
        preds = nn.functional.softmax(preds,dim=1)
        preds = torch.argmax(preds, dim=1)
        corrects += torch.sum(preds == label).item()  # item: 将torch张量转为对应的python基本数据类型
    return loss_val / len(test_loader.dataset),corrects/len(test_loader.dataset)

def save_arrays_in_csv(arrays,path):
    with open(path,"w",newline='') as f:
        np.savetxt(f,arrays,delimiter=",")
def main():
    # wrap to dataloader
    batch_size = 64
    # 定义转换:
    transform_train = T.Compose([
        T.ToTensor(),
        lambda x: add_gaussian_noise(x, mean=0, std=0.2)
    ])
    # load data
    image_dataset_train = datasets.MNIST('./MNIST_dataset', train=True, transform=transform_train, download=True)
    image_dataloader_train = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True)  # num_workers: 调用多少进程来加载数据集,
    image_dataloader_test = datasets.MNIST('./MNIST_dataset', train=False, transform=T.ToTensor())
    image_dataloader_test = DataLoader(image_dataloader_test, batch_size=batch_size, shuffle=False)
    # set parameters & model
    input_size = 784
    output_size = 10
    model = FCModel(input_size,output_size).to(device)
    print(model)
    # store initialized weights
    # parameters = copy.deepcopy(model.state_dict())
    # fc_weights1 = parameters['fc1.weight'].detach().cpu().numpy()
    # fc_weights2 = parameters['fc2.weight'].detach().cpu().numpy()
    # np.save('./results/W6/init_fc_weights_last_layer.npy', fc_weights2)
    # init_fc_weight = np.concatenate([fc_weights1.flatten(),fc_weights2.flatten()])
    # save_arrays_in_csv(init_fc_weight,'./results/W6/init_fc_weights.csv')
    epoches = 1
    folds = 2
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    # make an inference before training:
    init_loss,init_acc = init_prediction(model,image_dataloader_train,loss_func)
    print(f"initial loss {init_loss:.4f}; initial accuracy {init_acc:.4f}")
    model, train_loss, train_acc, CV_loss, CV_acc = train(model,
                                                          train_loader=image_dataloader_train, loss_func=loss_func,
                                                          optimizer=optimizer, epochs=epoches, device=device,
                                                          k_fold=folds)
    # # concat the data with one doing first time inference
    # train_loss = np.concatenate((np.array([init_loss]),train_loss))
    # train_acc = np.concatenate((np.array([init_acc]),train_acc))
    # CV_loss = np.concatenate((np.array([init_loss]),CV_loss))
    # CV_acc = np.concatenate((np.array([init_acc]),CV_acc))
    # # store train_loss
    # save_arrays_in_csv(train_loss,'./results/W6/fc_train_loss_merge.csv')
    # save_arrays_in_csv(train_acc,'./results/W6/fc_train_acc_merge.csv')
    # save_arrays_in_csv(CV_loss,'./results/W6/fc_test_loss_merge.csv')
    # save_arrays_in_csv(CV_acc,'./results/W6/fc_test_acc_merge.csv')
    # # np.save('./results/W6/fc_train_loss_merge.npy',train_loss)
    # # np.save('./results/W6/fc_train_acc_merge.npy',train_acc)
    # # np.save('./results/W6/fc_test_loss_merge.npy',CV_loss)
    # # np.save('./results/W6/fc_test_acc_merge.npy',CV_acc)
    #
    # parameters = copy.deepcopy(model.state_dict())
    # # store final fc weights
    # fc_weights1 = parameters['fc1.weight'].detach().cpu().numpy()
    # fc_weights2 = parameters['fc2.weight'].detach().cpu().numpy()
    # np.save('./results/W6/final_fc_weights_last_layer_merge.npy',fc_weights2)
    # fc_weight = np.concatenate([fc_weights1.flatten(), fc_weights2.flatten()])
    # save_arrays_in_csv(fc_weight, './results/W6/final_fc_weights.csv')
    predictions, labels = predict_on_test(model, image_dataloader_test)
    confMatrix_result_path = "./results/model_conf_Matrix_merge.jpg"
    # metrics = {}
    # metrics['accuracy'] = benchMarks.accuracy(predictions,labels)
    # metrics['precision'] = benchMarks.precision(predictions,labels)
    # metrics['recall'] = benchMarks.recall(predictions,labels)
    # metrics['F1'] = benchMarks.F1(predictions,labels)
    # with open('./results/W6/metrics_merge.json', 'w') as f:
    #     f.write(json.dumps(metrics,indent=4))
    benchMarks.plot_conf_matrix(predictions, labels, path=confMatrix_result_path)






if __name__ == '__main__':
    main()