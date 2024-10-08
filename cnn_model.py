import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,SubsetRandomSampler,Dataset
import torchvision.transforms as T
from torchvision import datasets
from sklearn.model_selection import KFold
from PIL import Image
import os
import numpy as np
import pandas as pd
import benchMarks


# 自定义一个数据集类：
# 我们所定义的数据集是一个有10类动作的影像图片数据集，每个数据有三帧的图像
class ActionDataset(Dataset):
    def __init__(self, root_dir, labels,transform):
        """
        数据集类的构造器
        :param root_dir: 整个数据的路劲
        :param transform: 对数据进行处理的函数
        :param labels: 图片的标签
        """
        self.root_dir = root_dir
        self.transform = transform
        self.length = len(os.listdir(root_dir))
        self.labels = labels
    def __len__(self):
        return self.length*3 # 每个片段都包含了三帧
    def __getitem__(self, idx):
        """
        每次获取一个sample时会自动调用这个方法
        :param idx:sample的序号（start from 0），注意这个序号是按照图片的数量数的，所以下面对于folder ID和图片ID都进行了处理
        :return:
        """
        folder = idx // 3+1
        imidx = idx % 3+1
        folder = format(folder,'05d') # 填充至5位字符串，不够的位数用0填充
        imgname = str(imidx) + ".jpg"
        img_path = os.path.join(self.root_dir, folder,imgname) # 形成图片的路径
        img = Image.open(img_path) # 打开图片

        if self.transform:      # 如果要先对数据进行预处理，则经过transform函数，transform定义对图像预处理的方法
            image = self.transform(img)
        if len(self.labels)!=0:
            Label = self.labels[idx // 3] - 1  # 传入labels前记得将labels整平
            sample={'image':image,'img_path':img_path,'label':Label} # 字典的形式获取每一条训练数据
        else:
            sample={'image':image,'img_path':img_path}
        return sample


class CNNModel(nn.Module):
    """
    build an CNN model:
    """
    def __init__(self, parameters:dict):
        """
        通过传入的参数字典来灵活调整卷积层的参数
        :param parameters: {}
        """
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(parameters['conv1']['input_channel'], parameters['conv1']['output_channel'],
                               kernel_size=parameters['conv1']['kernel_size'], stride=parameters['conv1']['stride'])
        self.maxpool1 = nn.MaxPool2d(kernel_size=parameters['MaxPool']['kernel_size'], stride=parameters['MaxPool']['stride'])
        self.conv2 = nn.Conv2d(parameters['conv2']['input_channel'], parameters['conv2']['output_channel'],
                               kernel_size=parameters['conv2']['kernel_size'], stride=parameters['conv2']['stride'])
        self.maxpool2 = nn.MaxPool2d(kernel_size=parameters['MaxPool']['kernel_size'], stride=parameters['MaxPool']['stride'])
        self.linear = nn.Linear(in_features=parameters['linear']['input_size'],out_features=parameters['linear']['output_size'])


    def forward(self, x):
        x = nn.functional.relu(self.conv1(x), inplace=True) # 1*28*28 -> 16*26*26
        x = self.maxpool1(x) # 16*26*26 -> 16*13*13
        x = nn.functional.relu(self.conv2(x), inplace=True) # 16*13*13 -> 8*11*11
        x = self.maxpool2(x) # 8*11*11 -> 8*5*5
        x = torch.flatten(x, 1) # flatten all to 1D array
        x = self.linear(x)  # (200,)
        return x


def train(model, train_loader, optimizer, loss_func, epochs,device,batch_size=64,k_fold=3,print_every=200):
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
            print(f"Train Loss: {train_loss:.4f}")
            train_loss_epochs.append(train_loss)
            train_acc_epochs.append(train_acc)
            # 进行validation之前, 替换模型权重
            model_info = copy.deepcopy(model.state_dict())
            model_info['linear.weight'] = (change_weights(model_info.get('linear.weight'),device)).reshape(10,-1) # 从(dims,1) tensor变为(10,dims) tensor
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
    weights = (pd.read_csv("./weights/LTP.csv")).to_numpy()  # change to numpy array
    weights = torch.tensor(weights.reshape(-1,1).flatten()).to(device) # flatten to 1D array
    model_params = model_params.view(-1) # 展平原本权重
    indice_of_nearest = torch.argmin((model_params.unsqueeze(-1) - weights).abs(),dim=-1)
    weights = weights[indice_of_nearest]
    return weights

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
        preds = nn.Softmax(dim=1)(preds)
        preds = torch.argmax(preds, dim=1)
        predictions.append(preds)
        labels.append(label)
    return torch.cat(predictions,dim=-1).cpu().detach().numpy(),torch.cat(labels,dim=-1).cpu().detach().numpy()

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

if __name__=='__main__':
    torch.manual_seed(3407)
    # wrap to dataloader
    batch_size = 64
    # 定义转换:
    transform_train = T.Compose([
        T.ToTensor(),
        lambda x:add_gaussian_noise(x,mean=0.5,std=0.2)
    ])
    # load data
    image_dataset_train = datasets.MNIST('./MNIST_dataset', train=True, transform=transform_train, download=True)
    image_dataloader_train = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4) # num_workers: 调用多少进程来加载数据集,
    image_dataloader_test = datasets.MNIST('./MNIST_dataset', train=False, transform=T.ToTensor())
    image_dataloader_test = DataLoader(image_dataloader_test, batch_size=batch_size, shuffle=False, num_workers=4)
    # set model parameters
    parameters = {
        'conv1':{'input_channel':3,'output_channel':16,'kernel_size':3,'stride':1},
        'MaxPool':{'kernel_size':2,'stride':2}, # 池化窗口大小2, 每次滑动2个像素
        'conv2':{'input_channel':16,'output_channel':32,'kernel_size':3,'stride':1},
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNNModel(parameters).to(device)
    print(model)

    # #test the model
    # x = Variable(torch.randn(32,3,64,64).type(torch.cuda.FloatTensor))
    # x = x.to(device)
    # y = model(x)
    # print(y.size())
    # print(np.array_equal((np.array([32,10])),np.array(y.size())))

    # train this model
    # hyperparameters
    epoches = 30
    folds = 3
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    model,train_loss,train_acc,CV_loss,CV_acc = train(model,
                                                      train_loader=image_dataloader_train,loss_func=loss_func,
                                                      optimizer=optimizer,epochs=epoches,device=device,k_fold=folds)

    # 保存模型：
    torch.save(model.state_dict(), './results/cnn_model_with_noise.pth') # 保存整个模型
    loss_result_path = "./results/W4/cnn_model_loss_with_noise.jpg"
    acc_result_path = "./results/W4/cnn_model_accuracy_with_noise.jpg"
    confMatrix_result_path = "./results/W4/cnn_model_confusion_Matrix_with_noise.jpg"
    benchMarks.plot_performance_all_loss(epoches*3,train_loss,CV_loss,loss_result_path)
    benchMarks.plot_performance_all_acc(epoches*3,train_acc,CV_acc,acc_result_path)
    predictions,labels = predict_on_test(model, image_dataloader_test)
    print(
        f"accuracy of this model:{benchMarks.accuracy(predictions,labels):.2f}\n" +
        f"macro precision of this model:{benchMarks.precision(predictions,labels):.2f}\n" +
        f"macro recall of this model:{benchMarks.recall(predictions,labels):.2f}\n" +
        f"macro F1 score of this model:{benchMarks.F1(predictions,labels):.2f}\n"
    )
    benchMarks.plot_conf_matrix(predictions, labels, path=confMatrix_result_path)




