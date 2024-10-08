import numpy as np
from scipy import io
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader,SubsetRandomSampler,Dataset
import reservoirpy as rpy
import pandas as pd
from reservoirpy.nodes import Reservoir, Input
from sklearn.model_selection import KFold
from PIL import Image
import benchMarks

"""
an interesting try: use both CNN and RC
CNN is applied to extract image features, where RC is applied to make time seres analysis
just like CNN + LSTM
UCF101 dataset is applied, but the result is not that good(only 45% accuracy)
"""
rpy.verbosity(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(3407)

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
        self.labels = labels.flatten()
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        """
        每次获取一个sample时会自动调用这个方法
        :param idx:sample的序号（start from 0），注意这个序号是按照图片的数量数的，所以下面对于folder ID和图片ID都进行了处理
        :return:
        """
        folder = format(idx+1,'05d') # 填充至5位字符串，不够的位数用0填充
        sample = {}
        for i in range(3):
            imgname = str(i+1) + ".jpg"
            img_path = os.path.join(self.root_dir, folder,imgname) # 形成图片的路径
            img = Image.open(img_path) # 打开图片
            if self.transform:      # 如果要先对数据进行预处理，则经过transform函数，transform定义对图像预处理的方法
                img = self.transform(img)
            sample[f"image{i+1}"] = img.to(device)

        if len(self.labels)!=0:
            Label = self.labels[idx] - 1
            sample['label'] = Label
        return sample
class CNNModel(nn.Module):
    """
    build an CNN model:
    """
    def __init__(self, cnn_parameters:dict):
        """
        通过传入的参数字典来灵活调整卷积层的参数
        :param cnn_parameters: {}
        """
        super(CNNModel, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(cnn_parameters['conv1']['input_channel'], cnn_parameters['conv1']['output_channel'],
                               kernel_size=cnn_parameters['conv1']['kernel_size'], stride=cnn_parameters['conv1']['stride'])
        self.maxpool1 = nn.MaxPool2d(kernel_size=cnn_parameters['MaxPool']['kernel_size'], stride=cnn_parameters['MaxPool']['stride'])
        self.conv2 = nn.Conv2d(cnn_parameters['conv2']['input_channel'], cnn_parameters['conv2']['output_channel'],
                               kernel_size=cnn_parameters['conv2']['kernel_size'], stride=cnn_parameters['conv2']['stride'])
        self.maxpool2 = nn.MaxPool2d(kernel_size=cnn_parameters['MaxPool']['kernel_size'], stride=cnn_parameters['MaxPool']['stride'])


    def forward(self, x):
        """
        output_dim c
        """
        x = nn.functional.relu(self.conv1(x), inplace=True) # 1*28*28 -> 8*26*26
        x = self.maxpool1(x) # 8*26*26 -> 8*13*13
        x = nn.functional.relu(self.conv2(x), inplace=True) # 8*13*13 -> 16*11*11
        x = self.maxpool2(x) # 16*11*11 -> 16*5*5
        x = x.view(x.size(0), -1)
        return x

class RCmodel(nn.Module):
    def __init__(self,res_unit,time_step,ler,sr,bi_enable=True):
        super(RCmodel,self).__init__()
        self.res_unit = res_unit
        self.RC = self.create_RC_model(res_unit,ler,sr)
        self.bi_enable = bi_enable
        self.time_step = time_step
        input_size= res_unit * 2 if bi_enable else res_unit
        self.readout = nn.Linear(input_size, 10)

    def create_RC_model(self, res_unit, ler, sr):
        data = Input()
        reservoir = Reservoir(res_unit, lr=ler, sr=sr, name='res1-1')
        model = data >> reservoir
        return model

    def forward(self, input,batch_size):
        # 这里每一个sample只保留最后一个时间步的state
        input = input.detach().cpu().numpy()
        if self.bi_enable:
            curr_state_fwd = self.RC.run(input)
            curr_state_fwd = [curr_state_fwd] if type(curr_state_fwd) is not list else curr_state_fwd
            permutation = [self.time_step-i-1 for i in range(self.time_step)]
            curr_state_bwd = self.RC.run(input[:,permutation,:])
            curr_state_bwd = [curr_state_bwd] if type(curr_state_bwd) is not list else curr_state_bwd
            curr_states = torch.empty((batch_size, self.res_unit * 2)).to(device)
            for i in range(batch_size):
                curr_states[i] = torch.cat(
                    [torch.tensor(curr_state_fwd[i][-1], dtype=torch.float32),
                     torch.tensor(curr_state_bwd[i][-1],dtype=torch.float32)]
                ,dim=-1)  # 取最后一个时间步状态,沿行合并
            x = self.readout(curr_states)
            return x
        else:
            curr_state = self.RC.run(input)  # 返回每个时间步的step, 是一个列表（len=batchsize）, 列表的元素是np.array(shape=(timestep,units))
            if type(curr_state) is not list:  # batch size=1, 还要重新包装成list
                curr_state = [curr_state]
            curr_states = torch.empty((batch_size, self.res_unit)).to(device)
            for i in range(batch_size):
                curr_states[i] = torch.tensor(curr_state[i][-1], dtype=torch.float32) # 取最后一个时间步状态
            x = self.readout(curr_states)
            return x
class LSTMmodel(nn.Module):
    def __init__(self,input_size,hidden_size,bi_enable=False):
        super(LSTMmodel,self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True,bidirectional=bi_enable)
        self.linear = nn.Linear(hidden_size, 10)
    def forward(self,input):
        #input = torch.transpose(input, 0, 1)
        out,hidden = self.lstm(input,None)
        out = self.linear(out[:,-1,:]) # 只要最后一个时间步的输出
        return out

class Model(nn.Module):
    def __init__(self, cnn_model,rc_model=None,lstm_model=None,dropout_prob=0.1):
        super(Model, self).__init__()
        self.cnn = cnn_model
        self.rc_model = rc_model
        self.lstm_model = lstm_model
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input):
        image1 = self.cnn(input['image1'])
        image2 = self.cnn(input['image2'])
        image3 = self.cnn(input['image3'])
        image_stack = torch.stack((image1, image2, image3), dim=1)  # 在第二个维度上堆叠，最后shape=(64,3,feature_size)
        image_stack = self.dropout(image_stack)
        if self.rc_model is not None:
            batch_size = input['image1'].shape[0]
            result = self.rc_model(image_stack,batch_size)
            return result
        elif self.lstm_model is not None:
            return self.lstm_model(image_stack)




def train(model, train_loader, optimizer, loss_func, epochs,device,batch_size=64,k_fold=3):
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
            for sample in curr_train_loader:
                # datas: (batch_size,1,28,28)
                # labels: (batch_size,10)
                labels = sample['label'].long().to(device)
                preds = model(sample)  # 前向传播

                loss = loss_func(preds, labels)  # 计算损失
                optimizer.zero_grad()  # 清除优化器梯度（来自于上一次反向传播）
                loss.backward()  # 反向传播, 计算模型参数梯度
                optimizer.step()  # 根据计算得到的梯度，使用优化器更新模型的参数。

                # 检查准确率
                preds = torch.nn.functional.softmax(preds,dim=1)
                preds = torch.argmax(preds, dim=1)
                corrects += torch.sum(preds == labels).item() # item: 将torch张量转为对应的python基本数据类型

                loss_val += loss.item() * (sample['image1'].shape[0])  # 获取loss，并乘以当前批次大小

            train_loss = loss_val / len(train_sampler) # 计算整个模型的总损失
            train_acc = corrects / len(train_sampler) # 计算本次epoch的总正确率
            print(f"Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.4f}")
            train_loss_epochs.append(train_loss)
            train_acc_epochs.append(train_acc)
            # # 进行validation之前, 替换模型权重
            # model_info = copy.deepcopy(model.state_dict())
            # model_info['rc_model.readout.weight'] = (change_weights(model_info.get('rc_model.readout.weight'),device)).reshape(10,-1) # 从(dims,1) tensor变为(10,dims) tensor
            # model.load_state_dict(model_info)
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
    for sample in val_loader:
        # datas: (batch_size,3,64,64)
        # labels: (batch_size,10)
        labels = sample['label'].to(device)
        preds = model(sample)  # 前向传播
        loss = loss_func(preds, labels.long())
        loss_val += loss.item() * (sample['image1'].shape[0])

        # 检查准确率
        preds = torch.nn.functional.softmax(preds,dim=1)
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
    for sample in test_loader:
        # datas: (batch_size,3,64,64)
        # labels: (batch_size,10)
        label = sample['label'].to(device)
        preds = model(sample)  # 前向传播
        preds = nn.functional.softmax(preds,dim=1)
        preds = torch.argmax(preds, dim=1)
        predictions.append(preds)
        labels.append(label)
    return torch.cat(predictions,dim=-1).cpu().detach().numpy(),torch.cat(labels,dim=-1).cpu().detach().numpy()

def main():
    # load data
    label_mat = io.loadmat('q3_2_data.mat')
    label_train = label_mat['trLb']
    image_dataset_train = ActionDataset('.\\testClips', labels=label_train,transform=transforms.ToTensor()) # change the directory of dataset if you wish
    # label_test = label_mat['valLb']
    # image_dataset_test = ActionDataset('/root/autodl-fs/data/valClips', labels=label_test,transform=transforms.ToTensor())

    # wrap to dataloader
    batch_size = 32
    image_dataloader_train = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True,num_workers=4)  # num_workers: 调用多少进程来加载数据集
    #image_dataloader_test = DataLoader(image_dataset_test, batch_size=batch_size, shuffle=False)

    # 模型建立
    # some hyperparameters:
    epoches = 25
    folds = 3
    time_step =3
    ler = 0.5  # leaky rate
    sr = 0.9  # spectual radius
    lr = 1e-3  # learning rate
    res_unit = 196  # neuron number in reservoir(14x14)
    bi_enable = True
    cnn_parameters =  {
        'conv1':{'input_channel':3,'output_channel':32,'kernel_size':3,'stride':1},
        'MaxPool':{'kernel_size':2,'stride':2}, # 池化窗口大小2, 每次滑动2个像素
        'conv2':{'input_channel':32,'output_channel':64,'kernel_size':3,'stride':1},
    }
    cnn_model = CNNModel(cnn_parameters).to(device)
    #rc_model = RCmodel(res_unit,time_step,ler, sr,bi_enable=bi_enable).to(device)
    lstm_model = LSTMmodel(input_size=64*14*14,hidden_size=98).to(device)
    model = Model(cnn_model, lstm_model=lstm_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()

    model, train_loss, train_acc, CV_loss, CV_acc = train(model,
                                                          train_loader=image_dataloader_train, loss_func=loss_func,
                                                          optimizer=optimizer,epochs=epoches, device=device,
                                                          k_fold=folds)
    #
    loss_result_path = "./results/CNNLSTM_model_loss.jpg"
    acc_result_path = "./results/CNNLSTM_model_accuracy.jpg"
    confMatrix_result_path = "./results/CNNLSTM_model_confusion_Matrix.jpg"
    benchMarks.plot_performance_all_loss(epoches * folds, train_loss, CV_loss, loss_result_path)
    benchMarks.plot_performance_all_acc(epoches * folds, train_acc, CV_acc, acc_result_path)
    # predictions, labels = predict_on_test(model, image_dataloader_test)
    # print(
    #     f"macro accuracy:{benchMarks.accuracy(predictions, labels):.2f}\n" +
    #     f"macro precision:{benchMarks.precision(predictions, labels):.2f}\n" +
    #     f"macro recall:{benchMarks.recall(predictions, labels):.2f}\n" +
    #     f"macro F1:{benchMarks.F1(predictions, labels):.2f}\n"
    # )
    # benchMarks.plot_conf_matrix(predictions, labels, path=confMatrix_result_path)


if __name__=='__main__':
    main()
