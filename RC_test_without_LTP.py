import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,SubsetRandomSampler
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Input

rpy.verbosity(0)
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past,len(dataset)):
        dataX.append(dataset[i-n_past:i,0:dataset.shape[1]])
        dataY.append(dataset[i,0]) # 预测open列
    return np.array(dataX),np.array(dataY)

#获取数据
df = pd.read_csv("data\\train.csv",parse_dates=["Date"],index_col=[0]) # 将date列解析为日期时间对象，并用第0列作为index
test_split = round(len(df)*0.20)
df_for_training = df.iloc[:-test_split] # 0~末尾-test_split
df_for_testing = df.iloc[-test_split:]

# 用MinMaxScalar缩放
scaler=MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.fit_transform(df_for_testing)
trainX, trainY = createXY(df_for_training_scaled,30)
testX, testY = createXY(df_for_testing_scaled,30)

# 封装为tensor:
trainX_tensor = torch.tensor(trainX,dtype=torch.float32)
trainY_tensor = torch.tensor(trainY,dtype=torch.float32)
testX_tensor = torch.tensor(testX,dtype=torch.float32)
testY_tensor = torch.tensor(testY,dtype=torch.float32)

# 创建 TensorDataset
train_dataset = TensorDataset(trainX_tensor, trainY_tensor)
test_dataset = TensorDataset(testX_tensor, testY_tensor)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试集通常不打乱


# 模型建立
# some hyperparameters:
lr = 1e-3 # 学习率
batch_size = 16
epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fold = 3

ler = 0.5 # leaky rate
sr = 0.9 # spectual radius
lr = 1e-3 # learning rate
res_unit = 100 # neuron number in reservoir


# 建立模型：单层reservoir
class RCModel(nn.Module):
    def __init__(self, res_unit, ler, sr):
        super(RCModel, self).__init__()
        self.res_unit = res_unit
        self.RCmodel = self.create_RC_model(res_unit, ler, sr)
        self.readout = nn.Linear(res_unit, 1)

    def create_RC_model(self, res_unit, ler, sr):
        data = Input()
        reservoir = Reservoir(res_unit, lr=ler, sr=sr, name='res1-1')
        model = data >> reservoir
        return model

    def forward(self, x):
        """
        x: input data with shape: [batch_size, time_step, feature_size]
        """
        # running reservoir and print output
        batch_size = x.shape[0]
        data = np.array(x)
        # every sample, which has a timestep=30 and feature_size=5
        # 这里每一个sample只保留最后一个时间步的state, 和reservoirPy里面的warmup逻辑不一样（warmup似乎是把20个state的输入都拿去训练了）
        curr_state = self.RCmodel.run(data) # 返回每个时间步的step, 是一个列表（len=batchsize）, 列表的元素是np.array(shape=(timestep,units))
        curr_states = torch.empty((batch_size, self.res_unit))
        for i in range(batch_size):
            curr_states[i] = torch.tensor(curr_state[i][-1], dtype=torch.float32)
        # one layer readout,
        x = self.readout(curr_states)
        x = torch.flatten(x) # 平整为1维向量
        return x

# 测试函数
def test(model, test_loader, loss_func, device):
    model.eval()
    loss_val = 0.0
    for datas, labels in test_loader:
        datas = datas.to(device)
        labels = labels.to(device)

        preds = model(datas)
        loss = loss_func(preds, labels.long())
        loss_val += loss.item() * datas.size(0)

    test_loss = loss_val / len(test_loader.dataset)  # 计算整个测试集的总损失
    print(f"Test Loss: {test_loss:.7f}")
    return test_loss

# 训练函数
def train(model, train_loader, optimizer, loss_func, epochs, device,k_fold):
    import copy
    """
    define training process of upon model, 使用K折验证
    return:
    model: the model after training
    train_losses: loss for each epoch in training(最优一折的数据)
    CV_losses: loss for each epoch in validation(最优一折的数据)
    """
    k = k_fold  # k折交叉验证
    best_val_loss_list = [0.0 for i in range(k)]  # 跟踪每个折最佳的损失
    best_models = []  # 记录每一折训练得到的最优model，然后根据正确率得到一个全局最优model
    best_model_params = copy.deepcopy(model.state_dict())  # 创建当前模型参数的深拷贝
    kf = KFold(n_splits=k)
    train_loss_folds = []
    CV_loss_folds = []

    for fold, (train_indexes, val_indexes) in enumerate(kf.split(train_loader.dataset)):
        train_sampler = SubsetRandomSampler(train_indexes)  # 告诉dataloader应该加载与len(train_indexes)数量相同，与train_indexes对应的样本
        val_sampler = SubsetRandomSampler(val_indexes)
        curr_train_loader = DataLoader(train_loader.dataset, batch_size=16, sampler=train_sampler)
        val_loader = DataLoader(train_loader.dataset, batch_size=16, sampler=val_sampler)

        # 记录训练损失和正确率，用于画图：
        train_loss_epochs = []
        # 记录每次验证的损失和正确率，用于画图：
        CV_loss_epochs = []

        for epoch in range(epochs):
            print(f" fold{fold+1}, epoch{epoch + 1}:...")
            model.train()  # 设置为训练模式
            loss_val = 0.0
            for datas, labels in curr_train_loader:
                # datas: (batch_size,input_size(30),features(5))
                # labels: (batch_size,1)
                datas = datas.to(device)
                labels = labels.to(device)

                preds = model(datas)  # 前向传播
                loss = loss_func(preds, labels)  # 计算损失

                optimizer.zero_grad()  # 清除优化器梯度（来自于上一次反向传播）
                loss.backward()  # 反向传播, 计算模型参数梯度
                optimizer.step()  # 根据计算得到的梯度，使用优化器更新模型的参数。

                loss_val += loss.item() * datas.size(0)  # 获取loss，并乘以当前批次大小

            train_loss = loss_val / len(curr_train_loader.dataset) # 计算整个模型的总损失
            print(f"Train Loss: {train_loss:.7f}")
            train_loss_epochs.append(train_loss)
            # 本笔记中不进行模型权重的替换
            # # 进行validation之前, 替换模型权重
            # model_info = copy.deepcopy(model.state_dict())
            # model_info['readout.weight'] = (change_weights(model_info.get('readout.weight'))).reshape(1,-1) # 从(100,1) tensor变为(1,100) tensor
            # model.load_state_dict(model_info)
            val_loss = []
            # 每个epoch都进行评估：
            val_loss = test(model, val_loader, loss_func, device)
            if (best_val_loss_list[fold] < val_loss):  # 出现最优模型时(损失最小的模型)，保存最优模型
                best_val_loss_list[fold] = val_loss
                best_model_params = copy.deepcopy(model.state_dict())
            # 更新平均loss指标
            CV_loss_epochs.append(val_loss)


        # 更新每个fold的模型和训练及测试的loss记录
        model.load_state_dict(best_model_params)
        best_models.append(model)
        train_loss_folds.append(train_loss_epochs)
        CV_loss_folds.append(CV_loss_epochs)

    best_val_loss_index = np.argmax(np.array(best_val_loss_list))
    print(best_val_loss_index)
    model = best_models[best_val_loss_index]
    train_losses = train_loss_folds[best_val_loss_index]
    CV_losses = CV_loss_folds[best_val_loss_index]
    return model, train_losses, CV_losses

def change_weights(model_params):
    weights = (pd.read_csv("weights\\LTP.csv")).to_numpy()  # change to numpy array
    weights = torch.tensor(weights.reshape(-1, 1).flatten(), dtype=torch.float32) # change to 1D tensor
    indice_of_nearest = torch.argmin(torch.abs(model_params.reshape(-1,1)-weights), dim=1)
    replaced_weights = weights[indice_of_nearest]
    return replaced_weights


# 训练过程定义：
model = RCModel(res_unit,ler,sr)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
loss_func = nn.MSELoss()
model,train_loss,CV_loss = train(model,train_loader,optimizer,loss_func,epochs,device,fold)

prediction = model(testX)
print("prediction shape-", prediction.shape)
print(len(prediction))