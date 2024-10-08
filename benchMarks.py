from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report
import numpy as np

def plot_performance_each_loss(epochs, loss, title,path):
    """
    绘制损失(可以是训练或测试损失)的数值关于epochs数的关系
    :param epochs:
    :param loss:
    :param title:
    :param path: example of path if you use autodl(linux): ./results/W3/Model_loss_moreTs_fold5_epoch50.jpg
    example of path if you use windows: results\\W3\\Model_loss_moreTs_fold5_epoch50.jpg
    :return: train loss curve and validation loss curve
    """
    xlabel = "Epoch"

    epochs_list = [i+1 for i in range(epochs)]
    epochs_list_show = [i+1 for i in range(0, epochs, 10)]

    plt.figure(figsize=(20, 5))
    plt.plot(epochs_list, loss)
    plt.title(f"{title}:\n", fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.xticks(epochs_list_show, epochs_list_show)
    plt.grid()
    #plt.show()
    # 保存到本地:
    plt.savefig(path,format='jpg')


def plot_performance_all_acc(epochs, Train_acc, cv_acc,path):
    """
    绘制训练准确率和测试准确率的数值关于epochs数的关系;但是把train_acc 和 cv_acc画在一起
    :param epochs:
    :param Train_acc:
    :param cv_acc:
    :param path:
    :return: train loss curve and validation loss curve are saved to the specific path
    """
    xlabel = "Epoch"
    legends = ["Training", "Validation"]

    epochs_list = [i + 1 for i in range(epochs)]
    epochs_list_show = [i + 1 for i in range(0, epochs, 10)]

    train_acc = Train_acc
    CV_acc = cv_acc
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(epochs_list, train_acc)
    plt.title("Train Accuracy:\n", fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.xticks(epochs_list_show, epochs_list_show)
    plt.grid()
    # plt.show()

    plt.subplot(122)
    plt.plot(epochs_list, CV_acc)
    plt.title("Validation Accuracy:\n", fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.xticks(epochs_list_show, epochs_list_show)
    plt.grid()
    plt.legend(legends, loc="best")
    # plt.show()
    # 保存到本地:
    plt.savefig(path, format='jpg')

def plot_performance_all_loss(epochs, Train_loss, cv_loss, path):
    """
    绘制训练损失和测试损失的数值关于epochs数的关系;但是把train_loss 和 cv_loss画在一起
    :param epochs:
    :param Train_loss:
    :param cv_loss:
    :param path:
    :return: train loss curve and validation loss curve are saved to the specific path
    """
    xlabel = "Epoch"
    legends = ["Training", "Validation"]

    epochs_list = [i + 1 for i in range(epochs)]
    epochs_list_show = [i + 1 for i in range(0, epochs, 10)]

    train_loss = Train_loss
    CV_loss = cv_loss
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(epochs_list, train_loss)
    plt.title("Train Loss:\n", fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.xticks(epochs_list_show, epochs_list_show)
    plt.grid()
    # plt.show()

    plt.subplot(122)
    plt.plot(epochs_list, CV_loss)
    plt.title("Validation Loss:\n", fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.xticks(epochs_list_show, epochs_list_show)
    plt.grid()
    plt.legend(legends, loc="best")
    # plt.show()
    # 保存到本地:
    plt.savefig(path, format='jpg')

def prediction_of_testset(pred,original,title,xlabel,ylabel,path,label_pred="predicted values",label_original="original values",):
    """
    save the prediction figure of your model to specific path
    :param pred:
    :param original:
    :param title:
    :param xlabel:
    :param ylabel:
    :param path:
    :param label_pred:
    :param label_original:
    :return:
    """
    plt.figure(figsize=(10, 5))
    plt.plot(original, color='red', label=label_original)
    plt.plot(pred, color='blue', label=label_pred)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    # plt.show()
    plt.savefig(path, format='jpg')

def NRMSE(pred,original):
    """
    return the normalized root mean squared error(NRMSE) of predicted values and original values
    :param pred:
    :param original:
    :return: NRMSE
    """
    power_diff = np.power((original-pred),2)
    power_avg = np.mean(power_diff)
    return np.sqrt(power_avg)/(np.max(original) - np.min(original))


def MAPE(pred,original):
    """
    return the mean absolute percentage error(in percentage) of predicted values and original values
    :param pred:
    :param original:
    :return:
    """
    array_diff = np.abs(original-pred)
    array_avg = np.mean(array_diff / pred)
    return array_avg * 100

def plot_conf_matrix(preds,labels,path):
    """
    return a diagram of confusion matrix
    :param preds: prediction values
    :param labels: labels
    :return:
    """
    # 将预测值和标签转为scikit-learn形式的confusion matrix
    showed_labels = sorted(set(labels))
    conf_matrix = confusion_matrix(labels,preds)
    # 绘制混淆矩阵的热力图
    plt.figure(figsize=(8,7))  # 设置图形的大小
    img = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues,aspect='auto')  # 使用 'nearest' 插值方法，避免数值模糊
    plt.title('Confusion Matrix',fontname='Arial',fontsize=22,fontweight='bold')  # 设置图形的标题
    plt.colorbar(img) # 显示颜色条
    # 在混淆矩阵的每个单元格中添加文本标签
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),  # 'd' 表示整数格式
                    ha="center", va="center", color="red",fontsize=12)

    plt.xlabel('Predicted Label',fontname='Arial',fontsize=22,fontweight='bold')  # 设置 x 轴的标签 , 测试集sample手势对应的索引
    plt.ylabel('True Label',fontname='Arial',fontsize=22,fontweight='bold')  # 设置 y 轴的标签, 测试集label手势对应的索引
    plt.xticks(showed_labels, showed_labels,fontsize=12)
    plt.yticks(showed_labels, showed_labels,fontsize=12)
    # 显示图形
    plt.tight_layout()
    plt.savefig(path,format="jpg")

def classification_report(preds,labels):

    return classification_report(labels,preds)
def accuracy(preds,labels):
    """
    return macro accuracy of predicted values and original values
    """
    return np.sum(np.equal(preds,labels)) / len(preds)
def precision(preds,labels):
    """
    return the macro precision score of predicted values and original values
    macro: 计算每个类别的得分，然后求他们的算术平均值
    :param preds:
    :param labels:
    :return:
    """
    return precision_score(labels,preds,average='macro')

def recall(preds,labels):
    """
    return the macro recall score of predicted values and original values
    :param preds:
    :param labels:
    :return:
    """
    return recall_score(labels,preds,average='macro')

def F1(preds,labels):
    """
    return the f1 score of predicted values and original values
    :param preds:
    :param labels:
    :return:
    """
    return f1_score(labels,preds,average='macro')







