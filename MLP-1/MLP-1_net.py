#MLP的模型框架结构(1层隐藏层),sklearn搭建
#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from keras_flops import get_flops
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#读入数据集并作标准化处理
def get_datasets():
    # 每个样本为（2048，）
    datasets, labels = np.load('./data/database.npy'), np.load('./data/labels.npy')
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2, random_state=0)
    y_train, y_test = y_train.flatten(), y_test.flatten()
    #数据标准化，MLP-API中提到，Multi-layer Perceptron is sensitive to feature scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def model_train():
    X_train, X_test, y_train, y_test = get_datasets()
    #搭建模型，迭代次数设置为100
    mlp = MLPClassifier(max_iter=100)
    #网格搜索的方式来大致确定下超参数设置,hidden_layer_size中，元素个数表示隐层层层数，数值表示改层神经元数量，这里是单层神经元，只调个数
    parameter_space = {
        'hidden_layer_sizes': [(10,), (20,), (30,),(40,),(50,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    #n-jobs表示使用CPU数，-1为所有可用CPU，cv为交叉验证折数
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)
    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)
    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    #模型验证，输出结果报告
    y_true, y_pred = y_test, clf.predict(X_test)
    print('Results on the test set:')
    print(classification_report(y_true, y_pred))

    '''
    flops = get_flops(model,batch_size=32)
    flops = flops/2
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    '''
    #保存模型
    joblib.dump(clf, "mlp.pkl")
    print("[INFO] Model has been saved !")

if __name__ == "__main__":
    model_train()
