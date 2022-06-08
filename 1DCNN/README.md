# 6月8日更新：
该方案是利用1DCNN模型对轴承故障信号直接进行识别分类。

在1DCNN模型训练之前，需要对所用数据集进行预处理,通过dataset_generate.py完成

## How to use
1. 在模型训练前，先将用于模型训练的轴承振动数据存放在dataset文件夹下.（数据格式为mat文件）

1. Before model training, store the bearing vibration data for model training in the "dataset" directory.（The data format is .mat）

2. 运行"dataset_generate.py"，设置每个样本长度，每个mat文件采样样本数。数据集将存储在“data”文件夹下。

Run "dataset_generate.py", set the number of points in each sample, and the number of samples per mat file to be sampled. The dataset will be stored in the "data" folder.

3. 运行“1DCNN_net.py”，训练模型，模型参数可自行在程序中进行调整。训练过程将保存在图片中（1DCNN_acc.png 和 1DCNN_loss.png）.

Run "1DCNN_net.py" to train the model, the model parameters can be adjusted in the program. The training process will be saved in the images (1DCNN_acc.png and 1DCNN_loss.png)
