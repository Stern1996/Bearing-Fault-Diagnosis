# 6月7日更新：
该方案是利用LeNet5模型对轴承故障信号转化而成的灰度图进行识别分类。

在LeNet5模型训练之前，需要对所用一维振动信号进行采样、灰度图转换并存储，建立对应的训练数据集。

## How to use
1. 在模型训练前，先将用于模型训练的轴承振动数据存放在dataset文件夹下.（数据格式为mat文件）

1. Before model training, store the bearing vibration data for model training in the "dataset" directory.（The data format is .mat）

2. 运行"dataset_generate.py"，设置需要抽取灰度图的尺寸，每个mat文件采样样本数。数据集将存储在“data”文件夹下。

2. Run "dataset_generate.py", set the size of the gray image to be extracted, and the number of samples per mat file to be sampled. The dataset will be stored in the "data" folder.

3. 运行“grey_lenet5_net.py”，训练模型，模型参数可自行在程序中进行调整。训练过程将保存在图片中（LeNet5_acc.png 和 LeNet5_loss.png）.

3. Run "grey_lenet5_net.py" to train the model, the model parameters can be adjusted in the program. The training process will be saved in the images (LeNet5_acc.png and LeNet5_loss.png)
