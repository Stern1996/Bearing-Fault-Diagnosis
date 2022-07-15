# 6月8日更新：
该方案是利用CNN模型对轴承故障信号通过时频转换后得到的特征图进行识别，从而对轴承故障类型进行分类。

在CNN模型训练之前，需要对所用数据集进行处理。模型所识别数据为基于振动信号的时频转换图，故会先对数据进行转换、整理。

## How to use
1. 在模型训练前，先将用于模型训练的轴承振动数据存放在dataset文件夹下.（数据格式为mat文件）

1. Before model training, store the bearing vibration data for model training in the "dataset" directory.（The data format is .mat）

2. 运行"dataset_generate.py"，设置需要用于stft处理的数据段（样本长度），每个mat文件采样样本数。数据集将存储在“data”文件夹下。通过stft转换得到的图片将缩放尺寸为96x96。

2. Run "dataset_generate.py", set the number of datapoints used for stft-transformation in each sample, and the number of samples per mat file to be sampled. The dataset will be stored in the "data" folder. The images obtained by stft-transformation will be resized to 96x96.

3. 运行“stft_CNN_net.py”，训练模型，模型参数可自行在程序中进行调整。训练过程将保存在图片中（stft_CNN_acc.png 和 stft_CNN_loss.png）.

3. Run "stft_CNN_net.py" to train the model, the model parameters can be adjusted in the program. The training process will be saved in the images (stft_CNN_acc.png and stft_CNN_loss.png)
