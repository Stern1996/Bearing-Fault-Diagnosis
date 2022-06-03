# Bearing-Fault-Diagnosis
6月3日更新：

重写了SVM部分代码，结构相对更加简单，不对原始数据进行预处理


2月6日更新：

轴承故障检测所使用数据集来源于CWRU轴承数据中心。

CWRU数据集包含特征有：3种故障尺寸，4种工作负载，3种故障类型（其中轴承外圈故障还包含三种不同故障位置）

在本资源库中，所有模型识别标签种类均为4。故障尺寸与工作负载并未作为划分不同识别标签的指标。

06, Feb.

The datasets used for bearing failure detection are from the CWRU Bearing Data Center.

The CWRU datasets contain features such as: 3 fault diameter, 4 motor load, 3 fault types (with outer race faults also containing three different fault positions)

In this repository, the number of labels for fault diagnosis are the same -- 4. Fault diameter and motor load are not used as criteria to classify the different labels.

2月3日更新：

在这个资源库中，列出了一些典型的机器学习模型。这些模型用于轴承故障诊断。所有模型的结构与参数均来源于对已有相关论文的复现。

机器学习模型的各模块被存储在相应的文件夹中。由于到目前为止，我只对这个主题的不同机器学习模型方案做了初步研究，暂未写一个主程序将所有模块连接在一起，所以模块必须按顺序运行。

你可以在相应的文件夹中看到运行顺序。(Readme文件)

03, Feb.

In this repository, some typical machine learning models are listed. These models are used for bearing fault diagnosis. The structure and parameters of all models are derived from the reproduction of existing related papers.

The modules of ML-models are stored in the corresponding folders. Since so far I have only done a preliminary study of the different ML-Models of this theme, there is no one main program that connects all modules together, so the modules have to be run in order.

You can see the running order in the corresponding folder.(Readme-File)
