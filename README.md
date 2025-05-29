# 神经网络与深度学习期中项目报告

本项目旨在使用预训练的 ResNet-18 模型，通过微调技术对 Caltech-101 数据集进行图像分类。实验对比了微调模型与从头开始训练模型的性能，并利用 TensorBoard 对训练过程进行了可视化（尽管最终的超参数搜索和从头训练部分为了简洁性移除了 TensorBoard 日志记录）。

## 实验内容简介

-   **数据集**: [Caltech-101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)。该数据集包含101个日常物品类别以及一个背景类别，共约9000张图像。在本实验中，图像被统一调整为 128x128 像素。
-   **模型**: ResNet-18。我采用了在 ImageNet 上预训练的 ResNet-18 模型。
-   **训练方法**:
    1.  **微调 (Fine-tuning)**:
        *   替换预训练模型的全连接层（分类头）以适应 Caltech-101 的101个类别。
        *   新添加的全连接层从随机初始化开始训练，并使用较大学习率。
        *   ResNet-18 较早的卷积层被冻结，以保留其通用的特征提取能力。
        *   选择性地解冻并微调了 ResNet-18 的 `layer3` 和 `layer4`，使用较小的学习率。
    2.  **从头训练 (Training from Scratch)**:
        *   使用与微调模型相同的 ResNet-18 架构，但不加载任何预训练权重。
        *   所有网络参数从随机初始化开始训练。
-   **关键结果**:
    *   微调模型在经过5个 epoch 的训练后，在验证集上达到了约 **90.55%** 的准确率 (具体数值请参照 `report.ipynb` 中ID为 `8c32aef4` 的单元格输出的最后一个epoch的验证准确率)。
    *   从头训练的模型在经过5个 epoch 的训练后，在验证集上达到了约 **62.62%** 的准确率 (具体数值请参照 `report.ipynb` 中ID为 `4d1f1b99` 的单元格输出的最后一个epoch的验证准确率)。
    *   实验清晰地表明，在有限的数据和训练轮次下，微调预训练模型相比从头训练能够取得显著更优的性能。

--- 

## 如何运行

### 1. 数据准备

1.  下载 Caltech-101 数据集。
2.  解压数据集。确保图像按类别存放在子文件夹中。
3.  将数据集的主类别文件夹 (通常是 `101_ObjectCategories`) 放置到项目工作目录下的 `caltech-101/101_ObjectCategories/` 路径下。
    最终的图像路径结构应类似于：
    ```
    <工作目录>/
    ├── caltech-101/
    │   └── 101_ObjectCategories/
    │       ├── accordion/
    │       │   ├── image_0001.jpg
    │       │   └── ...
    │       ├── airplanes/
    │       │   ├── image_0001.jpg
    │       │   └── ...
    │       └── ... (其他类别文件夹)
    ├── main.py
    ├── report.ipynb
    ├── requirements.txt
    └── ... (其他项目文件)
    ```
4.  **重要**: 数据集中的 `BACKGROUND_Google` 不被视为一个类别。如果数据集中存在 `BACKGROUND_Google` 文件夹，请**手动删除**该文件夹，或者确保代码中的数据加载部分能够正确处理或过滤它。本项目提供的 `main.py` 和 `report.ipynb` 中的 `get_data_loaders` 函数默认直接使用 `ImageFolder`，它会将所有子文件夹视为类别。

### 2. 执行训练与评估

#### a)  安装依赖

项目根目录下提供了一个 `requirements.txt` 文件。使用 pip 安装所有列出的依赖项：
```bash
pip install -r requirements.txt
```

#### b) 使用 `main.py` 脚本 (原始微调实验)

可以直接运行 `main.py` 脚本来执行标准的微调训练。

```bash
python main.py
```

该脚本会：
-   加载数据。
-   定义并微调 ResNet-18 模型。
-   默认训练5个 epoch (如 `NUM_EPOCHS` 变量所定义)。
-   打印训练过程中的损失和准确率。
-   将 TensorBoard 日志保存到 `runs/Caltech101_ResNet18_<timestamp>` 目录下。

要查看 TensorBoard：
```bash
tensorboard --logdir=runs
```
然后在浏览器中打开 `http://localhost:6006`。

#### c) 使用 Jupyter Notebook (`report.ipynb`)

`report.ipynb` 文件包含了对实验的详细介绍、代码分块执行以及结果分析。可以按顺序执行 Notebook 中的单元格。

-   **默认微调训练（同 `main.py`）**: Notebook 中的主执行逻辑单元格（ID `8c32aef4`）会执行一次标准的微调训练，并将 TensorBoard 日志保存到 `runs/Notebook_Run_<timestamp>`。
-   **超参数搜索**: Notebook中包含一个单元格 (ID `b5023a24`) 用于执行一个简化的超参数搜索，可以修改其中的超参数范围。此部分当前配置为不记录 TensorBoard 日志。
-   **从头训练**: Notebook中包含一个单元格 (ID `4d1f1b99`) 用于从头训练 ResNet-18。
-   **TensorBoard 可视化**: Notebook中包含一个单元格 (ID `d1fa4f12`) 用于内嵌显示指定日志目录的 TensorBoard。请确保路径正确并取消相应命令的注释。

### 3. 测试 (加载已训练模型)

代码中已包含保存模型参数的功能 (在 `report.ipynb` 中的单元格 ID `31332775`，或可以在 `main.py` 中取消注释相关代码)。要加载已训练的模型进行测试或推理，请：

1.  确保有已保存的 `.pth`权重文件（例如 `model_parameters.pth`）。
2.  重新实例化模型架构。
3.  使用 `model.load_state_dict(torch.load('path/to/your/model_parameters.pth'))` 加载权重。
4.  将模型设置为评估模式 `model.eval()`。
5.  在测试数据上进行推理。

(具体的测试脚本未在此项目中直接提供，但上述步骤是标准的PyTorch流程。)

## 训练好的模型权重

我提供了通过微调策略（具体配置见 `report.ipynb` 中ID为 `8c32aef4` 的单元格的训练输出）训练5个 epoch 后得到的模型权重。

-   **下载地址**: [Google Drive 链接 - model_parameters.pth](https://drive.google.com/file/d/1TpXDjjGhcojaFSK53TGxED5ZgdDrY3OV/view?usp=drive_link)

该模型在验证集上达到了约 **90.55%** 的准确率。

---
