## Mask-RCNN Implementation with Tensorflow 1.13 and Keras

### 说明
这个仓库是本人对<a href="https://github.com/matterport/Mask_RCNN" style="text-decoration:none">基于 tensorflow 和 keras 实现的 Mask-RCNN 模型的 Repository</a> 中相关代码的研读、超详细的注释和主要程序流程的可视化。以及一些应用实例。
* 关于该实现的 Mask_RCNN 模型的使用，详见原 repository: <a href="https://github.com/matterport/Mask_RCNN" style="text-decoration:none">Mask R-CNN for Object Detection and Segmentation</a>。
* 一些必备的 python 库及版本要求，详见 requirements.txt。
* 
* mrcnn 这个文件夹为模型实现的核心代码。本人详细研读并注释了其中 model.py、utilis.py 以及 visualize.py 这三个脚本。并基于个人理解绘制出了 training 和 predicting 两个阶段的相应的流程图（流程图见下方）。
* datasets 文件夹里为各应用实例的数据集。
* samples文件夹里为各应用实例的代码，里面各个应用各自独立成一个文件夹。my_notebook 文件夹中也有部分相关应用实例的试验性和探索性的代码（.ipynb 格式文件）。
* logs 文件夹里是模型在不同应用中的训练权重。
* my_images 和 my_videos 这两个文件夹里是



这个仓库仅用于本人个人学习之用。如果有幸能被有缘的朋友看到，因个人才疏学浅或粗心大意，代码注释和流程图中可能会存在谬误之处，君若发现，恳请不吝指出。如果这个分享对您有帮助，那本人将感到高兴和荣幸。
