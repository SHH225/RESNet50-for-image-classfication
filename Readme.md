# 基于Keras 实现作品集分类

<hr>


### 讲在前面,易出现问题的地方

GPU 显存占用问题
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
```
图片分辨率过高 
```python
from PIL import Image
```

### 网络选取
* **RESNET**
   ILSVRC2015比赛中取得冠军
   
   >Paper：Deep Residual Learning for Image Recognition 2015
   
   Resnet50 Resnet152 ......
   
   网络结构
   
   <img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g7e1e2xy3vj30u20cun1b.jpg" style="zoom: 50%;" />
   
      	依赖于微架构模组，提出残差学习的思想，**残差网络**，传统的卷积网络或者全连接网络在信息传递的时候或多或少会存在信息丢失，损耗等问题，同时还有导致梯度消失或者梯度爆炸，导致很深的网络无法训练。
      	
      	ResNet在一定程度上解决了这个问题，通过直接将输入信息绕道传到输出，保护信息的完整性，整个网络只需要学习输入、输出差别的那一部分，简化学习目标和难度。
      	
      	ResNet最大的区别在于有很多的旁路将输入直接连接到后面的层，这种结构也被称为shortcut或者skip connections。
   
   
   

 残差块

<img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g7e1nabqklj309c05t74i.jpg" style="zoom:67%;" />

* **VGG**
   VGG在2014年的 ILSVRC localization and classification 两个问题上分别取得了第一名和第二名
   >Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition

   VGG16 VGG19...
   
   网络结构
   
   <img src="/Users/pluto/Desktop/屏幕快照 2019-09-27 下午3.24.43.png" style="zoom:50%;" />
   
   ![](https://tva1.sinaimg.cn/large/006y8mN6gy1g7e413ubr2j30d207o759.jpg)
   
   VGGNet有两个不足：

      1. 训练很慢；

      2. weights很大。

      由于深度以及全连接节点数量的原因，VGG16的weights超过533MB，VGG19超过574MB，这使得部署VGG很令人讨厌。虽然在许多深度学习图像分类问题中我们仍使用VGG架构，但是小规模的网络架构更受欢迎（比如SqueezeNet, GoogleNet 等等）



### 数据准备

本地数据存放格式要求

数据集总文件夹(./task_2)下分为/train和/test两个文件夹

![](https://tva1.sinaimg.cn/large/006y8mN6ly1g7e2mluqpij30g6018dfv.jpg)

./train 下级文件结构为：./award_train ./middle_train ./low_train

./test 下级文件结构为：./award_test ./middle_test ./low_test

![](https://tva1.sinaimg.cn/large/006y8mN6ly1g7e2n7gazvj30ed022t8w.jpg)

./award_train 文件夹为数据文件

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g7e2oq9c4dj306d07xgm5.jpg)

数据集 图像分辨率过大 resizeTo 200*200

网络输入 ：224*224
数据增广
```python
train_datagen = ImageDataGenerator( 
	rescale=1./255,
	featurewise_center=True,
	featurewise_std_normalization=True,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest'
)                        
test_datagen = ImageDataGenerator(rescale=1./255)
```
迭代器取数据
```python
train_generator = train_datagen.flow_from_directory(  #迭代器取数据
    train_dir,
    target_size=(Width,Height),
    batch_size=128,
    class_mode='binary',
	#classes=[]
)
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(Width, Height),
    batch_size=128,
    class_mode='binary',
    #classes=['award_test','middle_test','low_test']
)
```
检验是否符合类别结果
```python
label_dict=train_generator.class_indices
print(label_dict)
```

### 模型使用

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g7e2du6166j31h10u040y.jpg)

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g7e2c8mb75j31jb0u0wl6.jpg)



加载模型、使用预训练权重
```python
model_vgg19=applications.VGG19(include_top=False,weights='imagenet')
model_vgg16=applications.VGG16(include_top=False,weights='imagenet')
model_res=ResNet50(include_top=False,weights='imagenet')
#权重无需再训练
model_vgg16.trainable=False
model_res.trainable=False
model_vgg19.trainable=False

```

模型构建
```python
model = keras.Sequential()
model.add(model_vgg19)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
#model.add(layers.Dense(3, activation='softmax'))
```

### 训练

模型编译
```python
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),    #编译
              loss='binary_crossentropy',
              metrics=['acc'])
#损失函数若为多分类：categorical_crossentropy
#optimizer: 优化器
```
训练
```python
#使用tensorboard
tbCallBack=TensorBoard(log_dir='./log',histogram_freq=0,write_graph=True)

history=model.fit_generator(train_generator, steps_per_epoch=100,
                              epochs=300,
                              validation_data=validation_generator,
                              validation_steps=25,
							  callbacks=[tbCallBack])
```

### 结果分析

<img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g7e3mk5uujj31mg0aqtc4.jpg" style="zoom:67%;" />

**Tensorboard Rsults：**

![](https://tva1.sinaimg.cn/large/006y8mN6ly1g7eaq0t55dj31j70u0kjm.jpg)