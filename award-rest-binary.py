import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
import shutil
import keras
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import numpy as np
import tensorflow as tf
import os
from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras import layers
import tensorflow.contrib.slim as slim
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn import preprocessing
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator  #导入迭代器
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)

Width=224
Height=224
file_dir='D:\\deeplearning\\task1-award-rest'
train_dir=file_dir+'\\train'
test_dir=file_dir+'\\test'
#train_sample=24000
train_datagen = ImageDataGenerator(  #数据增广
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
train_generator = train_datagen.flow_from_directory(  #迭代器取数据
    train_dir,
    target_size=(Width,Height),
    batch_size=128,
    class_mode='binary',
	#classes=['award','rest']
)
label_dict=train_generator.class_indices
print(label_dict)
#bottleneck_features_train = model.predict_generator(train_generator, 1777)
# save the output as a Numpy array
#np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(Width, Height),
    batch_size=128,
    class_mode='binary',
    #classes=['award_test','middle_test','low_test']
)
#bottleneck_features_validation = model.predict_generator(validation_generator, 439)
#np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
#model_weight_path='D:\deeplearning\vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
model_vgg19=applications.VGG19(include_top=False,weights='imagenet')
model_vgg16=applications.VGG16(include_top=False,weights='imagenet')
model_res=ResNet50(include_top=False,weights='imagenet')
model = keras.Sequential()
model.add(model_vgg19)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model_vgg16.trainable=False
model_res.trainable=False
model_vgg19.trainable=False
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),    #编译
              loss='binary_crossentropy',
              metrics=['acc'])
tbCallBack=TensorBoard(log_dir='./log',histogram_freq=0,write_graph=True)
history=model.fit_generator(train_generator, steps_per_epoch=100,
                              epochs=300,
                              validation_data=validation_generator,
                              validation_steps=25,
							  callbacks=[tbCallBack])
validation_generator.reset()
pred = model.predict_generator(validation_generator, verbose=1,steps=100)

predicted_class_indices = np.argmax(pred, axis=1)
labels = (train_generator.class_indices)
label = dict((v,k) for k,v in labels.items())

# 建立代码标签与真实标签的关系
predictions = [label[i] for i in predicted_class_indices]

#建立预测结果和文件名之间的关系
filenames = validation_generator.filenames
predictY=[]
trueY=[]
target_names=['award','rest']
for idx in range(len(filenames )):
	#print('predict  ' +((predictions[idx])))
	if predictions[idx] in 'award':
		predictY.append(0)
	elif predictions[idx] in 'rest':
		predictY.append(1)
	#print('title    %s' % filenames[idx])
	if 'award' in filenames[idx]:
		trueY.append(0)
	elif 'rest' in filenames[idx]:
		trueY.append(1)
from sklearn.metrics import classification_report
print(classification_report(trueY, predictY, target_names=target_names))
#Pred = model.predict_generator(validation_generator,step=50,verbose=1)
#plot_curve(history)
#model.save('category.h')
#model.predict_generator(train_generator,train_sample)
'''
modelweight = '../input/VGG-19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
model_vgg16 = applications.VGG16(include_top=False, weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
model = keras.Sequential()
model.add(model_vgg16)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model_vgg16.trainable = False
model.summary()
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),    #编译
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit_generator(   #训练
    train_generator,
    steps_per_epoch=1000,
    epochs=20,
    validation_data=test_generator,
    validation_steps=50)
'''