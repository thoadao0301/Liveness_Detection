from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.layers import Input
from keras.applications import inception_resnet_v2
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D
import inception_resnet_v1

def build_model(input_shape, embedding =128):
    inputs = Input(input_shape)
    kernel_size = 3
    x = Conv2D(32,\
            (kernel_size,kernel_size),\
            input_shape = input_shape,\
            padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)


    x = Conv2D(64,\
            (kernel_size,kernel_size),\
            input_shape = input_shape,\
            padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(.25)(x)

    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embedding)(pooledOutput)

    model = Model(inputs,outputs)
    return model

def model_with_base_inception_resnet_v1(input_shape,embedding=128):
    base_model = inception_resnet_v1(input_shape=input_shape,classes=embedding)
    return base_model

def euclidean_distance(vects):
    x,y = vects
    sum_squared = K.sum(K.square(x-y),axis=1,keepdims=True)
    return K.square(K.maximum(sum_squared,K.epsilon()))

def constrastive_loss(y_true,y_pred):
    margin = 1
    y_true = tf.cast(y_true,'float32')
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin-y_pred,0)))

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()
    