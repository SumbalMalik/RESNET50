from keras import layers
from keras import models
import numpy as np
from convolutional_block import convolutional_block
from identity_block import identity_block
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

def resnet(X):
    #zero padding 3x3
    X = layers.ZeroPadding2D(padding=(3,3), data_format = "channels_last")(X)

    #Stage 1
    # 2D convolution, filters (7,7), stride (2,2) , filters 64
    X = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2,2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3,3), strides = (2,2))(X)

    #Stage 2
    #f = 3  #kernel size
    block = 'a'
    #convolutional block
    X = convolutional_block(X, f=3, filters=[64, 64, 256 ], stage = 2, block = 'a', s=1)
    #2 identity blocks of filters [64, 64, 256], kernel f=3
    for i in range(2):
        block = 'b' if i == 0 else 'c'
        X = identity_block(X, f = 3, filters = [64, 64, 256 ], stage = 2, block = block)

    #Stage 3
    block = 'a'
    #convolutional block
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = block, s = 2)
    #3 identity blocks of filter [128, 128, 512], f=3
    for i in range(3):
        if i == 0:
            block = 'b'
        elif i == 1:
            block = 'c'
        elif i == 2:
            block = 'd'

        X = identity_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = block)

    #Stage 4
    block = 'a'
    #convolutional block
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = block, s = 2)
    #5 identity blocks of filters [256, 256, 1024], f=3
    for i in range(5):
        if i == 0:
            block = 'b'
        elif i == 1:
            block = 'c'
        elif i == 2:
            block = 'd'
        elif i == 3:
            block = 'e'
        elif i == 4:
            block = 'f'
        
        X = identity_block(X, f = 3, filters  = [256, 256, 1024], stage = 4, block = block)

    #Stage 5
    block = 'a'
    #convolutional block
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = block, s = 2)
    #2 identity blocks with filters [512, 512, 2048], f=3
    for i in range(2):
        if i == 0:
            block = 'b'
        elif i == 1:
            block = 'c'
        
        X = identity_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = block)
   
    # 2D average pooling with window shape (2, 2)
    X = layers.AveragePooling2D(pool_size=(2,2),name = 'avg_pool', data_format='channels_last')(X)
    # flatten
    X = Flatten()(X)
    X = layers.Dropout(0.4)(X)
    X = layers.Dense(1, activation='sigmoid', name = 'fc')(X)
    return X


img_height = 224
img_width = 224
img_channels = 3
classes = 2  
train_data_dir = 'cats-dogs/data/train'
valid_data_dir = 'cats-dogs/data/validation'
datagen = ImageDataGenerator(rescale = 1./255)
train_generator = datagen.flow_from_directory(directory=train_data_dir,
											   target_size=(img_width,img_height),
											   classes=['dogs','cats'],
											   class_mode='binary',batch_size=8)

validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
											   target_size=(img_width,img_height),
											   classes=['dogs','cats'],
											   class_mode='binary',batch_size=8)
# input
input = layers.Input(shape=(img_height, img_width, img_channels))
#output
output = resnet(input)
model = models.Model(inputs=[input], outputs=[output])
#sgd = optimizers.SGD(lr=0.1)
model.compile(loss='binary_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit_generator(generator=train_generator, steps_per_epoch=2048/16 ,epochs=15,validation_data=validation_generator,validation_steps=832//16)
print (model.summary())
model.save("resnet.h5")
