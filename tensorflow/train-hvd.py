import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
import horovod.tensorflow.keras as hvd

def getVGGModel():
    input_tensor = Input(shape=(224,224,3))
    vgg_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = vgg_model.get_layer('block5_pool').output

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation='softmax')(x)
    cus_model = keras.Model(inputs=vgg_model.input, outputs=x)
    return cus_model

if __name__ == '__main__':
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    verbose = 1 if hvd.rank() == 0 else 0

    model = getVGGModel()

    for layer in model.layers[:19]:
        layer.trainable = False


    datagen = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_it = datagen.flow_from_directory('/tmp/Dataset_2/Train/', target_size=(224,224), class_mode='categorical', batch_size=4, shuffle=True)
    val_it = datagen.flow_from_directory('/tmp/Dataset_2/Validation/', target_size=(224,224), class_mode='categorical', batch_size=16, shuffle=True)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=5, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.0000001)

    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    #print(model.summary())

    model.fit_generator(train_it, 
                        steps_per_epoch=83, 
                        validation_data=val_it, 
                        validation_steps=363//16, 
                        callbacks=[learning_rate_reduction, hvd.callbacks.BroadcastGlobalVariablesCallback(0)],
                        epochs=3,
                        verbose=verbose)
