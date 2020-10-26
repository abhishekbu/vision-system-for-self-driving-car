# Importing the libraries 
import numpy as np
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1";
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D, Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adadelta, Adam
from keras.models import *
import os

# Total number of images
NO_OF_TRAINING_IMAGES = len(os.listdir('./dataset/img/train/'))

# Hyper parameters
batch_size = 8
epochs = 50
pool_size = (2, 2)
input_shape = (80, 160, 3)

# Creation of model
model = Sequential()
# Normalizes incoming inputs
model.add(BatchNormalization(input_shape=input_shape))
# Conv Layer 1
model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))
# Conv Layer 2
model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))
# Pooling 1
model.add(MaxPooling2D(pool_size=pool_size))
# Conv Layer 3
model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
model.add(Dropout(0.2))
# Conv Layer 4
model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
model.add(Dropout(0.2))
# Conv Layer 5
model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
model.add(Dropout(0.2)) 
# Pooling 2
model.add(MaxPooling2D(pool_size=pool_size))
# Conv Layer 6  
model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
model.add(Dropout(0.2))
# Conv Layer 7
model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
model.add(Dropout(0.2))
# Pooling 3
model.add(MaxPooling2D(pool_size=pool_size))
# Upsample 1
model.add(UpSampling2D(size=pool_size))
# Deconv 1
model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))
model.add(Dropout(0.2))
# Deconv 2
model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))
model.add(Dropout(0.2))
# Upsample 2
model.add(UpSampling2D(size=pool_size))
# Deconv 3
model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))
model.add(Dropout(0.2))
# Deconv 4
model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))
model.add(Dropout(0.2))
# Deconv 5
model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))
model.add(Dropout(0.2))
# Upsample 3
model.add(UpSampling2D(size=pool_size))
# Deconv 6
model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))
# flatten
model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'sigmoid', name = 'Final')) # (80, 160, 1)

# Optimization Algorithm
optimizer = Adam()

# Compiling the model
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

# Printing the model summary
model.summary()

# Set callbacks
callbacks = [
        EarlyStopping(monitor='loss', mode='min', patience=5, verbose=1),
        ModelCheckpoint('lane_detection_2.h5',
                        verbose=1, save_weights_only=False, monitor='loss', mode='min', save_best_only=True),
    ]

# Preparing the input pipeline
train_frames_datagen = ImageDataGenerator()
train_masks_datagen = ImageDataGenerator(rescale = 1./255)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
train_frames_generator = train_frames_datagen.flow_from_directory(
    'dataset/img/',
    target_size=(80, 160),
    seed=seed,
    class_mode=None,
    batch_size=batch_size)

train_masks_generator = train_masks_datagen.flow_from_directory(
    'dataset/labels/',
    color_mode="grayscale",
    target_size=(80, 160),
    seed=seed,
    class_mode=None,
    batch_size=batch_size)

# combine generators into one which yields image and masks
train_generator = zip(train_frames_generator, train_masks_generator)

# Fitting the generator to the model
model.fit_generator(
    train_generator,
    steps_per_epoch=(NO_OF_TRAINING_IMAGES//batch_size),
    epochs=epochs,
    callbacks=callbacks)
