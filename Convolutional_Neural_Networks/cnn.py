# -*- coding: utf-8 -*-
# Importing Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

# Initialize CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

# Step 3- Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
#                   # Neurons in layer, Initialize weights, Rectifier phi func

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Image Augmentation and Fitting Model (Randomly Rotating, Flipping etc.. of images and Training model
# on batches of images to Prevent Overfitting)
from keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage
trainDatagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

testDatagen = ImageDataGenerator(rescale=1./255)

xTrain = trainDatagen.flow_from_directory(
        'dataset/training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode='binary'
)
xTest = testDatagen.flow_from_directory(
        'dataset/test_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode='binary'
)

classifier.fit_generator(
        xTrain,
        steps_per_epoch = 8000,
        epochs = 25,
        validation_data = xTest,
        validation_steps = 2000)