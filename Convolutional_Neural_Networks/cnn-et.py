#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convolutional Neural Network

Created on Wed Jul 11 16:39:09 2018

@author: edward
"""

# Code to use GPU (not from class)
from datetime import datetime
from keras import backend as K
K.tf.logging.set_verbosity(K.tf.logging.INFO)
with K.tf.device('/gpu:0'):

    # Part 1 - Building the CNN
    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
    
    # Initializing the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape=(64,64,3), activation = 'relu'))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])
    
    # Part 2 - Fitting the CNN to the images
    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')
    
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

    # Code to use GPU (not from class)
    startTime = datetime.now()
    with K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        device = K.tensorflow_backend._get_current_tf_device()
        print(device)
        classifier.fit_generator(training_set,
                                 steps_per_epoch=2000,
                                 epochs=1,
                                 validation_data=test_set,
                                 validation_steps=400)

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)

print("Time taken:", datetime.now() - startTime)

print("\n" * 5)

#import tensorflow as tf
#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session()
#print(sess.run(hello))