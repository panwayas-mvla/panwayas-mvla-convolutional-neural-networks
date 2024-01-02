#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img

base_dir = "./data/"
happy_dir = os.path.join(base_dir, "happy/")
sad_dir = os.path.join(base_dir, "sad/")

print("Sample happy image:")
plt.imshow(load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}"))
plt.show()

print("\nSample sad image:")
plt.imshow(load_img(f"{os.path.join(sad_dir, os.listdir(sad_dir)[0])}"))
plt.show()

from tensorflow.keras.preprocessing.image import img_to_array

# Load the first example of a happy face
sample_image  = load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}")

# Convert the image into its numpy array representation
sample_array = img_to_array(sample_image)

print(f"Each image has shape: {sample_array.shape}")

print(f"The maximum pixel value used is: {np.max(sample_array)}")


# ## Defining the callback
# 

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True    

# ## Pre-processing the data
# 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# GRADED FUNCTION: image_generator
def image_generator():
    data_generator = ImageDataGenerator(rescale=1.0/255.0 )  
    train_generator = data_generator.flow_from_directory(directory='./data/',
                                                        target_size=(150, 150),
                                                        batch_size=10,
                                                        class_mode='binary')
    return train_generator
gen = image_generator()

from tensorflow.keras import optimizers, losses

def train_happy_sad_model(train_generator):

    # Instantiate the callback
    callbacks = myCallback()

    # Define the model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
        
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
        
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
               
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add dropout to reduce overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop
    model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(lr=0.001),
                 metrics =['accuracy'])     


    history = model.fit(x = train_generator,
                        epochs=20,
                        steps_per_epoch = 2,
                        verbose = 1, 
                        callbacks=[callbacks]
                       )
    
    return history

hist = train_happy_sad_model(gen)

print(f"Your model reached the desired accuracy after {len(hist.epoch)} epochs")


if not "accuracy" in hist.model.metrics_names:
    print("Use 'accuracy' as metric when compiling your model.")
else:
    print("The metric was correctly defined.")
