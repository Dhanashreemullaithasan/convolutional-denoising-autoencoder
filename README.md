# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

## Convolution Autoencoder Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

### STEP 2:

### STEP 3:

Write your own steps

## PROGRAM
### Name:DHANASHREE M
### Register Number: 212221230018
```
import pandas as pd
from tensorflow import keras 
from keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train.shape

x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape)
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))

Input_image =layers.Input(shape=(28,28,1))
y=layers.Conv2D(16,(3,3),activation='relu',padding='same')(Input_image)
y=layers.MaxPooling2D((2,2),padding='same')(y)
y=layers.Conv2D(8,(3,3),activation='relu',padding='same')(y)
y=layers.MaxPooling2D((2,2),padding='same')(y)
y=layers.Conv2D(8,(3,3),activation='relu',padding='same')(y)
encoder_output=layers.MaxPooling2D((2,2),padding='same')(y)

x= layers.Conv2D(8,(3,3),activation='relu',padding='same')(encoder_output)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(3,3),activation='relu')(x)
x=layers.UpSampling2D((2,2))(x)
decoder_output=layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

autoencoder=keras.Model(Input_image,decoder_output)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
print("Name: DHANASHREE M, Reg no: 212221230018")
metrics = pd.DataFrame(autoencoder.history.history)
metrics[['loss','val_loss']].plot()

decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
print("Name: Dhanashree M , Reg NO : 212221230018")
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-10-14 144759](https://github.com/user-attachments/assets/963b2d63-48ed-4233-92fb-6f61dfe22945)


### Original vs Noisy Vs Reconstructed Image

![Screenshot 2024-10-14 144827](https://github.com/user-attachments/assets/531c38dd-599a-4270-ae10-11abf340cd56)




## RESULT
