from django.shortcuts import render
from .models import Data
#import gan_dc
import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D , Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import os
import re
import numpy as np
from glob import glob
from PIL import Image
import random
import matplotlib.pyplot as plt

IMAGE_SIZE = 128
NOISE_SIZE = 100
LR_D = 0.00004
LR_G = 0.0004
BATCH_SIZE = 64
EPOCHS = 100
BETA1 = 0.5
WEIGHT_INIT_STDDEV = 0.02
EPSILON = 0.00005
SAMPLES_TO_SHOW = 5
adam1 = Adam(lr=LR_G,beta_1=0.5)
adam2 = Adam(lr=LR_D,beta_1=0.5)
HALF_BATCH = 32

def loading_weights():
    discriminator = Sequential()

    discriminator.add(Conv2D(32 , (5,5) , strides=(2,2) , padding='same', input_shape=(128,128,3) ,
                            kernel_initializer = keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    discriminator.add(BatchNormalization(momentum = 0.9 , epsilon = EPSILON))
    discriminator.add(LeakyReLU(alpha = 0.2))
    
    discriminator.add(Conv2D(64 , (5,5) , strides=(2,2) , padding='same' ,
                            kernel_initializer = keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    discriminator.add(BatchNormalization(momentum = 0.9 , epsilon = EPSILON))
    discriminator.add(LeakyReLU(alpha = 0.2))
  
    discriminator.add(Conv2D(128 , (5,5) , strides=(2,2) , padding='same' , 
                            kernel_initializer = keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    discriminator.add(BatchNormalization(momentum = 0.9 , epsilon = EPSILON))
    discriminator.add(LeakyReLU(alpha = 0.2))
    
    discriminator.add(Conv2D(256 , (5,5) , strides=(1,1) , padding='same' ,
                            kernel_initializer = keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    discriminator.add(BatchNormalization(momentum = 0.9 , epsilon = EPSILON))
    discriminator.add(LeakyReLU(alpha = 0.2))
  
    discriminator.add(Conv2D(512 , (5,5) , strides=(2,2) , padding='same' ,
                            kernel_initializer = keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    discriminator.add(BatchNormalization(momentum = 0.9 , epsilon = EPSILON))
    discriminator.add(LeakyReLU(alpha = 0.2))
    
    discriminator.add(Flatten())
    
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.load_weights('./generator/Weights/discriminator100.h5')
    
    generator = Sequential()
    generator.add(Dense(4*4*512 , activation='relu' ,input_shape = (NOISE_SIZE,)))
    generator.add(LeakyReLU(alpha = 0.2))
    generator.add(Reshape((4,4,512)))
    
    generator.add(Conv2DTranspose(512 , kernel_size=(5,5) , strides=(2,2) , padding='same' , 
                                  kernel_initializer = keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    generator.add(BatchNormalization(momentum=0.9 , epsilon=EPSILON))
    generator.add(LeakyReLU(alpha = 0.2))
    
    generator.add(Conv2DTranspose(256 , kernel_size=(5,5) , strides=(2,2) , padding='same' , 
                                  kernel_initializer = keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    generator.add(BatchNormalization(momentum=0.9 , epsilon=EPSILON))
    generator.add(LeakyReLU(alpha = 0.2))
    
    generator.add(Conv2DTranspose(128 , kernel_size=(5,5) , strides=(2,2) , padding='same' , 
                                  kernel_initializer = keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    generator.add(BatchNormalization(momentum=0.9 , epsilon=EPSILON))
    generator.add(LeakyReLU(alpha = 0.2))
    
    generator.add(Conv2DTranspose(64 , kernel_size=(5,5) , strides=(2,2) , padding='same' , 
                                  kernel_initializer = keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    generator.add(BatchNormalization(momentum=0.9 , epsilon=EPSILON))
    generator.add(LeakyReLU(alpha = 0.2))
    
    generator.add(Conv2DTranspose(32 , kernel_size=(5,5) , strides=(2,2) , padding='same' , 
                                  kernel_initializer = keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    generator.add(BatchNormalization(momentum=0.9 , epsilon=EPSILON))
    generator.add(LeakyReLU(alpha = 0.2))
    
    generator.add(Conv2DTranspose(3 , kernel_size=(5,5) , strides=(1,1) , padding='same' , 
                                  kernel_initializer = keras.initializers.TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    
    generator.add(Activation("tanh"))
    generator.load_weights('./generator/Weights/generator100.h5')

    discriminator.compile(optimizer=adam2 , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    discriminator.trainable = False
    gan_input = Input(shape=(NOISE_SIZE,))
    generated_img = generator(gan_input)
    gan_output = discriminator(generated_img)

    model = Model(gan_input,gan_output)

    model.compile(loss='binary_crossentropy',optimizer=adam1)
    
    return generator , discriminator , model

def save_imgs(generator , discriminator , samples=100):
    
    noise = np.random.normal(0,1,size=(samples,NOISE_SIZE))
    generated_imgs = generator.predict(noise)
    generated_imgs = generated_imgs.reshape(samples,128,128,3)
    
    
    for i in range(samples):
        plt.figure(figsize=(4,4))
        plt.imshow(generated_imgs[i],interpolation='nearest',cmap='gray')
        plt.axis("off")
        plt.savefig('./generator/static/generator/images/{0}.png'.format(i))
        #plt.show()
        #print(os.getcwd())
generator , discriminator , model=loading_weights()
save_imgs(generator , discriminator , 50)


def home(request):
    
    
    return render(request, 'generator/home.html')

def about(request):
    return render(request, 'generator/about.html', {'title':'About'})

def images(request):
    return render(request,  'generator/images.html', {'title':'Images'})

def aboutproject(request):
    return render(request, 'generator/aboutproject.html', {'title':'About Project'})

def future(request):
    return render(request, 'generator/future.html', {'title':'Future'})

def aboutdevelopers(request):
    return render(request, 'generator/aboutdevelopers.html', {'title':'About Developers'})