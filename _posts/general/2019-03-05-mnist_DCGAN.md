---
layout: article
title: "Making up Numbers - DCGAN and the MNIST dataset"
date: 2019-03-05
categories: general
author: adrian-ball
image:
  teaser: general/MNIST/MnistExamples_teaser.png
  feature: general/MNIST/MnistExamples_banner.png
---

<h2>DRAFT - CURRENTLY IN WRITING</h2> 

Recently I have spent some time learning and playing with Generative Adversarial Networks (GANS). In this article I provide a brief introduction into what a GAN is, and present a GAN that I built to generate numerical images similar to those found in the MNIST database. 

<h5>What is a GAN?</h5> 

WHAT A GAN IS

<h5>My Model</h5> 

To get a better understanding of GANs and how they work, I built a Deep Convolutional GAN (DCGAN) with the objective of training a model to generate numerical images. The ground truth labels that I would be trying to reproduce were the MNIST numbers. I chose to build a DCGAN over something simpler (such as a GAN with a handful of dense layers) as I wanted to give my GPU's a little more work to do. 

Before we dive into a explanation of my model, I would like to reference and acknowledge [Naoki Shibuya](https://towardsdatascience.com/understanding-generative-adversarial-networks-4dafc963f2ef) for providing a strong introduction article that accelerated my understanding of how to build a GAN. While my model is different to that of Naoki, it is similar to a lot of other models presented by others (such as [this](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter4-gan/dcgan-mnist-4.2.1.py) or [this](https://github.com/akashdeepjassal/Keras-DCGAN/blob/master/dcgan.py)). There is a lot of tuning that goes into a GAN, and building a model similar to those developed by others can save a lot of headache. 

Focusing on the model that I built, we first import everything needed to train the model. I built a model using the Keras package wrapped around a GPU variant of TensorFlow. 

{% highlight python %}
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Reshape, UpSampling2D, \
                         Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, SGD

import numpy as np
import matplotlib.pyplot as plt
{% endhighlight %}

asfasdf

{% highlight python %}
def model_generator():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model
{% endhighlight %}

With the generator function defined, we now look to define the discriminator. The role of the discriminator is to determine whether a given input is genuine or fake. The model is a combination of simple `MaxPooling` and `Dense` layers with a single `Conv2D` layer as shown below. The final output of this model is a single node that describes the models belief as to whether the input image was genuine (1) or fake (0). The final activation layer is a sigmoid curve to give us this probabilistic estimate.  

{% highlight python %}
def model_discriminator():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
{% endhighlight %}

With the two sub-models defined, we now import and define the training and test data (in this description I haven't used the test data, though we could pass it through the trained discriminator to understand its performance.). Because we are using `tanh` activation layers, the training data is transformed so that its range lies between [-1,1] (based on tricks listed [here](https://github.com/soumith/ganhacks)).

{% highlight python %}
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = (X_train / 255 - 0.5) * 2 #Want to change from 0-255 to -1,1

#Add dimensions for passing into models (saves a data.reshape() later)
X_train = X_train[:, :, :, None]
X_test = X_test[:, :, :, None]
{% endhighlight %}

{% highlight python %}
plt.figure(figsize=(10, 8))
for i in range(20):
    img = X_train[i,:,:,0]
    plt.subplot(4, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()
{% endhighlight %}

![X_train Images](../../../../images/general/MNIST/DCGAN/output_4_0.png){: .center-image}

{% highlight python %}
gen = model_generator()
disc = model_discriminator()
{% endhighlight %}

{% highlight python %}
gan = Sequential()
gan.add(gen)
disc.trainable = False
gan.add(disc)
{% endhighlight %}

{% highlight python %}
d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
# d_optim = Adam(lr=0.0005)
# g_optim = Adam(lr=0.0005)

gen.compile(loss='binary_crossentropy', optimizer="SGD")
gan.compile(loss='binary_crossentropy', optimizer=g_optim)
disc.trainable = True
disc.compile(loss='binary_crossentropy', optimizer=d_optim)
{% endhighlight %}

{% highlight python %}
BATCH_SIZE = 64

for epoch in range(100):
    print("Epoch:", epoch)
    for index in range(int(X_train.shape[0]/BATCH_SIZE)):
        #noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
        noise = np.random.normal(0, 0.3, size=(BATCH_SIZE, 100))
        image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
        generated_images = gen.predict(noise)
        X = np.concatenate((image_batch, generated_images))
        y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
        d_loss = disc.train_on_batch(X, y)
        #noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100)
        noise = np.random.normal(0, 0.3, size=(BATCH_SIZE, 100))
        disc.trainable = False
        g_loss = gan.train_on_batch(noise, [1] * BATCH_SIZE)
        disc.trainable = True
    print('d_loss : %f' % (d_loss))
    print('g_loss : %f' % (g_loss))
{% endhighlight %}

{% highlight python %}
def generate(BATCH_SIZE, nice=False):
    g = model_generator()
    g.compile(loss='binary_crossentropy', optimizer='SGD')
    g.load_weights('generator')
    #noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
    noise = np.random.normal(0, 0.3, size=(BATCH_SIZE, 100))
    generated_images = g.predict(noise)
    return generated_images
{% endhighlight %}

{% highlight python %}
def generate(BATCH_SIZE, nice=False):
    g = model_generator()
    g.compile(loss='binary_crossentropy', optimizer='SGD')
    g.load_weights('generator')
    #noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
    noise = np.random.normal(0, 0.3, size=(BATCH_SIZE, 100))
    generated_images = g.predict(noise)
    return generated_images
{% endhighlight %}

{% highlight python %}
a = generate(20);
plt.figure(figsize=(10, 8))
for i in range(20):
    img = a[i,:,:,0]
    plt.subplot(4, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()
{% endhighlight %}

![Generated Images](../../../../images/general/MNIST/DCGAN/output_10_1.png){: .center-image}