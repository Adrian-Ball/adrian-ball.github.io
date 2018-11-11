---
layout: article
title: "Keras and the MNIST Dataset"
date: 2018-11-11
categories: general
author: adrian-ball
image:
  teaser: general/MNIST/MnistExamples_teaser.png
  feature: general/MNIST/MnistExamples_banner.png
---

In this article, I repeat my previous attempt at training a model to recognise the MNIST hadn-written digits. The difference between these two attempts is that [previously](../introduction-to-tensor-flow-and-MNIST/) native Tensorflow was used, and in this scenario the Tensorflow library is accessed through the Keras wrapper. This has resulted in a much clearer and more concise model definition, making it easier to maintain an understanding of everything that happens. 

Furthermore, this is the first article for this site that I have written using Jupyter Notebooks. While I have used Jupyter Notebooks in the past, this is the first time that I have exported a page as markdown file and then incorporated it into a website. Minimal work was necessary after the export for integrating it into the website and making it stylistically consistent with other content. For those unaware, notebooks are definitely something that you should look into!

Jumping straight into the Keras implementation of the model, we import everything necessary for building the model. 


{% highlight python %}
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
{% endhighlight %}

To start, we acquire the MNIST data and slightly modify it so that it is compatible with the model. As we are looking to make a simple fully connected dense layer neural network, the input data is reshaped (flattened) so that it is one dimensional. The labels are also converted into a one-hot configuration. 

This model type was chosen so that a comparison can be made against the model previously implemented in Tensorflow. 


{% highlight python %}
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train,(x_train.shape[0],28*28));
x_test = np.reshape(x_test,(x_test.shape[0],28*28));

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
{% endhighlight %}

Now that we have the data, the model and its hyper-parameters can be defined. As mentioned before, the hyper-parameters are set to be the same as those of the previously described Tensorflow article.

As the middle layers are identical to each other, it is simple to define them through the use of a for loop. To my understanding, this is something that is not possible in Tensorflow (let alone possible in a clean and clear manner). What makes it possible to define the layers using a loop in Keras is that the size of the input for each layer is automatically inferred from the output of the previous layer. Only the first of the dense layers requires specific input shape definition. This is because the input data has not yet been defined. 

The other noteworthy feature here is that the bias for each of the dense layers do not have to be explicitly defined. 


{% highlight python %}
batch_size = 100
nodes_per_layer = 256
num_epochs = 20

model = Sequential()
model.add(Dense(nodes_per_layer, input_shape=(784,),activation='relu'))
for num_layers in range(0,3):
    model.add(Dense(nodes_per_layer,activation='relu'))
    
model.add(Dense(10,activation='sigmoid'))
{% endhighlight %}

With the model now defined, we can now compile it. Again, the Adam optimiser is used and we are trying to minimise the cross entropy loss.


{% highlight python %}
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
{% endhighlight %}

With the model compiled, training data and hyper-parameters are provided for training. 


{% highlight python %}
model.fit(x_train, y_train,
          epochs=num_epochs,
          batch_size=batch_size);
{% endhighlight %}

    Epoch 1/20
    60000/60000 [==============================] - 4s 70us/step - loss: 2.3257 - acc: 0.1012
    Epoch 2/20
    60000/60000 [==============================] - 4s 65us/step - loss: 2.3026 - acc: 0.0987
    Epoch 3/20
    60000/60000 [==============================] - 4s 64us/step - loss: 2.3026 - acc: 0.0987
    Epoch 4/20
    60000/60000 [==============================] - 4s 64us/step - loss: 2.3026 - acc: 0.0987
    Epoch 5/20
    60000/60000 [==============================] - 4s 65us/step - loss: 2.3026 - acc: 0.0987
    Epoch 6/20
    60000/60000 [==============================] - 4s 65us/step - loss: 2.3026 - acc: 0.0987
    Epoch 7/20
    60000/60000 [==============================] - 4s 65us/step - loss: 2.3026 - acc: 0.0987
    Epoch 8/20
    60000/60000 [==============================] - 4s 65us/step - loss: 2.3026 - acc: 0.0987
    Epoch 9/20
    60000/60000 [==============================] - 4s 65us/step - loss: 2.3026 - acc: 0.0987
    Epoch 10/20
    60000/60000 [==============================] - 4s 65us/step - loss: 2.3026 - acc: 0.0987
    Epoch 11/20
    60000/60000 [==============================] - 4s 66us/step - loss: 2.3026 - acc: 0.0987
    Epoch 12/20
    60000/60000 [==============================] - 4s 67us/step - loss: 2.3026 - acc: 0.0987
    Epoch 13/20
    60000/60000 [==============================] - 4s 64us/step - loss: 2.3026 - acc: 0.0987
    Epoch 14/20
    60000/60000 [==============================] - 4s 65us/step - loss: 2.3026 - acc: 0.0987
    Epoch 15/20
    60000/60000 [==============================] - 4s 66us/step - loss: 2.3026 - acc: 0.0987
    Epoch 16/20
    60000/60000 [==============================] - 4s 68us/step - loss: 2.3026 - acc: 0.0987
    Epoch 17/20
    60000/60000 [==============================] - 4s 65us/step - loss: 2.3026 - acc: 0.0987
    Epoch 18/20
    60000/60000 [==============================] - 4s 65us/step - loss: 2.3026 - acc: 0.0987
    Epoch 19/20
    60000/60000 [==============================] - 4s 65us/step - loss: 2.3026 - acc: 0.0987
    Epoch 20/20
    60000/60000 [==============================] - 4s 67us/step - loss: 2.3026 - acc: 0.0987


It looks like the perfomance of the model was terrible! Especially when the previous accuracy using Tensorflow was 95%. I suspect the reason for this difference is that there are subtle differences that arise from implicitly defined Keras settings when it calls Tensorflow functions, likely causing overfitting to happen.

To confirm that the model performed poorly, the model is evaluated using the test data. 


{% highlight python %}
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Model accuracy is', score[1])
{% endhighlight %}

    10000/10000 [==============================] - 0s 20us/step
    Model accuracy is 0.0979999999329


With my suspicions confirmed, I have reduced the number of neurons present in each of the hidden layers in an attempt to improve the model. 


{% highlight python %}
batch_size = 100
nodes_per_layer = 32
num_epochs = 20

model = Sequential()
model.add(Dense(nodes_per_layer, input_shape=(784,),activation='relu'))
for num_layers in range(1,4):
    model.add(Dense(nodes_per_layer,activation='relu'))
    
model.add(Dense(10,activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=num_epochs,
          batch_size=batch_size);

print('Evaluating the model...')
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Model accuracy is', score[1])
{% endhighlight %}

    Epoch 1/20
    60000/60000 [==============================] - 2s 32us/step - loss: 3.4618 - acc: 0.1019
    Epoch 2/20
    60000/60000 [==============================] - 1s 25us/step - loss: 2.9699 - acc: 0.2990
    Epoch 3/20
    60000/60000 [==============================] - 1s 24us/step - loss: 2.4736 - acc: 0.5833
    Epoch 4/20
    60000/60000 [==============================] - 1s 24us/step - loss: 2.2974 - acc: 0.6530
    Epoch 5/20
    60000/60000 [==============================] - 2s 25us/step - loss: 2.0499 - acc: 0.7573
    Epoch 6/20
    60000/60000 [==============================] - 1s 25us/step - loss: 1.0741 - acc: 0.8589
    Epoch 7/20
    60000/60000 [==============================] - 2s 25us/step - loss: 0.2415 - acc: 0.9346
    Epoch 8/20
    60000/60000 [==============================] - 1s 25us/step - loss: 0.1930 - acc: 0.9453
    Epoch 9/20
    60000/60000 [==============================] - 1s 25us/step - loss: 0.1646 - acc: 0.9517
    Epoch 10/20
    60000/60000 [==============================] - 2s 25us/step - loss: 0.1475 - acc: 0.9567
    Epoch 11/20
    60000/60000 [==============================] - 2s 25us/step - loss: 0.1294 - acc: 0.9599
    Epoch 12/20
    60000/60000 [==============================] - 2s 25us/step - loss: 0.1242 - acc: 0.9626
    Epoch 13/20
    60000/60000 [==============================] - 1s 25us/step - loss: 0.1179 - acc: 0.9640
    Epoch 14/20
    60000/60000 [==============================] - 1s 25us/step - loss: 0.1100 - acc: 0.9669
    Epoch 15/20
    60000/60000 [==============================] - 2s 25us/step - loss: 0.1037 - acc: 0.9678
    Epoch 16/20
    60000/60000 [==============================] - 2s 26us/step - loss: 0.0996 - acc: 0.9693
    Epoch 17/20
    60000/60000 [==============================] - 1s 25us/step - loss: 0.0941 - acc: 0.9706
    Epoch 18/20
    60000/60000 [==============================] - 1s 25us/step - loss: 0.0924 - acc: 0.9717
    Epoch 19/20
    60000/60000 [==============================] - 1s 24us/step - loss: 0.0850 - acc: 0.9735
    Epoch 20/20
    60000/60000 [==============================] - 1s 25us/step - loss: 0.0836 - acc: 0.9748
    Evaluating the model...
    10000/10000 [==============================] - 0s 32us/step
    Model accuracy is 0.961600005031


Through some iteration, I was able to find a value for the number of neurons in each layer such that a suitable model was trained. To achieve this, I just halved the number of nodes and retrained the model. The model presented above was the final model that I settled on. This model had an accuracy of 96% on the test data, an accuracy similar to the model learned through Tensorflow.

In comparison with using Tensorflow directly, it was significantly easier to get a model up and running using Keras as a wrapper. Equally significant was the clarity and conciseness of the code when working in Keras. I was a little surprised to see that the same model design with the same hyper-parameter values yielded two completely different results when the model was trained in native Tensorflow or with Tensorflow through the use of Keras.
