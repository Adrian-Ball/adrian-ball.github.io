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

x_train = np.reshape(x_train,(x_train.shape[0],28*28))/255;
x_test = np.reshape(x_test,(x_test.shape[0],28*28))/255;

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
{% endhighlight %}

Now that we have the data, the model and its hyper-parameters can be defined. As mentioned before, the hyper-parameters are set to be the same as those of the previously described Tensorflow article.

As the middle layers are identical to each other, it is simple to define them through the use of a for loop. To my understanding, this is something that is not possible in Tensorflow (let alone possible in a clean and clear manner). What makes it possible to define the layers using a loop in Keras is that the size of the input for each layer is automatically inferred from the output of the previous layer. Only the first of the dense layers requires specific input shape definition. This is because the input data has not yet been defined. 

When originally writing up this script up, I was obtaining terrible accuracies (in the order of 9%). Aside from a slip-up where I used the wrong activation function for my final layer (sigmoid instead of soft-max), I also missed the normalisation of my input data. The images are greyscale, and should have a value between 0 and 1, yet I had left them in the range of 0 to 255. It is amazing the difference that missing something like this has! I must thank those that had a look to find my mistake, particularly snakeand1 from [this reddit post](https://www.reddit.com/r/datascience/comments/9w2ozl/model_difference_between_keras_and_native/).

The other noteworthy feature here is that the bias for each of the dense layers do not have to be explicitly defined. 


{% highlight python %}
batch_size = 100
nodes_per_layer = 256
num_epochs = 20

model = Sequential()
model.add(Dense(nodes_per_layer, input_shape=(784,),activation='relu'))
for num_layers in range(0,3):
    model.add(Dense(nodes_per_layer,activation='relu'))
    
model.add(Dense(10,activation='softmax'))
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
    60000/60000 [==============================] - 4s 70us/step - loss: 0.2390 - acc: 0.9286
    Epoch 2/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0922 - acc: 0.9716
    Epoch 3/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0662 - acc: 0.9798
    Epoch 4/20
    60000/60000 [==============================] - 4s 66us/step - loss: 0.0513 - acc: 0.9839
    Epoch 5/20
    60000/60000 [==============================] - 4s 66us/step - loss: 0.0409 - acc: 0.9870
    Epoch 6/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0343 - acc: 0.9886
    Epoch 7/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0318 - acc: 0.9900
    Epoch 8/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0272 - acc: 0.9911
    Epoch 9/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0233 - acc: 0.9926
    Epoch 10/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0217 - acc: 0.9935
    Epoch 11/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0198 - acc: 0.9938
    Epoch 12/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0172 - acc: 0.9948
    Epoch 13/20
    60000/60000 [==============================] - 4s 67us/step - loss: 0.0166 - acc: 0.9949
    Epoch 14/20
    60000/60000 [==============================] - 4s 66us/step - loss: 0.0141 - acc: 0.9955
    Epoch 15/20
    60000/60000 [==============================] - 4s 66us/step - loss: 0.0180 - acc: 0.9945
    Epoch 16/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0142 - acc: 0.9957
    Epoch 17/20
    60000/60000 [==============================] - 4s 67us/step - loss: 0.0118 - acc: 0.9964
    Epoch 18/20
    60000/60000 [==============================] - 4s 66us/step - loss: 0.0132 - acc: 0.9960
    Epoch 19/20
    60000/60000 [==============================] - 4s 66us/step - loss: 0.0121 - acc: 0.9962
    Epoch 20/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0109 - acc: 0.9968


It looks like the model will have a high accuracy, though there is the chance of overfitting. The model is evaluated against the test data, and as we can see from below, a good accuracy is obtained. The combination of a drop in the accuracy, and the high accuracy obtained when learning suggest that some over-fitting might be happening. In this instance I am not too worried, as the objective was to draw a basic comparison between native Tensorflow and Keras.

{% highlight python %}
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Model accuracy is', score[1])
{% endhighlight %}

    10000/10000 [==============================] - 0s 30us/step
    Model accuracy is 0.979100008011


In conclusion, it was significantly easier to get a model up and running using Keras rather than native Tensorflow. The significant difference between the two models was the clarity and conciseness of the code when working in Keras (plus Keras also has fantastic documentation). Not only does this make is easier to conceptualise and understand what is happening with the model, but the ease of understanding also helps to speed up model design iteration. Though I also want to look at this model again in Julia, for now all standard neural net models that I generate will be done through Keras and not native Tensorflow. 
