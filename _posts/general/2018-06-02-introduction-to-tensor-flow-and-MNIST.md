---
layout: article
title: "My First Neural Network, Tensorflow, and the MNIST Dataset"
date: 2018-06-02
categories: general
author: adrian-ball
image:
  teaser: general/MNIST/MnistExamples_teaser.png
  feature: general/MNIST/MnistExamples_banner.png
---

Neural networks have gained a lot of traction in the past years, yet have remained a machine learning technique that I have not had a chance to learn and play with until recently.  One of the most popular applications of this has been the development of [AlphaGo](https://www.nature.com/articles/nature16961), a bot that has repeatedly bested the best humans in the world at the board game Go. There are already several good resources that explain neural networks, such as [here](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/) for a blog, or [here](http://www.deeplearningbook.org/) for an online textbook. In short though, neural networks are a series of layers that perform a linear function on the data from the previous layer and then a non-linear function. These networks are trained with large amounts of data, and are often used for supervised learning problems.

To work on covering this gap in my understanding, I have built a simple neural network model that can recognise hand written digits. This problem is often referred to the "Hello World" of modelling on neural networks. I originally wanted to use neural networks to develop a chat-bot that could learn to speak like others in a chat room, and while that has been done at the time of writing this article, I am starting the documentation of my adventures at the start of my learning process and will be as thorough as I can. In this article I am going to focus more on the application of the neural network to achieve my goal. To do this, a few items need to be presented. 

<h5> Tensorflow</h5>

[Tensorflow](https://www.tensorflow.org/) is an open source software library, originally produced by Google, designed for high performance numerical calculations. This library is appealing as it makes running machine learning processes on the GPU a relatively straight forward process. This is particularly appealing when developing deep learning/neural network models, as a GPU is faster than a CPU and can handle large matrix operations (which form the core of a deep learning model). While there are several libraries that allow for this to happen, the popularity of the Tensorflow library in particular means that there is good community sup

<h5> Docker</h5>

Docker is an application that 'performs operating-system-level virtualization also known as containerization' according to the website. This is the first time that I have worked with Docker, and it feels to me, in slightly more lay terms, like an application that sits somewhere between a virtual machine and a bash script.

The installation of everything in order to get Tensorflow to work was a real nightmare for me. Several packages need to be installed, each of which have several versions, and they have a tendency to not play well together. Docker was suggested to me (Thanks Brandon!) as an option for setting up an 'image' where all kinks in driver configuration have already been sorted, which means I can now have the desired environment in essentially one command now.

<h5> MNIST </h5>

The [MNIST data set](http://yann.lecun.com/exdb/mnist/) is a set of hand written digits split into a training and test groups. The data has been normalised such that each image contains only one number, and each image is the same size of 28*28 pixels. This makes the data set attractive for testing new machine learning techniques as data pre-processing and cleaning steps have already been performed. 

<h5> Writing the code </h5>

This code was written (with a few minor variations, like adding another layer) while following a tutorial that was presented [here](https://pythonprogramming.net/tensorflow-introduction-machine-learning-tutorial/). The linked tutorial is highly detailed, and is presented both in text and as a video. For those looking to get started, this was a fantastic source from an application perspective.

There are two main sections required to get this to work, defining and training the model. Again, steps like data pre-processing are not needed here as the data set has been cleaned and normalised for us. 

{% highlight python %}

def nn_model(data):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784, nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([nodes_hl1]))}

    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([nodes_hl2]))}

    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([nodes_hl3]))}

    hidden_layer_4 = {'weights':tf.Variable(tf.random_normal([nodes_hl3, nodes_hl4])),
                      'biases':tf.Variable(tf.random_normal([nodes_hl4]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([nodes_hl4, num_classes])),
                    'biases':tf.Variable(tf.random_normal([num_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3,hidden_layer_4['weights']), hidden_layer_4['biases'])
    l4 = tf.nn.relu(l3)

    output = tf.matmul(l4,output_layer['weights']) + output_layer['biases']

    return output

{% endhighlight %}


{% highlight python %}

def train_neural_network(x):
    prediction = nn_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    num_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):    
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',num_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

        #The above accuracy spits the dummy and complains of running out of memory, so we 
        #assess the accuracy in batches. We use the same batch size as training for simplicity.
        #c/o - https://github.com/tensorflow/tensorflow/issues/136
        batch_num = int(mnist.test.num_examples / batch_size)
        test_accuracy = 0
    
        for i in range(batch_num):
            batch = mnist.test.next_batch(batch_size)
            test_accuracy += accuracy.eval(feed_dict={x: batch[0],
                                              y: batch[1]})
                                              #keep_prob: 1.0})

        test_accuracy /= batch_num
        print("test accuracy %g"%test_accuracy)

{% endhighlight %}