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