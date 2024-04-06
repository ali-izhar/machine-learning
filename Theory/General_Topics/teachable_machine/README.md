# Teachable Machine
[Teachable Machine](https://teachablemachine.withgoogle.com/) is a web-based tool that makes creating machine learning models fast, easy, and accessible to everyone. It allows you to train a model in your browser without writing any code. It is based on the [TensorFlow.js](https://www.tensorflow.org/js) library.

# Model
The model is trained to classify images of dogs into four classes. The four classes are:
1. chihuahua
2. japanese_spaniel
3. maltese_dog
4. pekinese

# Dataset
The dataset is a subset of the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). The training set contains 27 images per class (108 training images in total).

# Training
The model is trained using the following parameters:
- Epochs = 150 (50 default)
- Batch Size = 32 (16 default)
- Learning rate = 0.001