# Classification
Classification tasks in Machine Learning can be categorized into:

- **binary classification**
- **multiclass classification**
- **multilabel classification**

## Multiclass Classification
In multiclass classification, the models are designed to predict one single class label from multiple possible classes. Each instance in the training set belongs to one and only one class. A common example of multiclass classification is handwritten digit recognition, where an image of a handwritten digit can belong to one class from zero to nine (10 classes in total).

## Multilabel Classification
Multilabel classification is a variant of classification where each instance can be associated with multiple classes simultaneously. In other words, it's not required for the classes to be mutually exclusive. For example, consider a street view image that contains multiple objects: pedestrians, cars, traffic lights, etc. A multilabel classifier trained on this image may predict three labels for it: [pedestrian, car, traffic light].

- If an image contains a pedestrian, a car, and a traffic light, the classifier will output [1, 1, 1].
- If an image contains only a car and a traffic light, the classifier will output [0, 1, 1].
- If an image contains only a pedestrian, the classifier will output [1, 0, 0].

## Key Differences
-**Class Exclusivity:** In multiclass classification, each instance belongs to one and only one class, while in multilabel classification, an instance can belong to multiple classes simultaneously.

-**Output Vector:** In multiclass classification, the output is a single class label. However, in multilabel classification, the output is a binary vector that indicates the class membership.

-**Loss Function:** In multiclass classification, the loss function is typically calculated considering the difference between the predicted class and the actual class. However, in multilabel classification, the loss is calculated for each class label independently, and then summed up (or averaged) to create a total loss.

-**Problem Complexity:** Multilabel classification is generally more complex than multiclass classification due to the increase in the number of possible label combinations.