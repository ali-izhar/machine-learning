# Decision Trees Learning
A decision tree is a machine learning model that makes decisions based on a sequence of questions asked about the features of the data. These models are particularly useful for classification tasks, such as identifying whether an animal is a cat or a dog based on specific features.

## Building a Decision Tree
Building a decision tree involves several key steps:

- **Choosing the feature to split on:** The first step in building a decision tree is deciding which feature to use at the root node, i.e., the first node at the top of the tree. This is done using a specific algorithm that maximizes the purity of the nodes.

- **Creating branches based on feature values:** Once a feature is chosen, the training examples are divided based on the value of that feature. For example, if we choose 'ear shape' as the feature, the training examples will be divided into two groups: one group with pointy ears and the other group with floppy ears.

- **Continuing the process on each branch:** This process is then repeated for each branch of the tree. A new feature is chosen, and the training examples in that branch are further divided based on the value of the new feature.

- **Creating leaf nodes for predictions:** Once a subset of the examples in a branch belongs to a single class (e.g., all cats or all dogs), a leaf node is created that makes a prediction for that class.

## Key Decisions in Building a Decision Tree
There are two main decisions to make when building a decision tree:

- **How to choose the feature to split on:** The goal when choosing a feature is to maximize the `purity` of the nodes, i.e., make the subsets of examples as close as possible to belonging to a single class (either all cats or all dogs in our example).

- **When to stop splitting:** The criteria for stopping the splitting process could be when all examples in a node belong to the same class or when the depth of the tree reaches a certain limit. This is done to prevent the tree from becoming too large and to avoid overfitting.