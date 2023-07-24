# Model Selection and Evaluation

## Introduction
Choosing the right machine learning model and evaluating its performance are crucial steps in applying ML to real-world problems. 

## Training, Validation, and Test Sets
Best practice is to split data into 3 sets:

- **Training set:** Used to fit the machine learning model parameters. We want the model to learn from these examples.
- **Validation set:** Used to evaluate model performance during training and select the best model. Helps prevent overfitting to the training data.
- **Test set:** Used to provide an unbiased evaluation of the final model. Since this data is never used for model selection or hyperparameter tuning, it gives a true estimate of model performance.

```python
data = load_data() 

# 60% for training
train_set = data[:60%] 

# 20% for validation
val_set = data[60%:80%]

# 20% for testing  
test_set = data[80%:]
```

## Model Selection
We train several candidate models on the training set:

```python
# Train 3 different polynomial models
model1 = train_polynomial(train_set, degree=1) 
model2 = train_polynomial(train_set, degree=3)
model3 = train_polynomial(train_set, degree=5)
```

Then we evaluate them on the validation set:

```python
# Evaluate models on validation data
val_error1 = evaluate(model1, val_set)
val_error2 = evaluate(model2, val_set) 
val_error3 = evaluate(model3, val_set)
```

We select the best performing model based on validation error:

```python
# Pick model with lowest validation error
best_model = model2
```

## Model Evaluation
Finally, we evaluate the selected model on the test set to estimate generalization performance:
    
```python
# Evaluate selected model on test set 
test_error = evaluate(best_model, test_set)

print('Expected test error: ', test_error)
```

Since the test set was not used during training or model selection, it provides an unbiased estimate of how the model will perform on new data.

## Key Points
- Training set: Used for learning model parameters
- Validation set: Used for model selection
- Test set: Used for evaluation of final model
- Avoid overfitting by keeping test set fully isolated