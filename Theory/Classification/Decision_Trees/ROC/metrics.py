import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(cm):
    """Compute various metrics from the confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    informedness = recall + specificity - 1
    markedness = ppv + npv - 1
    
    return specificity, fpr, fnr, ppv, npv, mcc, informedness, markedness

# Load the breast cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=10000)  # Increase max_iter to ensure convergence
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
specificity, fpr, fnr, ppv, npv, mcc, informedness, markedness = compute_metrics(cm)

# Print the metrics
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Specificity:", specificity)
print("False Positive Rate (FPR):", fpr)
print("False Negative Rate (FNR):", fnr)
print("Positive Predictive Value (PPV):", ppv)
print("Negative Predictive Value (NPV):", npv)
print("Matthews Correlation Coefficient (MCC):", mcc)
print("Informedness (Youden's J statistic):", informedness)
print("Markedness:", markedness)

# Visualizing the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


"""
Confusion Matrix:
[[ 92   6]
 [  8 179]]
Accuracy: 0.9508771929824561
Precision: 0.9512110004741584
Recall: 0.9508771929824561
F1 Score: 0.9509932374108945
Specificity: 0.9387755102040817
False Positive Rate (FPR): 0.061224489795918366
False Negative Rate (FNR): 0.0427807486631016
Positive Predictive Value (PPV): 0.9675675675675676
Negative Predictive Value (NPV): 0.92
Matthews Correlation Coefficient (MCC): 0.8917712100388813
Informedness (Youden's J statistic): 0.8896527031865378
Markedness: 0.8875675675675676
"""