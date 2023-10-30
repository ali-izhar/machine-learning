import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def generate_data(mean, std, num_samples):
    """Generate random data based on mean, standard deviation, and number of samples."""
    return mean + rand(num_samples) * std

def compute_roc_curve(labels, scores):
    """Compute False Positive Rate, True Positive Rate and ROC AUC."""
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_distribution(positive_data, negative_data):
    """Plot distribution of positive and negative data."""
    sns.distplot(positive_data, hist=False, kde=True, kde_kws = {'shade': True}, label='Covid19 Positive')
    sns.distplot(negative_data, hist=False, kde=True, kde_kws = {'shade': True}, label='Covid19 Negative')
    plt.legend()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot the ROC curve."""
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='navy', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    # Parameters
    num_positives, num_negatives = 1000, 1000
    CovidPositive_mean, CovidPositive_std = 1.5, 1
    CovidNegative_mean, CovidNegative_std = 1, 1
    
    # Generate data
    CovidPositive_output = generate_data(CovidPositive_mean, CovidPositive_std, num_positives)
    CovidNegative_output = generate_data(CovidNegative_mean, CovidNegative_std, num_negatives)

    # Plot distribution
    plot_distribution(CovidPositive_output, CovidNegative_output)
    
    # Prepare data for ROC curve
    labels_positive = [1] * num_positives
    labels_negative = [0] * num_negatives
    test_labels = labels_positive + labels_negative
    test_scores = np.concatenate((CovidPositive_output, CovidNegative_output))
    
    # Compute ROC curve values
    fpr, tpr, roc_auc = compute_roc_curve(test_labels, test_scores)
    
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, roc_auc)
