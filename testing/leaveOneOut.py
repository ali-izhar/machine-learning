import random


def getStats(truePos, falsePos, trueNeg, falseNeg):
    """truePos, falsePos, trueNeg, falseNeg are counts"""
    try:
        accuracy = (truePos + trueNeg)/(truePos + falsePos + trueNeg + falseNeg)
        precision = truePos/(truePos + falsePos)
        recall = truePos/(truePos + falseNeg)
        fscore = 2 * precision * recall/(precision + recall)
        print('Accuracy =', round(accuracy, 4))
        print('Precision =', round(precision, 4))
        print('Recall =', round(recall, 4))
        print('F-score =', round(fscore, 4))
    except ZeroDivisionError:
        print('There were no positive predictions')


def leaveOneOut(examples, method, toPrint=True):
    """
    Leave-one-out cross-validation is a special case of cross-validation in which the
    number of folds is equal to the number of examples. In other words, we create
    N folds, each of which contains a single example. We then train on N-1 examples
    and test on the one example in the fold we left out. We do this N times, each time
    leaving out a different example. This is a very expensive procedure, but it is useful
    when you have a small number of examples and you want to get the most out of each one.

    Inputs:
        examples: list of examples to be divided into training and testing sets
        method: function that takes as input a training set and a testing set and returns
                a tuple of the form (truePos, falsePos, trueNeg, falseNeg)
        toPrint: boolean that indicates whether or not to print the statistics

    Returns:
        truePos, falsePos, trueNeg, falseNeg
    """
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i in range(len(examples)):
        testCase = examples[i]
        trainingData = examples[:i] + examples[i+1:]
        results = method(trainingData, [testCase])
        truePos += results[0]
        falsePos += results[1]
        trueNeg += results[2]
        falseNeg += results[3]
    if toPrint:
        getStats(truePos, falsePos, trueNeg, falseNeg)
    return truePos, falsePos, trueNeg, falseNeg


def randomSplits(examples, method, numSplits, toPrint=True):
    """
    Random splits cross-validation is a less expensive alternative to leave-one-out
    cross-validation. In this case, we randomly divide the examples into N folds, and
    then train on N-1 folds and test on the remaining fold. We do this N times, each
    time using a different fold for testing. This is a good choice when you have a large
    number of examples.

    Inputs:
        examples: list of examples to be divided into training and testing sets
        method: function that takes as input a training set and a testing set and returns
                a tuple of the form (truePos, falsePos, trueNeg, falseNeg)
        numSplits: number of training-testing splits to create
        toPrint: boolean that indicates whether or not to print the statistics

    Returns:
        truePos, falsePos, trueNeg, falseNeg
    """
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    random.seed(0)
    for t in range(numSplits):
        trainingSet, testSet = split80_20(examples)
        results = method(trainingSet, testSet)
        truePos += results[0]
        falsePos += results[1]
        trueNeg += results[2]
        falseNeg += results[3]
    if toPrint:
        getStats(truePos/numSplits, falsePos/numSplits, trueNeg/numSplits, falseNeg/numSplits)
    return truePos/numSplits, falsePos/numSplits, trueNeg/numSplits, falseNeg/numSplits


def split80_20(examples):
    sampleIndices = random.sample(range(len(examples)), len(examples)//5)
    trainingSet, testSet = [], []
    for i in range(len(examples)):
        if i in sampleIndices:
            testSet.append(examples[i])
        else:
            trainingSet.append(examples[i])
    return trainingSet, testSet