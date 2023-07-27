# Human-level Performance Benchmarking: Understanding High Bias and High Variance in Learning Algorithms
In this resource, we illustrate how to determine whether a learning algorithm has high bias or high variance using concrete examples from the domain of speech recognition.

## Speech Recognition: A Case Study
Speech recognition systems are increasingly being utilized for various tasks such as web search on mobile phones. These systems transcribe audio clips into text, such as "What is today's weather?" or "Coffee shops near me."

## Measuring Training and Cross-validation Errors
Let's suppose that our speech recognition system achieves a training error of 10.8%, meaning that it fails to transcribe 10.8% of the audio clips from the training set perfectly. When tested on a separate cross-validation set, it gets 14.8% error. These numbers may initially suggest high bias as the system is getting 10% of the training set wrong.

## Benchmarking Against Human-level Performance
However, when dealing with tasks such as speech recognition, it's important to consider the human-level performance. For instance, even fluent speakers may achieve a transcription error rate of 10.6% due to various factors such as noisy audio.

By benchmarking against this human-level performance, we realize that our learning algorithm's training error (10.8%) is just slightly worse than humans (10.6%). Meanwhile, the cross-validation error (14.8%) is significantly higher than both the training error and human-level performance, indicating high variance.

## Establishing a Baseline Level of Performance
Establishing a baseline level of performance is crucial in understanding whether an algorithm has high bias or high variance. This baseline can be set by human-level performance or by another competing algorithm's performance. Once we have this baseline level, we can measure:

- The difference between the training error and the baseline level. If this is large, we say that the algorithm has a high bias problem.
- The difference between the training error and the cross-validation error. If this is high, we say that the algorithm has a high variance problem.

For some tasks, the baseline level of performance could be 0%, indicating perfect performance. However, in tasks like speech recognition, where some audio can be noisy, the baseline level could be higher.

## High Bias and High Variance
An algorithm can potentially suffer from both high bias and high variance. For instance, if the baseline performance, training error, and cross-validation error yield significant gaps, it would indicate that the algorithm has both high bias (for not achieving baseline performance) and high variance (for the high gap between training and cross-validation errors).