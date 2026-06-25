# Cancer Logistic Regression From Scratch

A from-scratch implementation of logistic regression in Python using NumPy, tested on Scikit-Learn’s breast cancer classification dataset.

This project demonstrates how logistic regression works internally by manually implementing the sigmoid function, cross-entropy loss, gradient descent, parameter updates, probability prediction, and binary classification.

I also included a detailed guide explaining how I implemented the model, including the mathematical theory behind logistic regression and how it works under the hood. Improvements to the document are welcomed.
- [Logistic Regression From Scratch: Theory and Implementation Guide](assets/Logistic-Regression.pdf)

## Visual Preview

### Sigmoid Training Animation

This animation shows a separate one-feature logistic regression model learning a sigmoid curve over time.

![Sigmoid Training Animation](assets/sigmoid-training-fit.gif)

### Cross-Entropy Loss vs Epochs

![Cross-Entropy Loss vs Epochs](assets/cross-entropy-loss-vs-epochs.png)

### Confusion Matrix

![Confusion Matrix](assets/confusion-matrix.png)

### Predicted Probability Distribution

![Predicted Probability Distribution](assets/predicted-probability-distribution.png)

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Results](#results)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation and Usage](#installation-and-usage)
- [Technologies Used](#technologies-used)
- [Learning Outcomes](#learning-outcomes)
- [Notes](#notes)

## Project Overview

Logistic regression is a classification algorithm used to predict the probability that an input belongs to a particular class.

In this project, logistic regression is implemented manually rather than using Scikit-Learn’s `LogisticRegression` model. Scikit-Learn is only used to load the dataset, split the data, scale the features, and calculate accuracy.

The model is trained using gradient descent and evaluated on unseen test data.

## Dataset

This project uses the Breast Cancer Wisconsin dataset from Scikit-Learn.

The dataset contains numerical features describing cell nuclei, such as:

- radius
- texture
- perimeter
- area
- smoothness
- compactness
- concavity
- concave points

The task is binary classification:

- Class 0
- Class 1

## Results

The model achieved strong performance on the breast cancer classification dataset.

```text
Train Accuracy: ...
Test Accuracy: ...