# MNIST Digit Classification using Naive Bayes and Logistic Regression

This project demonstrates how to classify handwritten digits (7 and 8) from the MNIST dataset using two machine learning algorithms: **Naive Bayes** and **Logistic Regression**. The project involves data preprocessing, feature extraction, model training, and evaluating the performance of both classifiers on a test set.

## Table of Contents
- [Introduction](#introduction)
- [Feature Extraction](#feature-extraction)
- [Naive Bayes Classification](#naive-bayes-classification)
- [Logistic Regression](#logistic-regression)
- [Results](#results)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)

## Introduction

This project focuses on classifying the digits 7 and 8 from the MNIST dataset. The two classifiers used are:
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem with a Gaussian assumption for each class.
- **Logistic Regression**: A linear model for binary classification optimized using gradient ascent.

The purpose of this project is to compare the performance of these two classifiers based on accuracy.

## Feature Extraction

The following preprocessing steps are performed before training the classifiers:
1. **Loading the Dataset**: The MNIST dataset is loaded from a `.mat` file.
2. **Filtering the Data**: Only the samples corresponding to digits 7 and 8 are retained for training and testing.
3. **Normalization**: The features (pixel values) are standardized to have a mean of 0 and a standard deviation of 1 using `StandardScaler`.
4. **Mean and Covariance Calculation**: For Naive Bayes, the mean and covariance matrices for each class (7 and 8) are calculated.

## Naive Bayes Classification

Naive Bayes uses Bayes' theorem with the assumption that the features are conditionally independent within each class. For this task:
1. **Mean and Covariance**: The training data for digits 7 and 8 is used to compute the mean and covariance matrices for each class.
2. **Regularization**: A small regularization term is added to the covariance matrices to ensure they are positive definite.
3. **Prediction**: The classifier computes the log-probability of each test sample belonging to class 7 or class 8 and assigns the sample to the class with the higher log-probability.

## Logistic Regression

Logistic Regression is a linear model that predicts the probability of a sample belonging to class 1 (digit 8). In this implementation:
1. **Sigmoid Function**: The logistic (sigmoid) function is used to convert the linear combination of features and weights into probabilities.
2. **Cost Function**: The binary cross-entropy loss is used to measure the difference between the predicted and true labels.
3. **Gradient Ascent**: The model is trained by applying gradient ascent to optimize the parameters and minimize the cost function.
4. **Prediction**: After training, the model predicts the class of the test samples based on the optimized parameters.

## Results

The classifiers are evaluated on the test set, and the accuracy of each model is computed:
- **Logistic Regression Accuracy**: Achieved an accuracy of **X%** for classifying digits 7 and 8.
- **Naive Bayes Accuracy**: Achieved an accuracy of **Y%** for classifying digits 7 and 8.

Both models demonstrated reasonable performance, with slight differences in accuracy based on the underlying algorithm.

## Dependencies

To run this project, you will need to install the following Python libraries:
- `numpy`
- `scipy`
- `sklearn`
- `matplotlib` (optional, for visualizations)

You can install the required libraries using pip:

```bash
pip install numpy scipy scikit-learn
