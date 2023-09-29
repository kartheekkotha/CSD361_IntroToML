# Shuttle Data Set Classification

This project focuses on experimenting with three classifiers—BDR, LDA, and QDA—using the Statlog(Shuttle) data set. The data set consists of 8 numerical attributes and 7 classes, with a significant class imbalance, where approximately 80% belong to class 1. The objective is to achieve accuracies greater than 95%, especially for classes 2 to 7.

## Data Set

Download the Statlog(Shuttle) data set from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/148/statlog+shuttle). The data set is unbalanced, making it challenging to achieve high accuracy, especially for minority classes.

## Classifiers

### 1. BDR (Bayesian Decision Rule)

- Construct a class conditional probability distribution for the classes using the learning set (80% of the data).
- Set up a suitable-sized grid in the feature space.
- Use the grid to infer a density function for the volume element.
- Assign labels to each volume element based on the BDR.
- Assume equal prior probabilities for each class based on their proportions in the learning set.
- Evaluate accuracy/error rate on the test set (20% of the data).

### 2. LDA (Linear Discriminant Analysis) and QDA (Quadratic Discriminant Analysis)

- Use the corresponding classes from `sklearn.discriminant_analysis`.
- Build LDA and QDA classifiers using the learning set.
- Evaluate accuracy on the test set.

### 3. Density Estimation with KernelDensity

- Use `KernelDensity` class from `sklearn.neighbors` for multivariate class conditional probability density function.
- Utilize `GridSearchCV` from `sklearn.model_selection` to find the best density estimator in a non-parametric way.
- Refer to scikit-learn documentation and online tutorials for understanding density estimation using `KernelDensity`.

## Evaluation and Comparison

- Assess the accuracy of each classifier on the test set.
- Compare the performance of BDR, LDA, and QDA in terms of accuracy and effectiveness on minority classes.
- Explore the impact of grid size on the BDR classifier's accuracy.

## Instructions

1. Download the Statlog(Shuttle) data set from the provided link.
2. Implement and experiment with BDR, LDA, and QDA classifiers as described.
3. Use 80% of the data for training and 20% for testing.
4. Analyze the performance of each classifier and compare results.

