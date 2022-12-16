# What is Classification?
Classification is defined as the process of recognition, understanding, and grouping of objects and ideas into preset categories a.k.a “sub-populations.” With the help of these pre-categorized training datasets, classification in machine learning programs leverage a wide range of algorithms to classify future datasets into respective and relevant categories.
# What is Classification Algorithm?
Based on training data, the Classification algorithm is a Supervised Learning technique used to categorize new observations. In classification, a program uses the dataset or observations provided to learn how to categorize new observations into various classes or groups. For instance, 0 or 1, red or blue, yes or no, spam or not spam, etc. Targets, labels, or categories can all be used to describe classes. The Classification algorithm uses labeled input data because it is a supervised learning technique and comprises input and output information. A discrete output function (y) is transferred to an input variable in the classification process (x).

Classification algorithms used in machine learning utilize input training data for the purpose of predicting the likelihood or probability that the data that follows will fall into one of the predetermined categories. One of the most common applications of classification is for filtering emails into “spam” or “non-spam”, as used by today’s top email service providers.

In simple words, classification is a type of pattern recognition in which classification algorithms are performed on training data to discover the same pattern in new data sets.
# Types of Classification Algorithms
You can apply many different classification methods based on the dataset you are working with. It is so because the study of classification in statistics is extensive. The top five machine learning algorithms are listed below.

## 1. Logistic Regression
It is a supervised learning classification technique that forecasts the likelihood of a target variable. There will only be a choice between two classes. Data can be coded as either one or yes, representing success, or as 0 or no, representing failure. The dependent variable can be predicted most effectively using logistic regression. When the forecast is categorical, such as true or false, yes or no, or a 0 or 1, you can use it. A logistic regression technique can be used to determine whether or not an email is a spam.
## 2. Naive Bayes
Naive Bayes determines whether a data point falls into a particular category. It can be used to classify phrases or words in text analysis as either falling within a predetermined classification or not.
## 3. K-Nearest Neighbors
It calculates the likelihood that a data point will join the groups based on which group the data points closest to it are a part of. When using k-NN for classification, you determine how to classify the data according to its nearest neighbor.
## 4. Decision Tree
A decision tree is an example of supervised learning. Although it can solve regression and classification problems, it excels in classification problems. Similar to a flow chart, it divides data points into two similar groups at a time, starting with the "tree trunk" and moving through the "branches" and "leaves" until the categories are more closely related to one another.
## 5. Random Forest
The random forest algorithm is an extension of the Decision Tree algorithm where you first create a number of decision trees using training data and then fit your new data into one of the created ‘tree’ as a ‘random forest’. It averages the data to connect it to the nearest tree data based on the data scale. These models are great for improving the decision tree’s problem of forcing data points unnecessarily within a category.
## 6. Support Vector Machine
Support Vector Machine is a popular supervised machine learning technique for classification and regression problems. It goes beyond X/Y prediction by using algorithms to classify and train the data according to polarity.
# Types Of Classification Tasks In Machine Learning
Before diving into the four types of Classification Tasks in Machine Learning, let us first discuss Classification Predictive Modeling.
## Classification Predictive Modeling
A classification problem in machine learning is one in which a class label is anticipated for a specific example of input data.

Problems with categorization include the following:

* Give an example and indicate whether it is spam or not.
* Identify a handwritten character as one of the recognized characters.
* Determine whether to label the current user behavior as churn.

A training dataset with numerous examples of inputs and outputs is necessary for classification from a modeling standpoint.

A model will determine the optimal way to map samples of input data to certain class labels using the training dataset. The training dataset must therefore contain a large number of samples of each class label and be suitably representative of the problem.

When providing class labels to a modeling algorithm, string values like "spam" or "not spam" must first be converted to numeric values. Label encoding, which is frequently used, assigns a distinct integer to every class label, such as "spam" = 0, "no spam," = 1.

There are numerous varieties of algorithms for classification in modeling problems, including predictive modeling and classification.

It is typically advised that a practitioner undertake controlled tests to determine what algorithm and algorithm configuration produces the greatest performance for a certain classification task because there is no strong theory on how to map algorithms onto issue types.

Based on their output, classification predictive modeling algorithms are assessed. A common statistic for assessing a model's performance based on projected class labels is classification accuracy. Although not perfect, classification accuracy is a reasonable place to start for many classification jobs.

Some tasks may call for a class membership probability prediction for each example rather than class labels. This adds more uncertainty to the prediction, which a user or application can subsequently interpret. The ROC Curve is a well-liked diagnostic for assessing anticipated probabilities.

There are four different types of Classification Tasks in Machine Learning and they are following:
* Binary Classification
* Multi-Class Classification
* Multi-Label Classification
* Imbalanced Classification

## 1. Binary Classification
Those classification jobs with only two class labels are referred to as binary classification.

Examples comprise:

* Prediction of conversion (buy or not).
* Churn forecast (churn or not).
* Detection of spam email (spam or not).

Binary classification problems often require two classes, one representing the normal state and the other representing the aberrant state.

For instance, the normal condition is "not spam," while the abnormal state is "spam." Another illustration is when a task involving a medical test has a normal condition of "cancer not identified" and an abnormal state of "cancer detected."

Class label 0 is given to the class in the normal state, whereas class label 1 is given to the class in the abnormal condition.

A model that forecasts a Bernoulli probability distribution for each case is frequently used to represent a binary classification task.

The discrete probability distribution known as the Bernoulli distribution deals with the situation where an event has a binary result of either 0 or 1. In terms of classification, this indicates that the model forecasts the likelihood that an example would fall within class 1, or the abnormal state.

The following are well-known binary classification algorithms:

* Logistic Regression
* Support Vector Machines
* Simple Bayes
* Decision Trees

Some algorithms, such as Support Vector Machines and Logistic Regression, were created expressly for binary classification and do not by default support more than two classes.

Let us now discuss Multi-Class Classification.
## 2. Multi-Class Classification
Multi-class labels are used in classification tasks referred to as multi-class classification.

Examples comprise:

* Categorization of faces.
* Classifying plant species.
* Character recognition using optical.

The multi-class classification does not have the idea of normal and abnormal outcomes, in contrast to binary classification. Instead, instances are grouped into one of several well-known classes.

In some cases, the number of class labels could be rather high. In a facial recognition system, for instance, a model might predict that a shot belongs to one of thousands or tens of thousands of faces.

Text translation models and other problems involving word prediction could be categorized as a particular case of multi-class classification. Each word in the sequence of words to be predicted requires a multi-class classification, where the vocabulary size determines the number of possible classes that may be predicted and may range from tens of thousands to hundreds of thousands of words.

Multiclass classification tasks are frequently modeled using a model that forecasts a Multinoulli probability distribution for each example.

An event that has a categorical outcome, such as K in 1, 2, 3,..., K, is covered by the Multinoulli distribution, which is a discrete probability distribution. In terms of classification, this implies that the model forecasts the likelihood that a given example will belong to a certain class label.

For multi-class classification, many binary classification techniques are applicable.

The following well-known algorithms can be used for multi-class classification:

* Progressive Boosting
* Choice trees
* Nearest K Neighbors
* Rough Forest
* Simple Bayes

Multi-class problems can be solved using algorithms created for binary classification.

In order to do this, a method is known as "one-vs-rest" or "one model for each pair of classes" is used, which includes fitting multiple binary classification models with each class versus all other classes (called one-vs-one).

* One-vs-One: For each pair of classes, fit a single binary classification model.

The following binary classification algorithms can apply these multi-class classification techniques:

* One-vs-Rest: Fit a single binary classification model for each class versus all other classes.

The following binary classification algorithms can apply these multi-class classification techniques:

* Support vector Machine
* Logistic Regression

Let us now learn about Multi-Label Classification.
## 3. Multi-Label Classification
Multi-label classification problems are those that feature two or more class labels and allow for the prediction of one or more class labels for each example.

Think about the photo classification example. Here a model can predict the existence of many known things in a photo, such as “person”, “apple”, "bicycle," etc. A particular photo may have multiple objects in the scene.

This greatly contrasts with multi-class classification and binary classification, which anticipate a single class label for each occurrence.

Multi-label classification problems are frequently modeled using a model that forecasts many outcomes, with each outcome being forecast as a Bernoulli probability distribution. In essence, this approach predicts several binary classifications for each example.

It is not possible to directly apply multi-label classification methods used for multi-class or binary classification. The so-called multi-label versions of the algorithms, which are specialized versions of the conventional classification algorithms, include:

* Multi-label Gradient Boosting
* Multi-label Random Forests
* Multi-label Decision Trees

Another strategy is to forecast the class labels using a different classification algorithm.

Now, we will look into the Imbalanced Classification Task.
## 4. Imbalanced Classification
The term "imbalanced classification" describes classification jobs where the distribution of examples within each class is not equal.

A majority of the training dataset's instances belong to the normal class, while a minority belong to the abnormal class, making imbalanced classification tasks binary classification tasks in general.

Examples comprise:

* Clinical diagnostic procedures
* Detection of outliers
* Fraud investigation

Although they could need unique methods, these issues are modeled as binary classification jobs.

By oversampling the minority class or undersampling the majority class, specialized strategies can be employed to alter the sample composition in the training dataset.

Examples comprise -

* SMOTE Oversampling
* Random Undersampling

It is possible to utilize specialized modeling techniques, like the cost-sensitive machine learning algorithms, that give the minority class more consideration when fitting the model to the training dataset.

Examples comprise:

* Cost-sensitive Support Vector Machines
* Cost-sensitive Decision Trees
* Cost-sensitive Logistic Regression

Since reporting the classification accuracy may be deceptive, alternate performance indicators may be necessary.

Examples comprise -

* F-Measure
* Recall
* Precision
