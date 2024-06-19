# DecisionTreeML

The decision tree is a popular learning method for regression and classification, because of its interpretability. The decision tree relies on the bias that the response variable can be predicted by a hierarchical sequence of linear discriminant decisions. As with all learning methods, bias and variance must be balanced in decision trees. Pruning provides a means of reducing the effects of overfitting, which decision trees are susceptible to when the training set is large. We expect that pruning may yield greater improvements on datasets were larger sample sizes or larger numbers of features, which may be more prone to overfitting. Full and post-pruned decision trees are constructed for six datasets (see below). Decision tree classification and regression outperform a null model classifier and regressor except on the forest fires data set. Pruning was found to remove nodes in all data sets, but more strikingly for the regression data sets. Pruned trees also showed a larger improvement in performance for abalone and the forest fires data. The abalone data is also the largest data set in sample size, and features are measured to the finest precision, consistent with the hypothesis that the full decision tree overfits this dataset.

- [Video](Project_demo.mp4) demonstrating functionality of the decision tree
- [Report](Decision%20tree%20classification%20and%20regression.pdf) on classification and regression performance of the decision tree


1. Breast Cancer [Classification]
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
This breast cancer data set was obtained from the University of Wisconsin
2. Car Evaluation [Classification]
https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
The data is on evaluations of car acceptability based on price, comfort, and technical specifications.
3. Congressional Vote [Classification]
https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
This data set includes votes for each of the U.S. House of Representatives Congressmen on the 16 key
votes identified by the Congressional Quarterly Almanac.
4. Abalone [Regression]
https://archive.ics.uci.edu/ml/datasets/Abalone
Predicting the age of abalone from physical measurements.
5. Computer Hardware [Regression]
https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
The estimated relative performance values were estimated by the authors using a linear regression
method. The gives you a chance to see how well you can replicate the results with these two models.
6. Forest Fires [Regression]
https://archive.ics.uci.edu/ml/datasets/Forest+Fires
This is a difficult regression task, where the aim is to predict the burned area of forest fires, in the
northeast region of Portugal, by using meteorological and other data .

