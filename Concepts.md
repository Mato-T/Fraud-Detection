# Correlation
- The relationship of covariances is not easily interpreted, so the solution is to use correlation, which is the covariance estimation after having standardized the variables
- Correlation values are bound between values of -1 and +1. The bigger the number, the more positive is the relationship. The smaller the number, the more negative is the relationship. A value close to 0 signals no relationship at all
- In my analysis, I have included both Pearson's and Spearman's correlation coefficients. Pearson measures the linear correlation between two continuous variables and is defined as 

  $$r = \frac{\sum(x_i -\bar x)(y_i - \bar y)} {\sqrt{\sum(x_i - \bar x)²\sum(y_i - \bar y)²}}$$
 
- The numerator is simply the covariance that measures the joint variability of the two variables, while the denominator scales the variability by multiplying the standard deviations of the individual variables
- Spearman, on the other hand, transforms the numeric values into rankings and then correlates the rankings, thus minimizing the influence of any nonlinear relationship between the two variables under scrutiny. It is defined as:

  $$r_s = 1 - \frac{(6\sum d²)}{n(n² - 1)}$$
  
- where d is the difference in rank between the pairs of observations and n is the sample size. This method converts the variables to their ranks. Spearman is then defined as the Pearson correlation coefficient between the ranks of the two variables

# Skewness
- Skewness defines the asymmetry of data with respect to the mean. If the skew is negative, the left tail is too long and the mass of the observations are on the right side of the distribution. If it is positive, it is exactly the opposite

  $$\frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3$$
 
- The numerator of the formula calculates the sum of the differences between each observation and the sample mean
- The denominator is a scaling factor that adjusts for the range of the data. If the numerator is not divided by the standard deviation, the magnitude of the skewness would be influenced by the scale of the data, making it not comparable across different datasets. Cubing the results emphasizes extreme values, making it more sensitive to the shape of the distribution

# Gradient Boosting Classifier
- The boosting algorithm basically tries to reduce the bias error which arises when models are not able to identify relevant trends in the data. Boosting tires to build a strong predictive model from the mistakes of several weaker models
- Bagging is a concept used in the Random Forrest algorithm, where it trains base learners from independently bootstrapped subsets of the dataset. In contrast to bagging, where all base learners are trained simultaneously, boosting trains the base learners sequentially, making trees learn from the mistake of previous trees
- The entire process can be summarized by this image:

  ![image](https://user-images.githubusercontent.com/127037803/224346951-9fa02b57-ac9b-4db7-9e4b-7d898453516d.png)
- To begin with, the loss function must be defined. In the case of classification, cross-entropy is used    

    $$L_C=-(y_i\log(p)+(1-y_i)\log(1-p))$$
    
- The first term of the formula, $y_i\log(p)$ measures the loss when y is 1, and the second term $(1-y_i)\log(1-p)$ measures the loss when y is 0. The negative sign ensures the overall loss is positive. But instead of assuming $\gamma$ to be the predicted probability *p*, the log of odds is used to make the computation easier. The log of odds is defined to be as
    
    $$\log(\text{odds})=\log(p/(1-p))$$
    
- Using this equation, solve for *p* to represent the log of odds in the loss function
    
    $$p=\frac{e^{\log(\text{odds})}}{1+e^{\log(\text{odds})}}$$
    
- The *p* in the original loss function is replaced by the expression above and after some transformation and simplification, the new loss function will look as follows:
    
    $$L=-\Big(y_i\log(\text{odds})-\log(1+e^{\log(\text{odds})})\Big)$$
    
- Now, find $\gamma$ that minimizes the sum of losses by taking the derivative of the sum of losses with respect to $\gamma$. Note that $\gamma$ is still assumed to be the log of odds
    
    $$\frac{\delta}{\delta\log(\text{odds})}\sum_{i=1}^nL=\frac{\delta}{\delta\log(\text{odds})}\sum_{i=1}^n\Big[y_i\log(\text{odds})-\log(1+e^{\log(\text{odds})})\Big]$$
    $$=-\sum_{i=1}^ny_i+n\frac{e^{\log(\text{odds})}}{1+e^{\log(\text{odds})}}$$
    $$=-\sum_{i=1}^ny_i+np$$
    
- To take the derivative of the loss functions, simply use the chain rule (derivative of the nesting expression times the derivative of the nested expression). Note that the fraction is actually defined to be *p* so it can be replaced to arrive at the final solution
- Now, to find the critical value, set the solution equal to 0 and solve for *p*. The result will be the optimal value *p* that minimizes the loss function (loss equal to 0)
    
    $$-\sum_{i=1}^ny_i+np=0$$
    $$np=\sum_{i=1}^ny_i$$
    $$p=1/n\sum_{i=1}^ny_i=\bar y$$
    
- In a binary classification problem *y* can either be 0 or 1. So, the mean of *y,* in this case, is actually the proportion of 1. As $\gamma$  is the log of odds instead of the probability *p*, it must be converted to arrive at the first prediction
    
    $$F_0(x)=\dot y=\log(\frac{\bar y}{1-\bar y})$$
    
- The following steps are iterated *M*  times, where *M* denotes the number of trees created and *m* represents the index of each tree. So with the first prediction at hand, compute the effect that the prediction has on the loss function
- This is done by taking the derivative of the loss function with respect to the previous prediction $F_{m-1}$ and multiplying it by -1. This results in what is known to as the residuals

$$r_{i,m}=-\Big[\frac{\delta L(y_i, F(x_i))}{\delta F(x_i)}\Big ]_{F(x)=F_{m-1}(x)}$$
    
- The residuals are computed for each single sample *i*. The gradient provides guidance on the directions (+/-) and the magnitude in which the loss function can be minimized by altering the prediction. Now, substituting for the actual loss function:
    
    $$r_{i, m}=\frac{\delta}{\delta\log(\text{odds})}\Big[y_i\log(\text{odds})-\log(1+e^{\log(\text{odds})})\Big]$$
    $$=y_i-\frac{e^{\log(\text{odds})}}{1+e^{\log(\text{odds})}}$$
    $$=y_i-p$$
    
- Again, the derivative results in the difference between the target variable and the log of odds, which is why *r* is called residuals
- Now, a regression tree is used with all *x* features to predict the residual. The training data can be run down the first tree and the resulting prediction gets then added to previous prediction. The sum of these two values result in the new predicted target variable

$$\gamma_{j, m}=argmin_{\gamma}\sum_{x_i\in R_{j, m}}^nL(y_i, F_{m-1}(x_i)+\gamma)$$
$$=argmin_{\gamma}\sum_{x_i\in R_{j, m}}^n-\Big(y_iF_{m-1}(x_i+\gamma)-\log(1+e^{F_{m-1}(x_i+\gamma})\Big)$$

- Training the regression tree results in $R_{j, m}$ for $j=1, ...,J_m$, where *R* is the subset of samples that are assigned to (or predicted to be) at *j (the* terminal node; i.e., a leave in the tree) at tree index *m*. *J* is the total number of leaves
- A value $\gamma$ is needed that minimizes the loss function on each terminal node *j*. The $\sum_{x_i\in R_{j, m}}$ term means that the loss is aggregated on all samples that belong to the subset *R* at terminal node *j*
- The best value for $\gamma$ is found by taking the derivative of the previous equation, However, solving this equation is very difficult so an approximation of the loss function using the second-order Taylor polynomial is done instead
- The second-order Taylor polynomial is used to approximate a function around a given point (in this case, the current model’s predictions) using its first and second derivative. This simplifies the computation done and the loss function is now being replaced by an approximation of it
    
    $$L(y_i, F_{m-1}(x_i)+\gamma)\approx L(y_i, F_{m-1}(x_i))+\frac{\delta}{\delta F}L(y_i, F_{m-1}(x_i))\gamma+1/2\frac{\delta²}{\delta F²}L(y_i, F_{m-1}(x_i))\gamma$$
    

- The approximation for the loss function can now be placed into the formula for calculating the best prediction, $\gamma$, that minimizes the sum of losses. For this, take the derivative of the loss function with respect to $\gamma$ and the best value, of course, results in the sum of losses being 0
    
    $$\frac{\delta}{\delta \gamma}\sum_{x_i\in R_{j, m}}L(y_i, F_{m-1}(x_i))+\frac{\delta}{\delta F}L(y_i, F_{m-1}(x_i))\gamma+1/2\frac{\delta²}{\delta F²}L(y_i, F_{m-1}(x_i))\gamma=0$$
    
- After solving this equation, the optimal value for $\gamma$ that minimizes the losses is defined to be as
    
    $$\gamma=\frac{\sum_{x_i\in R_{j, m}}(y_i-p)}{\sum_{x_i\in R_{j, m}}p(1-p)}$$
    
- With the optimal value at hand, update the prediction of the combined model $F_m$
    
    $$F_m(x)=F_{m-1}(x)+v\sum_{j=1}^{J_m}\gamma_{j, m}1(x\in R_{j, m})$$
    
- The $\gamma_{j, m}1(x\in R_{j, m})$ term means that the value $\gamma$ is picked if a given sample *x* falls in the subset *R*. As all terminal nodes are exclusive, any given sample falls into only a single terminal node and the corresponding $\gamma$  value is added to the previous prediction to make up the new
- The *v* is the learning rate that controls the degree of contribution of the $\gamma$  prediction to the new, updated prediction $F_m$
- The idea is to add many more trees that all add up to many small steps, getting closer to the observed target variable and resulting in lower variance. To add new trees, the whole process starts from the beginning, but this time using the new predicted value for calculating the new residuals

# Resampling
## Tomek Links
- Tomek links help to reduce the imbalance by identifying pairs of samples from different classes that very so close to each other in feature space, and then removing the sample from the majority class in each pair

  ![tomek](https://user-images.githubusercontent.com/127037803/224483385-391c65b4-12d1-4fc9-85ec-4ce41a8f0881.png)

- As shown in this graph, the algorithm works by going through each sample of one class, finding the nearest neighbor that is of the opposite class, and repeating the process for the other class. Then removing the the sample from the majority class
  

## SMOTE

- Synthetics Minority Oversampling Technique (SMOTE) works by randomly picking a point from the minority class and computing the k-nearest neighbors for this point. Then, synthetic points are added between the chosen point and its neighbors
    
    ![smote](https://user-images.githubusercontent.com/127037803/224483393-357809ab-48e0-4c62-9f37-8b8d7c55aeeb.png)

- Smote can be effective in increasing the performance of classifiers on imbalanced datasets and is often used in combination with other techniques such as under-sampling the majority class, but in some cases, it might even decrease the performance if overused

# Evaluation Metrics
## Accuracy
- Accuracy is the simplest error measure, counting (as a percentage) how many of the predictions are correct
    
    $$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} = \frac{TP+TN}{TP+TN+FP+FN}$$
    
- Accuracy may cause problems when an imbalance exists between classes. For example, it may become a problem when the class is frequent or preponderant, such as in fraud detection, where most transactions are mostly legitimate with respect to a few criminal transactions
- In these situations, machine learning algorithms tend to guess the class in favor of the preponderant class and be wrong most of the time with the minor classes

## Precision
- Precision can help in these situations because it is about being precise when guessing. It tracks the percentage of times, when forecasting a class, that a class was right. For example 10 people were classified as having cancer, 9 really did have cancer, thus precision is 90 percent
    
    $$\text{Precision} = \frac{\text{Number of true positives}}{\text{Number of true positives}+\text{Number of false positives}}$$
    $$= \frac{TP}{TP+FP}$$

## Recall
- Another measure is the recall measure, which is defined as the ratio of the number of true positive instances (correctly classified) to the sum of the true positive and false negative (relevant instances missed by the model). A high recall score indicates that the model is able to capture the most important information from the source text
    
    $$\text{Recall} = \frac{\text{Number of true positives}}{\text{Number of true positives}+\text{Number of false negatives}}$$
    $$= \frac{TP}{TP+FN}$$

## F1-Score
- Precision and recall can be maximized together using the F1-score, which ensures that one always gets the best precision and recall combined
    
    $$\text{F1-score} = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}= \frac{2\cdot TP}{2\cdot TP + FP + FN}$$

## AUC-ROC
- Another useful metric is the Area Under The Receiver Operating Characteristic Curve (AUR-ROC). It provides a measure of the ability of a binary classification model to distinguish between positive and negative classes
- In binary classification, the model makes a prediction for each data point, assigning it to 0 or to 1. For each possible threshold value between 0 and 1, the model calculates the true positive rate (TPR) and the false positive rate (FPR)
    
    $$\text{TPR}=\frac{TP}{TP+FN}$$
    $$\text{FPR}=\frac{FP}{TN+FP}$$
    
- The TPR is the proportion of positive samples that are correctly classified as positive by the model, while the FPR is the proportion of negative samples that are incorrectly classified as positive
- The predicted probabilities of the positive class can be used to classify the instances into the positive or negative class by choosing a threshold value. For instance, if the threshold is set to 0.5, then all instances with predicted probability greater than or equal to 0.5 are classified as positive and the rest are classified as negative
    
    ![roc](https://user-images.githubusercontent.com/127037803/224483405-3b44de4a-8ef7-4f7f-a678-728e593ad27e.png)

- The ROC curve is a plot of the TPR versus the FPR for all possible threshold values. A random classifier would have a TPR and FPR that are both equal to the proportion of positive samples in the dataset, resulting in a diagonal line from the bottom-left to the top-right of the plot
    
    ![auc_roc](https://user-images.githubusercontent.com/127037803/224483410-e2876954-0a1d-4b77-9c9c-ee8f615d41a1.png)

- As seen in the split, the AUC-ROC is then calculates as the area under the ROC curve. The AUC-ROC ranges from 0 to 1, where a perfect classifier would have an AUC-ROC of 1, while a random classifier would have an AUC-ROC of 0.5

# Sources
https://www.manning.com/books/introducing-data-science (Every Topic Discussed)
https://www.youtube.com/watch?v=xZ_z8KWkhXE (Correlation)
https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5 (AUC-ROC)
https://www.kdnuggets.com/2020/01/5-most-useful-techniques-handle-imbalanced-datasets.html (Resampling)
https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-2-classification-d3ed8f56541e (Gradient Boost)
https://www.youtube.com/watch?v=jxuNLH5dXCs (Gradient Boost)
https://www.youtube.com/watch?v=StWY5QWMXCw (Gradient Boost)
