# Correlation
- The relationship of covariances is not easily interpreted, so the solution is to use correlation, which is the covariance estimation after having standardized the variables
- Correlation values are bound between values of -1 and +1. The bigger the number, the more positive is the relationship. The smaller the number, the more negative is the relationship. A value close to 0 signals no relationship at all
- In my analysis, I have included both Pearson's and Spearman's correlation coefficients. Pearson measures the linear correlation between two continuous variables and is defined as 

  $$r = \frac{\sum(x_i -\bar x)(y_i - \bar y)} {\sqrt{\sum(x_i - \bar x)² \sum(y_i - \bar y)²}}$$
 
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
    
    $$
    \log(\text{odds})=\log(p/(1-p))
    $$
    
- Using this equation, solve for *p* to represent the log of odds in the loss function
    
    $$
    p=\frac{e^{\log(\text{odds})}}{1+e^{\log(\text{odds})}}
    $$
    
- The *p* in the original loss function is replaced by the expression above and after some transformation and simplification, the new loss function will look as follows:
    
    $$
    L=-\Big(y_i\log(\text{odds})-\log(1+e^{\log(\text{odds})})\Big)
    $$
    
- Now, find $\gamma$ that minimizes the sum of losses by taking the derivative of the sum of losses with respect to $\gamma$. Note that $\gamma$ is still assumed to be the log of odds
    
    $$
    \frac{\delta}{\delta\log(\text{odds})}\sum_{i=1}^nL=\frac{\delta}{\delta\log(\text{odds})}\sum_{i=1}^n\Big[y_i\log(\text{odds})-\log(1+e^{\log(\text{odds})})\Big]\\=-\sum_{i=1}^ny_i+n\frac{e^{\log(\text{odds})}}{1+e^{\log(\text{odds})}}\\=-\sum_{i=1}^ny_i+np
    $$
    
- To take the derivative of the loss functions, simply use the chain rule (derivative of the nesting expression times the derivative of the nested expression). Note that the fraction is actually defined to be *p* so it can be replaced to arrive at the final solution
- Now, to find the critical value, set the solution equal to 0 and solve for *p*. The result will be the optimal value *p* that minimizes the loss function (loss equal to 0)
    
    $$
    -\sum_{i=1}^ny_i+np=0\\np=\sum_{i=1}^ny_i\\p=1/n\sum_{i=1}^ny_i=\bar y
    $$
    
- In a binary classification problem *y* can either be 0 or 1. So, the mean of ***y,*** in this case, is actually the proportion of 1. As $\gamma$  is the log of odds instead of the probability *p*, it must be converted to arrive at the first prediction
    
    $$
    F_0(x)=\dot y=\log(\frac{\bar y}{1-\bar y})
    $$
    
- The following steps are iterated **M**  times, where *M* denotes the number of trees created and *m* represents the index of each tree. So with the first prediction at hand, compute the effect that the prediction has on the loss function
- This is done by taking the derivative of the loss function with respect to the previous prediction $F_{m-1}$ and multiplying it by -1. As with regression, this results in what is known to as the residuals
    
    $$
    r_{i,m}=-\Big[\frac{\delta L(y_i, F(x_i))}{\delta F(x_i)}\Big ]_{F(x)=F_{m-1}(x)}
    $$
    
- The residuals are computed for each single sample *i*. The gradient provides guidance on the directions (+/-) and the magnitude in which the loss function can be minimized by altering the prediction. Now, substituting for the actual loss function:
    
    $$
    r_{i, m}=\frac{\delta}{\delta\log(\text{odds})}\Big[y_i\log(\text{odds})-\log(1+e^{\log(\text{odds})})\Big]\\=y_i-\frac{e^{\log(\text{odds})}}{1+e^{\log(\text{odds})}}\\=y_i-p
    $$
    
- Again, the derivative results in the difference between the target variable and the log of odds, which is why *r* is called residuals
- Now, a regression tree is used with all *x* features to predict the residual. The training data can be run down the first tree and the resulting prediction gets then added to previous prediction. The sum of these two values result in the new predicted target variable

$$
\gamma_{j, m}=argmin_{\gamma}\sum_{x_i\in R_{j, m}}^nL(y_i, F_{m-1}(x_i)+\gamma)\\=argmin_{\gamma}\sum_{x_i\in R_{j, m}}^n-\Big(y_iF_{m-1}(x_i+\gamma)-\log(1+e^{F_{m-1}(x_i+\gamma})\Big)
$$

- Training the regression tree results in $R_{j, m}$ for $j=1, ...,J_m$, where **R** is the subset of samples that are assigned to (or predicted to be) at *j (the* terminal node; i.e., a leave in the tree) and at **m (**the tree index). *J* is the total number of leaves
- A value $\gamma$ is needed that minimizes the loss function on each terminal node *j*. The $\sum_{x_i\in R_{j, m}}^n$term means that the loss is aggregated on all samples that belong to the subset *R* at terminal node *j*
- The best value for $\gamma$ is found by taking the derivative of the previous equation, However, solving this equation is very difficult so an approximation of the lost function using the second-order Taylor polynomial is done instead
- The second-order Taylor polynomial is used to approximate a function around a given point (in this case, the current model’s predictions) using its first and second derivative. This simplifies the computation done and the loss function is now being replaced by an approximation of it
    
    $$
    L(y_i, F_{m-1}(x_i)+\gamma)\approx L(y_i, F_{m-1}(x_i))+\frac{\delta}{\delta F}L(y_i, F_{m-1}(x_i))\gamma+1/2\frac{\delta²}{\delta F²}L(y_i, F_{m-1}(x_i))\gamma
    $$
    

- The approximation for the loss function can now be placed into the formula for calculating the best prediction, $\gamma$, that minimizes the sum of losses. For this, take the derivative of the loss function with respect to $\gamma$ and the best value, of course, results in the sum of losses being 0
    
    $$
    \frac{\delta}{\delta \gamma}\sum_{x_i\in R_{j, m}}L(y_i, F_{m-1}(x_i))+\frac{\delta}{\delta F}L(y_i, F_{m-1}(x_i))\gamma+1/2\frac{\delta²}{\delta F²}L(y_i, F_{m-1}(x_i))\gamma=0
    $$
    
- After solving this equation, the optimal value for $\gamma$ that minimizes the losses is defined to be as
    
    $$
    \gamma=\frac{\sum_{x_i\in R_{j, m}}(y_i-p)}{\sum_{x_i\in R_{j, m}}p(1-p)}
    $$
    
- With the optimal value at hand, update the prediction of the combined model $F_m$
    
    $$
    F_m(x)=F_{m-1}(x)+v\sum_{j=1}^{J_m}\gamma_{j, m}1(x\in R_{j, m})
    $$
    
- The $\gamma_{j, m}1(x\in R_{j, m})$ term means that the value $\gamma$ is picked if a given sample *x* falls in the subset of *R*. As all terminal nodes are exclusive, any given sample falls into only a single terminal node and the corresponding $\gamma$  value is added to the previous prediction to make up the new
- The *v* is the learning rate that controls the degree of contribution of the $\gamma$  prediction to the new, updated prediction $F_m$
- The idea is to add many more trees that all add up to many small steps, getting closer to the observed target variable and resulting in lower variance. To add new trees, the whole process starts from the beginning, but this time using the new predicted value for calculating the new residuals

