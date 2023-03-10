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
- 


