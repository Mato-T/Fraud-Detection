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

# Chi-Square Test
- The chi-square test is another nonparametric test to determine if there is a significant difference between expected and observed frequencies of categorical data. It is defined as
$$\chi^2 = \sum_{i=1}^n\frac{(O_i - E_i)^2}{E_i}$$
- where *O* is the observed frequency and *E* is the expected frequency. The observed frequency is the actual number of occurrences or count of the category in a dataset
- If the observed frequencies are significantly different from the expected frequencies, it indicates that there is a significant difference between the two sets of data
- The expected frequency is obtained by assuming that there is no association between the two variables being tested for independence and that the frequencies in each cell of the contingency table are proportional to the corresponding marginal totals
$$E=\frac{\text{row total * }\text{column total}}{\text{grand total}}$$   
- These values are obtained from the contingency table (also referred to as crosstab), which is a table that displays the frequency distribution of the variables. Consider the following example
    
![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1a92ba5e-db1f-40b6-bfb0-72395ab0539e/Untitled.png)
