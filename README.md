# Project Description
## Introduction
- Data from the real world is unlikely to be presented to me in a neatly formatted CSV file where I just have to pick a model and start training; usually, there is much more effort involved. I have created this project to work on my skills to handle unbalanced datasets, which can present a challenge to conventional models and sampling techniques
- I have used a simulated credit card transaction dataset containing legitimate and fraudulent transactions in a time span of two years. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants. It contains various information about the transactions, such as personal information about the owner and relevant details about the transaction, such as the amount spent, the date and location, etc.
### Include snippet of .head()
- The dataset is found here on the Kaggle Website: https://www.kaggle.com/datasets/kartik2112/fraud-detection/code
## Preprocessing for Data Analysis
- I have divided preprocessing into two categories: Data Analysis and Training/Building. This is because I wanted to gain a deeper insight into the data before manipulating or encoding any features that might be relevant for the analysis. As the first step, i checked for any duplicated and missing values but there were none present
- This dataset provided the date of birth of each account owner, but I thought it was more appropriate to transform it into his/her actual age. Another transformation had to be done on the date and time of the transaction, so the individual components (year, month, day, and hour) are transformed into separate patters to help the model detect patterns in underlying data
- The last preprocessing suitable for data analysis is to tranform the individual coordinates of the resident of the account owner and merchant into a representation of distance. This also makes it easier to detect any connection between the distance of the account owner and merchant
## Data Analysis
- The first step I took is to check the correlation between each feature and the target variable. I have included both Pearson and Spearman, in case the dataset might be skewed. This is because Pearson assumes a linear relationship between the variables, while Spearman is better at capturing non-linear monotonic relationships
- Since the data might be skewed, I selected some appropriate features to see if they indeed point to such conclusions. I have also used histograms to visualize my findings
### include snipped of subplots
- As seen in the plot, some of these features happen to be skewed, meaning there is an asymmetry of data with respect to the mean (*descibe how it is found in graph*). For this reason, I have chosen to use another non-parametric test to get further insights into the relationships. I selected both *age* and and *amount* (since they seemed to be the most correlated to the target) to test for correlation using the Chi-square test. The findings confirm that both, espcially the amount, is highly correlated to the target variable
- Next, used a boxplot displaying how the amount spend in fraudulent transactions is distributed. To shortly explain the box plot, the box is drawn from the first quartile to the third quartile. A vertical line goes through the box at the median. The whiskers go from each quartile to the upper and lower limits. The mean is a green triangle in the middle of the box and the outliers appear above and below the upper and lower limit lines as green Xs
- In this case, most data points are spread between the range of 300 to 900, but being closer to the first quartile. It becomes apparent that there is definetly a connection between the amount spend and fraudulent activities since the median of both target splits have a significant distance inbetween
- Next, I took a look at the distribution of the jobs that were the most present in the fraudulent-activities split. Note that I also included the proportion of jobs in non-fraudulent activities, since a job might have a general high appearance in the dataset, making it also more likely to be present in fraud transactions, no matter the correlation
- I could not see a definite connection between jobs and fraudulent transactions, indicating that anyone, no matter the profession, can be come a victim of credit card theft, although the trading standards officer seems to be very present in fraudulent transactions in contrast to its presence in non-fraudulent transactions (about half compared to a materials engineer, the most frequent job)
- Another 
