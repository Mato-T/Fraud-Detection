# Project Description
## Introduction
- Data from the real world is unlikely to be presented to me in a neatly formatted CSV file where I just have to pick a model and start training; usually, there is much more effort involved. I have created this project to work on my skills to handle unbalanced datasets, which can present a challenge to conventional models and sampling techniques
- I have used a simulated credit card transaction dataset containing legitimate and fraudulent transactions in a time span of two years. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants. It contains various information about the transactions, such as personal information about the owner and relevant details about the transaction, such as the amount spent, the date and location, etc.
### Include snippet of .head()
- The dataset is found here on the Kaggle Website: https://www.kaggle.com/datasets/kartik2112/fraud-detection/code. It is important to note, however, that the data was artificially generated using a synthetic data generation tool called Sparkov
## Preprocessing for Data Analysis
- I have divided preprocessing into two categories: Data Analysis and Training/Building. This is because I wanted to gain a deeper insight into the data before manipulating or encoding any features that might be relevant for the analysis. As the first step, I checked for any duplicated and missing values but there were none present
- This dataset provided the date of birth of each account owner, but I think it is more appropriate to transform it into his/her actual age. Another transformation had to be done on the date and time of the transaction, so that the individual components (year, month, day, and hour) are transformed into separate components to help the model detect patterns in the underlying data
- The last preprocessing step I took is to tranform the individual coordinates of the residency of the account owner and merchant into a representation of distance. This also makes it easier to detect any connection between the distance of the account owner and merchant
## Data Analysis
- First, I wanted to check for the correlation between each feature and the target variable. I have included both Pearson and Spearman, in case the dataset might be skewed. This is because Pearson assumes a linear relationship between the variables, while Spearman is better at capturing non-linear monotonic relationships
- Since the data might be skewed, I selected some appropriate features to see if they indeed point to such conclusions. I have also used histograms to visualize my findings
### include snippet of subplots
- As seen in the plot, some of these features happen to be skewed, meaning there is an asymmetry of data with respect to the mean. The age and hour feature are only slightly skewed but the amount feature is heavly skewed. For this reason, I have chosen to use another non-parametric test to get further insights into the relationships. I selected both *age*, *amount*, and *hour* (since they seemed to be the most correlated to the target) to test for correlation using the Chi-square test. The findings confirm that both, espcially the amount, is highly correlated to the target variable
- Next, I used a boxplot displaying how the amount spend in fraudulent transactions is distributed. To shortly explain the box plot, the box is drawn from the first quartile to the third quartile. A vertical line goes through the box at the median. The whiskers go from each quartile to the upper and lower limits. The mean is a green triangle in the middle of the box and the outliers appear above and below the upper and lower limit lines as green Xs
### include snippet of plot
- In this case, most data points are spread between the range of 300 to 900, but being closer to the first quartile. It becomes apparent that there is definetly a connection between the amount spend and fraudulent activities since the median of both target splits have a significant distance inbetween
- Next, I took a look at the distribution of the jobs that were the most present in the fraudulent-activities split. Note that I also included the proportion of jobs in non-fraudulent activities, since a job might have a general high appearance in the dataset, making it also more likely to be present in fraud transactions, no matter the correlation
### include snippet of plot
- I could not see a definite connection between jobs and fraudulent transactions, indicating that anyone, no matter the profession, can be come a victim of credit card theft, although the trading standards officer seems to be very present in fraudulent transactions in contrast to its presence in non-fraudulent transactions (about half compared to a materials engineer, the most frequent job)
- Another valuable insight might the preferred category in fraudulent transactions. I included the preferred category in non-fraudulent transactions for the same reason as before
### include snippet of plot
- Some of these categories are further categorized by the "_pos" and "_net" suffix which indicate X and Y (maybe point of sale and network), respectively. It becomes apparent that most activities involves ... (*edit after it becomes clear what the suffixes mean*)
- I also included plots of seeing if there is any relationship between the age of a person and the distance between the merchant and the card owner but they seem to match the data seen in non-fraudulent activities
### include both plots
## Preprocessing for Training
- The second part of preprocessing is meant to prepare the data for the model. The first thing to do is t oremove any irrelevant features. I also excluded any personal information like home address, name, etc., to protect the identity of the people involved (even though they are artificially generated)
- One important aspect of choosing the model is making sure the target class of the dataset is balanced
### include plot
- In this case, however, the target class is highly unbalanced, requiring a different approach to training (especially sampling)
- As mentioned before, outliers make up a great deal of the dataset, so it is important to handle them correctly. In this case, they are genuine data points, so removing them might result in a loss of information. Instead, they will be detected using the interquartile range and then marked to pass this information on to the model
- Since the model accepts numerical values only, categorical variables, such as the merchant or the job title must be encoded. In this case, one-hot encoding cannot be applied since the features are of high cardinality; target encoding is also not suitable since the dataset is highly unbalanced. Instead, frequency encodig is used
- In addiion, I used cyclical encoding for the month, day, and hour to provide a better represenation to the model. For example, if the days of the week (Monday, Tuesday, etc.) are encoded using numbers from 0-6 (6 being Sunday), the model would think that Monday (0) is far more distant in time than Saturday (5)

## Building the Model

