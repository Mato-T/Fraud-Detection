# Project Description
## Introduction
- Data from the real world is unlikely to be presented to me in a neatly formatted CSV file where I just have to pick a model and start training; usually, there is much more effort involved. I have created this project to work on my skills to handle unbalanced datasets, which can present a challenge to conventional models and sampling techniques.
- I have used a simulated credit card transaction dataset containing legitimate and fraudulent transactions in a time span of two years. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants. It contains various information about the transactions, such as personal information about the owner and relevant details about the transaction, such as the amount spent, the date and location, etc.
### Include snippet of .head()
- The dataset is found here on the Kaggle Website: https://www.kaggle.com/datasets/kartik2112/fraud-detection/code. It is important to note, however, that the data was artificially generated using a synthetic data generation tool called Sparkov.
## Preprocessing for Data Analysis
- I have divided preprocessing into two categories: Data Analysis and Training/Building. This is because I wanted to gain a deeper insight into the data before manipulating or encoding any features that might be relevant for the analysis. As the first step, I checked for any duplicated and missing values but there were none present.
- Since a person can make several transactions, thus be listed more than once, I have included an *identifier* feature that groups transactions based on the credit card number, and the first and last name of the account owner. This becomes relevant for later analysis
- This dataset provided the date of birth of each account owner, but I think it is more appropriate to transform it into his/her actual age. Another transformation had to be done on the date and time of the transaction, so that the individual components (year, month, day, and hour) are transformed into separate components to help the model detect patterns in the underlying data.
- The last preprocessing step I took is to tranform the individual coordinates of the residency of the account owner and merchant into a representation of distance. This also makes it easier to detect any connection between the distance of the account owner and merchant.
## Data Analysis
- First, I wanted to check for the correlation between each feature and the target variable. I have included both Pearson and Spearman, in case the dataset might be skewed. This is because Pearson assumes a linear relationship between the variables, while Spearman is better at capturing non-linear monotonic relationships.
- Since the data might be skewed, I selected some appropriate features to see if they indeed point to such conclusions. I have also used histograms to visualize my findings.
-
![image](https://user-images.githubusercontent.com/127037803/224010269-cd2faf48-3f01-4e4c-8c70-3a8d6c1fdc35.png)
- As seen in the plot, some of these features happen to be skewed, meaning there is an asymmetry of data with respect to the mean. The age and hour feature are only slightly skewed but the amount feature is heavly skewed. For this reason, I have chosen to use another non-parametric test to get further insights into the relationships. I selected both *age*, *amount*, and *hour* (since they seemed to be the most correlated to the target) to test for correlation using the Chi-square test. The findings confirm that all features, espcially *amount*, is highly correlated to the target variable.
- Next, I used a boxplot displaying how the amount spend in fraudulent transactions is distributed. To shortly explain the box plot, the box is drawn from the first quartile to the third quartile. A vertical line goes through the box at the median. The whiskers go from each quartile to the upper and lower limits. The mean is a green triangle in the middle of the box and the outliers appear above and below the upper and lower limit lines as green Xs.

![image](https://user-images.githubusercontent.com/127037803/223981326-5239e8bc-c286-4732-b1b4-7e691bfcc37a.png)
- In this case, most data points are spread between the range of 300 to 900, but being closer to the first quartile. It becomes apparent that there is definetly a connection between the amount spend and fraudulent activities since the median of both target splits have a significant distance inbetween
- Next, I took a look at the distribution of the jobs that were the most present in the fraudulent-activities split. Note that I also included the proportion of jobs in non-fraudulent activities, since a job might have a general high appearance in the dataset, making it also more likely to be present in fraud transactions, no matter the correlation.
- This dataset includes data about different people making different transactions. Since many individuals are listed for more than one transaction, their overrepresent their profession in the dataset. For this reason, I grouped transactions based on their unique identifier (composed of credit card number, first and last name) so that the represention is not distorted.

![image](https://user-images.githubusercontent.com/127037803/223982160-28e033fb-f9b6-4cdc-9ab6-761dd6d0daba.png)
- I could not see a definite connection between jobs and fraudulent transactions, indicating that anyone, no matter the profession, can be come a victim of credit card theft, although the trading standards officer and copywriter for advertising seem to be very present in fraudulent transactions in contrast to their presence in non-fraudulent transactions (about half compared to a materials engineer) but I would not read too much into it.
- Another valuable insight might the preferred category in fraudulent transactions. I included the preferred category in non-fraudulent transactions for the same reason as before.

![image](https://user-images.githubusercontent.com/127037803/223984847-2c5c5d94-0999-4877-a517-54e97adf4542.png)
- Based on this plot, the grocery and shopping category seem to be the most popular one in fraudulent transactions. On the other hand, the health&fitness and home categories don't appear very often in fraudulent transactions, despite their presence in legitemate transactions
- Some of these categories are further categorized by the "_pos" and "_net" but to this point of writing, I did not find a valid answer. I have asked in the Kaggle forum, and checked the documentation for the Sparkov generation tool. One hypothesis is that it divides transactions made online and in person but I have not data to proof this hypothesis.
- Continuing with the analysis, I chose to visualize the difference in time of transactions that fall on the same month made by using the same credit card.

- The result is unambiguous: while the standard deviation from the mean ranges from a couple hundred to more than 4000 hours for legitemate transactions, fraudulent transactions seem to distribute over a range of a couply hours only. Outliers even cluster together at deviations of only up to five hours. This is an important insight so I added it as an additional feature to the dataset.
- I also wanted to see if there is any relationship between the age of a person and the distance between the merchant and the card owner but they seem to match the data seen in non-fraudulent activities.

![image](https://user-images.githubusercontent.com/127037803/224015017-c4166524-a1ed-4f3d-ae00-188e707e94f8.png)
![image](https://user-images.githubusercontent.com/127037803/224012311-37943629-dc95-44aa-ba6f-e8ee3dc28fe2.png)
## Preprocessing for Training
- The second part of preprocessing is meant to prepare the data for the model. The first thing to do is to remove any irrelevant features. I also excluded any personal information like home address, name, etc., to protect the identity of the people involved (even though they are artificially generated).
- I have also deleted most information about the date, like the month or year since I didn't see any reason to believe there was a "credit card theft season" (correlation underlines that point). I have included the hour and distance in time, since it seems to be correlated.
- One important aspect of choosing the model is making sure the target class of the dataset is balanced.

![image](https://user-images.githubusercontent.com/127037803/224013444-bdb87924-7d8d-4443-b749-80b99f3f52e4.png)
- In this case, however, the target class is highly unbalanced, requiring a different approach to training (especially sampling).
- As mentioned before, outliers make up a great deal of the dataset, so it is important to handle them correctly. In this case, they are genuine data points, so removing them might result in a loss of information. Instead, they will be detected using the interquartile range and then marked to pass this information on to the model.
- Since the model accepts numerical values only, categorical variables, such as the merchant or the job title must be encoded. In this case, one-hot encoding cannot be applied since the features are of high cardinality; target encoding is also not suitable since the dataset is highly unbalanced. Instead, frequency encodig is used.

## Building the Model
- Since the dataset is very unbalanced, I have decided to use the stratified KFold to ensure that each bin/split has the same proportions of positive and negative samples as the original dataset.
- I have also used a combination of Tomek links and SMOTE to resample the training dataset
- As a model, I decided to use a gradient boosting classifier since it is an ensemble model (combining several weak learners) allowing to capture complex non-linear relationships, it has built-in sample weights focussing on missclassified samples, and provides several hyperparameters, such as learning rate and maximum depth to increase its performance 
