# Project Description
## Introduction
- I am unlikely to be presented with real-world data that is in a neatly formatted CSV file from which I can simply select a model and begin training; much more effort is usually required. I set up this project to improve my skills in dealing with imbalanced data sets, which can be a challenge for traditional models and sampling methods.
- I used a simulated dataset of credit card transactions that includes legitimate and fraudulent transactions over a two-year period. It includes the credit cards of 1000 customers transacting with a pool of 800 merchants. It contains various information about the transactions, such as personal information about the holder, the amount spent, the date and location, etc.
- The record can be found here on the Kaggle website: https://www.kaggle.com/datasets/kartik2112/fraud-detection/code. However, it is important to note that the data was artificially generated using a synthetic data generation tool called Sparkov.
## Preprocessing for data analysis.
- I have divided the preprocessing into two categories: Data Analysis and Training/Building. The reason for this is that I wanted to gain deeper insight into the data before manipulating features that might be relevant to the analysis. As a first step, I checked the data for duplicate and missing values, but none were present.
- Since a person may have multiple transactions and thus be listed more than once, I included an *identifier* that groups the transactions based on the credit card number and the account holder's first and last name. This is important for later analysis.
- This dataset contained each account holder's date of birth, but I think it makes more sense to convert it to their actual age. Another conversion is needed for the date and time of the transaction so that the individual components (year, month, day, and hour) are converted into separate components so that the model can detect patterns in the underlying data.
- As a final pre-processing step, I converted the individual coordinates of the account holder's residence and the merchant's residence into a distance specification. This also facilitates the detection of correlations between the distance of the account holder and the merchant.
## Data Analysis
- First, I wanted to check the correlation between each feature and the target variable. I included both Pearson and Spearman in case the data set might be skewed. The reason for this is that Pearson assumes a linear relationship between the variables, while Spearman is better suited to capture non-linear (monotonic) relationships.
- Since the data may be skewed, I selected some appropriate features to see if they were indeed point to such conclusions. I also used histograms to visualize my results. 

  ![image](https://user-images.githubusercontent.com/127037803/224010269-cd2faf48-3f01-4e4c-8c70-3a8d6c1fdc35.png)
- As seen in the graph, some of these features are skewed, i.e. there is an asymmetry of the data with respect to the mean. The *age* and *hour* features are only slightly skewed, while the *amount* feature is highly skewed.
- Next, I used a boxplot that shows how the amount spent in fraudulent transactions is distributed. To briefly explain the boxplot, the box is drawn from the first quartile to the third quartile. A vertical line passes through the box at the median. Whiskers run from each quartile to the upper and lower limits. The mean is a green triangle in the center of the box and the outliers appear as green X's above and below the upper and lower boundary lines.

  ![image](https://user-images.githubusercontent.com/127037803/223981326-5239e8bc-c286-4732-b1b4-7e691bfcc37a.png)
- In this case, most data points are spread between the range of 300 to 900, but being closer to the first quartile. It becomes apparent that there is definetly a connection between the amount spend and fraudulent activities since the median of both target splits have a significant distance inbetween
- Next, I took a look at the distribution of the jobs that were the most present in the fraudulent-activities split. Note that I also included the proportion of jobs in non-fraudulent activities, since a job might have a general high appearance in the dataset, making it also more likely to be present in fraud transactions, no matter the correlation.
- This dataset includes data about different people making different transactions. Since many individuals are listed for more than one transaction, they overrepresent their profession in the dataset. For this reason, I grouped transactions based on their unique identifier (composed of credit card number, first and last name) so that the represention is not distorted.

  ![image](https://user-images.githubusercontent.com/127037803/223982160-28e033fb-f9b6-4cdc-9ab6-761dd6d0daba.png)
- I could not find a clear correlation between a job and fraudulent transactions, suggesting that anyone, regardless of profession, can be a victim of credit card theft, although the trading standards officer and the copywriter seem to be very present in fraudulent transactions, in contrast to their presence in non-fraudulent transactions (about half compared to a materials engineer), but I would not read too much into this.
- Another valuable insight might be the preferred category in fraudulent transactions. I included the preferred category in non-fraudulent transactions for the same reason as before.

  ![image](https://user-images.githubusercontent.com/127037803/223984847-2c5c5d94-0999-4877-a517-54e97adf4542.png)
- From this illustration, it can be seen that the categories "grocery" and "shopping" occur most frequently in fraudulent transactions. The "health & fitness" and "housing" categories, on the other hand, are not very common in fraudulent transactions, despite their presence in legitimate transaction.
- Some of these categories are further categorized by "_pos" and "_net", but up to this point I have not found a clear answer to what they represent. I asked on the Kaggle forum and checked the documentation for the Sparkov generation tool. One hypothesis is that it differentiates between transactions made online and in person, but I have no data to prove this hypothesis.
- To continue with the analysis, I decided to visualize the difference in time of transactions that fall in the same month and were made with the same credit card.

  ![image](https://user-images.githubusercontent.com/127037803/224032244-18c631bc-cd52-4f55-ac75-67dab0829e74.png)
- The result is clear: While the standard deviation from the mean for legitimate transactions ranges from a few hundred to more than 4,000 hours, fraudulent transactions appear to be spread over a range of only a few hours. Outliers can even be found with a deviation of only up to five hours. While I find this to be an important finding, I have not included it as an additional feature in the dataset (explained later on).
- I also wanted to see if there was a relationship between a person's age and the distance between the merchant and the cardholder, but they appear to be consistent with data observed for non-fraudulent activity, indicating no connection.

  ![image](https://user-images.githubusercontent.com/127037803/224015017-c4166524-a1ed-4f3d-ae00-188e707e94f8.png)
  ![image](https://user-images.githubusercontent.com/127037803/224012311-37943629-dc95-44aa-ba6f-e8ee3dc28fe2.png)
## Preprocessing for Training
- The second part of the preprocessing is used to prepare the data for the model. The first thing to do is to remove all irrelevant features. I also excluded all personal information such as home address, name, etc. to protect the identity of the people involved (even if they were artificially created).
- I also removed most information about time, such as month or year, as I saw no reason to believe that there was a "credit card theft season" (the correlation supports this point). I did include the hour and unix time as they seem to be correlated to the target variable.
- An important aspect of choosing a model is to make sure that the target class of the data set is balanced.

  ![image](https://user-images.githubusercontent.com/127037803/224013444-bdb87924-7d8d-4443-b749-80b99f3f52e4.png)
- In this case, however, the target class is very unbalanced, which requires a different approach to training (especially sampling).
- As mentioned earlier, outliers make up a large portion of the data set, so it is important to handle them properly. In this case, they are genuine data points, so removing them could result in a loss of information. Instead, they are detected using the interquartile range and then marked to pass this information to the model.
- Since the model only accepts numeric values, categorical variables, such as the merchant or job title, must be encoded. In this case, one-hot coding cannot be used because the features have high cardinality; target coding is also not appropriate because the data set is highly imbalanced. Instead, frequency coding is used.

## Building the Model
- Since the dataset is very unbalanced, I decided to use the stratified KFold algorithm to ensure that each bin/split has the same proportion of positive and negative samples as the original dataset. I also used a combination of Tomek links and SMOTE to resample the training dataset.
- I chose a gradient-boosting classifier as the model because it is an ensemble model (combining multiple weak learners) that can capture complex nonlinear relationships, has built-in sampling weights that focus on unclassified samples, and provides multiple hyperparameters such as learning rate and maximum depth to boost its performance.
- Using simple majority class prediction would achieve 99% accuracy due to imbalance; therefore, accuracy is not a very good performance measure. Recall and the F1 score are used instead.

## Performance Evaluation
- As stated before, accuracy is not very suitable for performance evaluation, so I have included, once again, recall, precision, and the F1-score. In addition, a confusion matrix is used to display all mispredictions made by the model.
- Another useful metric is Area Under the Receiver Operating Characteristic Curve (AUR-ROC). It measures teh ability of a binary classification model to distinguish between positive and negative classes.

  ![image](https://user-images.githubusercontent.com/127037803/224487079-119e1c85-0449-4020-95f3-9cfa2205f9ee.png)
- The result is a model (blue line) that appears to have no problem distinguishing between the two classes. Although it is not perfect (score 1), it clearly outperforms the random classifier (green line).

## Concolusion
- In summary, this project really helped me develop my data analysis skills, as I had to look more closely at each feature and decide if it provides valuable information or was somehow correlated with the target variable.
- For example, I added the time difference of transactions that fall in the same month and were made with the same credit card as a separate feature (commented out after the plot), which greatly improved the model (100% for all scores for test and training). However, I removed it because I don't think it represents real-world behaviour since making several transactions in a short time span is no reason to suspect fraudulent activities. In addition, no regularization technique (such as scaling or minimum cost complexity pruning) helped reduce overfitting, so I decided to simply use the Unix time instead.
- It also allowed me to address the imbalance problem more efficiently. I learned new methods of resampling (e.g., combining Tomek links and SMOTE) and which models worked best. I have included them in my notes so that I am better prepared for future problems like this one.
- Note that each concept described in this documentation is explained in more detail in the Concept.md file. 
