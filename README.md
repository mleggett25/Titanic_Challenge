# Titanic Challenge Using Pandas and Supervised Machine Learning

## Overview of the Titanic Challenge

### Purpose

The purpose of this challenge is to create a machine learning model that will predict whether or not a passenger of the Titanic survived based on various factors. This challenge is from [Kaggle](https://www.kaggle.com/c/titanic).

## Data Dictionary

Kaggle provided a training dataset and a test dataset. The training dataset included the 'Survived' variable and was used to train the machine learning model. The test dataset does not include the 'Survived' variable and was used to test whether or not the model was able to accurately predict whether or not the passenger survived. The predictions are submitted to Kaggle and an accuracy score is given.

The training dataset includes 891 observations and the test dataset includes 418 observations.

![Data Dictionary](Images/titanic_data_dictionary.PNG)

## Missing Values

In order to better understand the dataset and clean it as necessary, I first looked to see if the data had any missing values.

Train Dataframe

![Train Dataframe Missing Values](Images/train_df_missing_values.PNG)

Test Dataframe

![Test Dataframe Missing Values](Images/test_df_missing_values.PNG)


Only a few variables are missing values: Age (263), Fare (1), Cabin (1,014), and Embarked (2).

### Age Missing Values
Because age is a continuous variable, it would be beneficial to impute the missing age values with either the mean or the median. To see which would be best to use, I first looked to see if age followed a normal or skewed distribution.

![Age Distribution](Images/age_distribution.PNG)

Seeing as there is a slight skew, I believe it would be best to impute the missing age values with the median. Before simply imputing all the missing age values with one median, it is worth checking first whether age is correlated with any other variables.

Train Dataframe Age Correlation

![Train Dataframe Age Correlation](Images/train_df_cor.PNG)

Test Dataframe Age Correlation

![Test Dataframe Age Correlation](Images/test_df_cor.PNG)

As we can see, age is most correlated with passenger class. Therefore, I grouped age by passenger class and found the median for each. I then imputed these values for the missing age values.

![Median Age by Passenger Class](Images/median_age.PNG)

```
# Impute median ages in the training dataframe
train_frames = []
for i in list(set(train_df['Pclass'])):
    train_df_pclass = train_df[train_df['Pclass'] == i]
    train_df_pclass['Age'].fillna(train_df_pclass['Age'].median(),inplace=True)
    train_frames.append(train_df_pclass)
    new_train_df = pd.concat(train_frames)

# Impute median ages in the testing dataframe
test_frames = []
for i in list(set(test_df['Pclass'])):
    test_df_pclass = test_df[test_df['Pclass'] == i]
    test_df_pclass['Age'].fillna(train_df_pclass['Age'].median(),inplace=True)
    test_frames.append(test_df_pclass)
    new_test_df = pd.concat(test_frames)
```

### Fare Missing Values
For the missing Fare value, it might be safe to assume that the mean would be the best option to impute according to the passenger class. We can first check that the fare and passenger class are highly correlated.

Train Dataframe Fare Correlation

![Train Dataframe Fare Correlation](Images/train_df_fare_cor.PNG)

Test Dataframe Fare Correlation

![Test Dataframe Fare Correlation](Images/test_df_fare_cor.PNG)

Seeing as fare is indeed highly correlated with passenger class, I found the null Fare value and saw that the passenger was third class.

![Null Fare](Images/null_fare.PNG)

Using the train data, I found the mean of third class fare's and imputed the mean for the missing Fare value.

