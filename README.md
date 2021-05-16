# How-to-Handle-Imbalanced-Dataset-Credit-Card-Fraud-Detection-

# Case Study to understand the Imbalanced Dataset
In a cancer prediction analysis problem you have a target variable in which their are 2 unique values(Yes/No).

            1.Yes - 100 records(patients having cancer) -> minority class
            2.No  - 900 records(patients not having cancer) -> majority class

* Data is not evenly distributed between the two values clearly we see imbalance distribution of data.

* so **Yes** class is known as **majority class** and **No** class is known as **minority class**.

* The problem here is most of the ML algorithms tends towrds majority class where in such cases like cancer prediction analysis on patients your aim is to focus more on minority class(i.e Yes which has 100 records).

* To handle this imbalance problem in dataset we have several techniques:

      1. undersamplin
      2. oversampling
      3. smote

# Under Sampling
* In Under Sampling, we're just reducing the count of data points in majority class by randomly picking 100 data points so that the count of both majority class and minority class are equal and we finally have balanced data ditribution.

* Under Sampling technique is generally not recommended because in data science having more data is key but in this case we're drastically reducing the data.

## from imblearn.under_sampling import NearMiss

# Over Sampling
* In Over Sampling technique, we're just increasing the count of data points(records) in minority class by simple duplication based on the existing data points.

## from imblearn.over_sampling import RandomOverSampler

# SMOTE(Synthetic Minority Oversampling Technique)
* In smote(synthetic minority oversampling technique), we generate data as we do in oversampling but the difference here is we use knn mechanism in smote technique.

* We generate data points based on the average of existing neighbor data points synthetically.

* Note that we're not duplicating from existing data points we're synthetically creating new data points based on the average of existing neighbors.

## from imblearn.over_sampling import SMOTE
