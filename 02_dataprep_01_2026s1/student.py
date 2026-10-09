import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


"""
ASSIGNMENT 2 (STUDENT VERSION)

Each problem starts from the original input DataFrame and is evaluated
independently. Do not carry over results or modifications from previous
problems. Repeat a previous step only when explicitly instructed.
"""


def Q1(df):
    """
    Problem 1: Data Exploration

    How many rows are there in "titanic_to_student.csv"?
    """
    # TODO: Code here
    return None


def Q2(df):
    """
    Problem 2: Feature Filtering

    2.1 Drop variables with more than 50% missing values.
    2.2 Check all columns except "Age" and "Fare" for flat values and
        drop columns where a flat value is more than 70%.

    From 2.1 and 2.2, how many columns remain?

    Ensure missing values are considered in the calculation. If using
    value_counts(), include dropna=False.
    """
    # TODO: Code here
    return None


def Q3(df):
    """
    Problem 3: Target Cleaning

    Remove all rows with missing values in the target variable "Survived".
    How many rows remain?
    """
    # TODO: Code here
    return None


def Q4(df):
    """
    Problem 4: Outlier Handling

    For "Fare", replace outlier values with the boundary values.
    If value < (Q1 - 1.5IQR), replace it with (Q1 - 1.5IQR).
    If value > (Q3 + 1.5IQR), replace it with (Q3 + 1.5IQR).

    What is the mean of "Fare" after replacing the outliers, rounded to
    2 decimal places?

    Hint: Use round(_, 2).
    """
    # TODO: Code here
    return None


def Q5(df):
    """
    Problem 5: Numeric Imputation

    Build a scikit-learn numeric preprocessing pipeline.
    Separate the target variable "Survived" from the input features.
    For numeric input columns, build a Pipeline using
    SimpleImputer(strategy="mean"). Use ColumnTransformer to apply
    preprocessing to the selected columns.

    What is the mean of "Age" after imputing the missing values, rounded to
    2 decimal places?

    Hint: Use Pipeline, SimpleImputer, ColumnTransformer, and round(_, 2).
    """
    # TODO: Code here
    return None


def Q6(df):
    """
    Problem 6: Categorical Encoding

    Repeat the numeric preprocessing from Problem 5 and separate the target
    variable "Survived" from the input features.
    For "Embarked", build a categorical Pipeline using
    SimpleImputer(strategy="most_frequent") and OneHotEncoder. Use "C" as the
    reference category, handle unknown categories as all zeros, and return
    dense output.
    Combine the numeric and categorical pipelines using ColumnTransformer
    inside a Pipeline.

    What is the mean of "Embarked_Q", rounded to 2 decimal places?

    Hint: Check drop, handle_unknown, sparse_output, and use round(_, 2).
    """
    # TODO: Code here
    return None


def Q7(df):
    """
    Problem 7: Train-Test Splitting

    Repeat the target cleaning from Problem 3.
    Separate the input features (X) and target (y). Split the data into
    training and test sets using a 70:30 ratio, random_state=123, and
    stratify=y.

    What is the proportion of survivors in the training data, rounded to
    2 decimal places?

    Hint: Use dropna(), round(_, 2), and train_test_split() from
    sklearn.model_selection.
    """
    # TODO: Code here
    return None
