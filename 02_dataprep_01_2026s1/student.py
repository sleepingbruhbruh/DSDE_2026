import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


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
    return df.shape[0]


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
    cols_to_drop = []
    
    for col in df.columns:
        missing_ratio = df[col].isnull().sum() / len(df)
        if missing_ratio > 0.5:
            cols_to_drop.append(col)
            continue
            
        if col not in ['Age', 'Fare']:
            top_value_ratio = df[col].value_counts(dropna=False).iloc[0] / len(df)
            if top_value_ratio > 0.7:
                cols_to_drop.append(col)
                
    df = df.drop(columns=cols_to_drop)
    
    return df.shape[1]


def Q3(df):
    """
    Problem 3: Target Cleaning

    Remove all rows with missing values in the target variable "Survived".
    How many rows remain?
    """
    df_clean = df.dropna(subset=['Survived'])
    return df_clean.shape[0]


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
    q1 = df['Fare'].quantile(q=0.25)
    q3 = df['Fare'].quantile(q=0.75)
    iqr = q3 - q1

    df['Fare'] = df['Fare'].clip(lower=q1-1.5*iqr, upper=q3+1.5*iqr)

    return round(df['Fare'].mean(),2)


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
    num_cols = df.select_dtypes(include='number').columns.to_list()
    cat_cols = [col for col in df.columns if col not in num_cols]

    numeric_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='mean'))
    ])

    preprocessor = ColumnTransformer(
      transformers=[
        ('num', numeric_transformer, num_cols)
      ],
      remainder='passthrough'
    )

    df_imputed = preprocessor.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=num_cols + cat_cols)
    df_imputed = df_imputed[df.columns.tolist()]

    return round(df_imputed['Age'].mean(),2)


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
    num_cols = df.select_dtypes(include='number').columns.drop('Survived', errors='ignore').tolist()
    cat_cols = ['Embarked']

    num_pipeline = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='mean'))
    ])

    cat_pipeline = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='most_frequent')),
      ('encoder', OneHotEncoder(drop=['C'],handle_unknown='ignore',sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
      transformers=[
        ('cat', cat_pipeline, cat_cols),
        ('num', num_pipeline, num_cols)
      ],
      remainder='drop'
    )

    arr_transformed = preprocessor.fit_transform(df)
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
    all_feature_names = list(cat_feature_names) + num_cols

    df_transformed = pd.DataFrame(arr_transformed, columns=all_feature_names)

    return round(df_transformed['Embarked_Q'].mean(),2)


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
    df_clean = df.dropna(subset=['Survived'])
    X = df_clean.drop(columns=['Survived'])
    y = df_clean['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=123,
        stratify=y
    )
    return round(y_train.mean(), 2)