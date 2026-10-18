#import your other libraries here
import pandas as pd
# hint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

class MushroomClassifier:
    def __init__(self, data_path): # DO NOT modify this line
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.X_train = list()
        self.X_test = list()
        self.y_train = list()
        self.y_test = list()
        self.rf = None  # Your Random Forest Classfier

    def Q1(self): # DO NOT modify this line
        """
            1. (From step 1) Before doing the data prep., how many "na" are there in "gill-size" variables?
        """
        
        return self.df['gill-size'].isnull().sum()


    def Q2(self): # DO NOT modify this line
        """
            2. (From step 2-4) How many rows of data, how many variables?
            - Drop rows where the target (label) variable is missing.
            - Drop the following variables:
            'id','gill-attachment', 'gill-spacing', 'gill-size','gill-color-rate','stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring-rate','stalk-color-below-ring-rate','veil-color-rate','veil-type'
            - Examine the number of rows, the number of digits, and whether any are missing.
        """
        df_clean = self.df.dropna(subset=['label']).copy()
        df_clean.dropna(subset=['label'])
        df_clean = df_clean.drop(errors='ignore', columns=['id','gill-attachment', 'gill-spacing', 'gill-size','gill-color-rate','stalk-root', 'stalk-surface-above-ring',
                    'stalk-surface-below-ring', 'stalk-color-above-ring-rate','stalk-color-below-ring-rate','veil-color-rate','veil-type'])
        self.df = df_clean
        return self.df.shape


    def Q3(self):  # DO NOT modify this line
        self.Q2()  # remove this if you want to skip dependency and duplicate code
        
        num_cols = self.df.select_dtypes(include='number').columns.tolist()
        for col in num_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mean())
            
        cat_cols = [col for col in self.df.columns if col not in num_cols and col != 'label']
        for col in cat_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            
        temp_labels = self.df['label'].map({'e': 1, 'p': 0})
        class0_quan = (temp_labels == 0).sum()
        class1_quan = (temp_labels == 1).sum()
        
        return (int(class0_quan), int(class1_quan))


    def Q4(self):  # DO NOT modify this Line
        self.Q3()  # remove this if you want to skip dependency and duplicate code
        
        X_raw = self.df.drop(columns=['label'])
        y = self.df['label'].map({'e': 1, 'p': 0})
        
        X_encoded = pd.get_dummies(X_raw, drop_first=True)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y, test_size=0.20, stratify=y, random_state=2020
        )
        
        return (self.X_train.shape, self.X_test.shape)


    def Q5(self):
        """
            5. (From step 9) 2nd best feature name based on the importance.
            Create a Random Forest on training data using following parameters:
            - 'criterion': 'gini'
            - 'max_depth': 3
            - 'min_samples_leaf': 2
            - 'N_estimators': 100
            - 'random_state': 2020
            - Note: You need to reproduce the process (code) from Q2, Q3, Q4 to obtain the correct result.
        """
        self.Q4() # remove this if you want to skip dependency and duplicate code

        self.rf = RandomForestClassifier(criterion='gini', max_depth=3, min_samples_leaf=2, n_estimators=100, random_state=2020)
        self.rf.fit(self.X_train, self.y_train)

        importances = pd.Series(self.rf.feature_importances_, index=self.X_train.columns)
        importances_sorted = importances.sort_values(ascending=False)
        return importances_sorted.index[1]


    def Q6(self):
        """
            6. (From step 10) What is the value of macro f1 (2 digits)?
            Predict the testing data set with confusion_matrix and classification_report,
            using scientific rounding (less than 0.5 dropped, more than 0.5 then increased)
            - Note: You need to reproduce the process (code) from Q2, Q3, Q4, Q5 to obtain the correct result.
        """
        self.Q5() # remove this if you want to skip dependency and duplicate code
        y_pred = self.rf.predict(self.X_test)
        
        f1_classes = f1_score(self.y_test, y_pred, average=None)

        return (float(round(f1_classes[0], 2)), float(round(f1_classes[1], 2)))
