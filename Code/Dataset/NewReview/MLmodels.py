from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def my_SVM(sarcastic_data: pd.Dataframe, regular_data: pd.DataFrame):
    # ------------------------------ SPLIT DATA INTO TRAIN AND TEST -----------------------------------
    # shuffles data and splits
    regular_train, regular_test, x_train, x_test = train_test_split(regular_data, regular_data['label'], test_size=0.3)
    ironic_train, ironic_test, y_train, y_test = train_test_split(sarcastic_data, sarcastic_data['label'], test_size=0.3)
    model.fit(regular_train, ironic_train)
    a = model.score(regular_test, ironic_test)
    print('score: ', a)


# MACHINE LEARNING MODELS

model = SVC(gamma='auto', C=10, kernel='linear')

# Create a dataFrame with feature names as columns
#
# lreg = LogisticRegression()
# lreg.fit(regular_train, x_train)
# preds_valid = lreg.predict(regular_test)
# print(f1_score(x_test, preds_valid))