from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# LOAD VECTORS FROM PICKLE FILES

# READ AS A NUMPY ARRAY
# pickle_in = open("elmo_ironic.pickle", "rb")
# ironic = pickle.load(pickle_in)

# pickle_in = open("elmo_regular.pickle", "rb")
# regular = pickle.load(pickle_in)

# READ AS A PANDAS DATAFRAME
ironic_data = pd.read_pickle("elmo_ironic.pickle", compression=None)
regular_data = pd.read_pickle("elmo_regular.pickle", compression=None)
print(len(ironic_data))

i_df = pd.read_csv('Ironic.csv', encoding="ISO-8859-1")
r_df = pd.read_csv('Regular.csv', encoding="ISO-8859-1")

ironic_data = pd.DataFrame({'vector': ironic_data[:, 0]})
ironic_data['label'] = 1
ironic_data['title_and_review'] = i_df["title"] + '. ' + i_df["review"]

regular_data = pd.DataFrame({'vector': regular_data[:, 0]})
regular_data['label'] = 0
regular_data['title_and_review'] = r_df["title"] + '. ' + r_df["review"]

# ------------------------------ SPLIT DATA INTO TRAIN AND TEST -----------------------------------
# shuffles data and splits
regular_train, regular_test, x_train, x_test = train_test_split(regular_data, regular_data['label'], test_size=0.3)
ironic_train, ironic_test, y_train, y_test = train_test_split(ironic_data, ironic_data['label'], test_size=0.3)


# ----------------------------------------------------------------------
#
# MACHINE LEARNING MODELS
#
# model = SVC(gamma='auto', C=10, kernel='linear')

# Create a dataFrame with feature names as columns

# lreg = LogisticRegression()
# lreg.fit(regular_train, x_train)
# preds_valid = lreg.predict(regular_test)
# print(f1_score(x_test, preds_valid))
#
# model.fit(regular_train, ironic_train)
# a = model.score(regular_test, ironic_test)
# print('score: ', a)