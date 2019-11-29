import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
model = SVC(gamma='auto', C=10, kernel='linear')
iris = load_iris()

# Create a dataFrame with feature names as columns
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target  # append an extra column (the label)
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df)

df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker='+')
plt.xlabel('sepal length')
plt.ylabel('width')

# plt.show()

# Lets also plot petal length and width
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='+')
plt.xlabel('petal length')
plt.ylabel('width')

x = df.drop(['target', 'flower_name'], axis='columns')
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(len(x_test))
print(len(x_train))

model.fit(x_train, y_train)
a = model.score(x_test, y_test)
print(a)