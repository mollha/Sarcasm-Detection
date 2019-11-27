import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

# Create a dataFrame with feature names as columns
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target  # append an extra column (the label)
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df)