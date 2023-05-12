import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import graphviz
from sklearn.tree import export_graphviz

df = pd.read_csv('hoofdtabelv2.csv')

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

le = LabelEncoder()
x['PRODUCT_LINE_EN_x'] = le.fit_transform(x['PRODUCT_LINE_EN_x'])
x['PRODUCT_TYPE_EN'] = le.fit_transform(x['PRODUCT_TYPE_EN'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

estimator = clf.estimators_[0]
dot_data = export_graphviz(estimator, out_file=None, feature_names=x.columns, class_names=y.unique(), filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("random_forest_tree.png", view=True)