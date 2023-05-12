import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# load the data
df = pd.read_csv('hoofdtabelv2.csv')

# separate features (X) and target variable (y)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

#y = y.astype(str)

le = LabelEncoder()
x['PRODUCT_LINE_EN_x'] = le.fit_transform(x['PRODUCT_LINE_EN_x'])
x['PRODUCT_TYPE_EN'] = le.fit_transform(x['PRODUCT_TYPE_EN'])

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# create the decision tree classifier
clf = DecisionTreeClassifier(max_depth=3, random_state=0)

# train the model using the training set
clf = clf.fit(x_train, y_train)

# make predictions on the testing set
y_pred = clf.predict(x_test)

# calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", confusion_mat)

# visualize the decision tree
# dot_data = export_graphviz(clf, out_file=None, feature_names=x.columns, class_names=y.unique(), filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.format = 'png'
# graph.render("decision_treeMax3v2.png", view=True)

display = ConfusionMatrixDisplay(confusion_mat, display_labels=clf.classes_)
display.plot()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()