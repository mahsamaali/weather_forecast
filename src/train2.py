import pandas as pd
import numpy as np
# from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
#from sklearn.model_selection import train_test_split # Import train_test_split function
import graphviz
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import preprocessing
# import matplotlib.pyplot for plotting our result
import matplotlib.pyplot as plt

file='dataTrain.xlsx'
dataset = pd.read_excel(file, sheet_name='data')
# le = preprocessing.LabelEncoder()
# dataset = dataset.apply(le.fit_transform)

col_names=['date','avg_wind_speed ','precipitation','snow','avg_cloud']
dataset=pd.get_dummies(dataset,columns=col_names)
# print(dataset_dumies)
X = dataset.iloc[:, 0:5].values
y =dataset.iloc[:,5:6].values
print(X)
print(y)


# print(X[0:1095])
X_train=X[0:1095]
y_train=y[0:1095]
X_test=X[1095:1462]
y_test=y[1095:1462]

# print(X_test)

# #Déclaration de DecisionTree.
# clf=DecisionTreeClassifier(criterion = "gini", random_state = 100,
#                                max_depth=3, min_samples_leaf=10)
clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 100)

#Fit the regressor object to the dataset.
clf = clf.fit(X_train,y_train)

#obtenir les valeurs pour l'années 2020
y_pred = clf.predict(X_test)


dataset.to_excel("finalPredictionWeather.xlsx","2020-prediction")
pd.DataFrame(y_pred).to_csv("dataPredit.csv")

#save  decission tree

text_representation = tree.export_text(clf)


with open("decistion_tree2.log", "w") as fout:
     fout.write(text_representation)
# Visualising the Decision Tree Regression results
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=col_names,
                                class_names='avg_temp (celsius)',
                                filled=True,)
graph = graphviz.Source(dot_data, format="SVG")
graph.render("decision_tree_final2")

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# print("Average of test y",np.mean(y_test))
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# print(X_test[:, [1]])
# # scatter plot for original data
# plt.scatter(y_test, y_pred, color = 'red')
# # plot predicted data
# # plt.plot(X_test[:, [1]], y_pred, color = 'blue')
#
# # specify title
# plt.title('Prédiction de la tempretature (Decision Tree Regression)')
#
# # specify X axis label
# plt.xlabel('X')
#
# # specify Y axis label
# plt.ylabel('Y')
#
# # show the plot
# plt.show()