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

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import confusion_matrix
from random import sample
from matplotlib import pyplot
file='data.xlsx'
dataset = pd.read_excel(file, sheet_name='data')
le = preprocessing.LabelEncoder()
# le.fit("JAN","FEV","MAR","AVR","MAI","JUN","JUL","AOU","SEP","OCT","NOV","DEC")
dataset = dataset.apply(le.fit_transform)
col_names=['date','avg_wind_speed','precipitation','snow','avg_cloud']
X = dataset.iloc[:, 0:5].values
y =dataset.iloc[:,5:6].values
# print(dataset)
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
clf = DecisionTreeClassifier(criterion = "entropy")
# criterion = "entropy"



#Fit the regressor object to the dataset.
clf = clf.fit(X_train,y_train)

#obtenir les valeurs pour l'années 2020
y_pred = clf.predict(X_test)

# feat_importance = clf.tree_.compute_feature_importances(normalize=False)
# feat_imp_dict = dict(zip(col_names, clf.feature_importances_))
# feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
# feat_imp.rename(columns = {0:'FeatureImportance'}, inplace = True)
# importance=feat_imp.sort_values(by=['FeatureImportance'], ascending=False).head()
# print(type(importance))
# importance.to_csv("importanceEntropy.csv")


importance = clf.feature_importances_
pyplot.bar([x for x in range(len(importance))], importance)



dataset.to_excel("finalPredictionWeather.xlsx","2020-prediction")
pd.DataFrame(y_pred).to_csv("dataPredit.csv")

#save  decission tree

text_representation = tree.export_text(clf)


with open("decistion_tree.log", "w") as fout:
     fout.write(text_representation)
# Visualising the Decision Tree Regression results
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=col_names,
                                class_names='avg_temp (celsius)',
                                filled=True,)
graph = graphviz.Source(dot_data, format="SVG")
graph.render("decision_tree_final")

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
accuracy=metrics.accuracy_score(y_test, y_pred)
df_confusion=confusion_matrix(y_test, y_pred)
print(type(df_confusion))
#df_confusion=np.insert(df_confusion, 4,accuracy)
# df_confusion.insert(2, "Team", "Any")
np.savetxt("matrix.csv", df_confusion, delimiter=",")
np.savetxt("accuracy.csv", [accuracy], delimiter=",")
plot_confusion_matrix(clf, X_test, y_test)
plt.show()
pyplot.show()
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