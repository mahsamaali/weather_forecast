import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import graphviz
from dtreeviz.trees import dtreeviz # remember to load the package
from numpy import mean
file = 'weatherstats_montreal_donnees_finales.xlsx'

col_names=['date','max_temperature','min_temperature','avg_wind_speed','precipitation','snow','avg_cloud']

#lire les données
dataset = pd.read_excel(file, sheet_name='in')
dataset_predit=pd.read_excel(file, sheet_name='2020')

X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:,7].values
X_test=dataset_predit.iloc[:, 0:7].values
y_test=dataset_predit.iloc[:,3].values

avg = mean(y_test)
print(avg)
#afficher les données
# print(X)
# print(y)
#print(X_test)
#Vérifier les données pour savoir si nous avons des données manquantes
# print(dataset.info())
# print(np.where(np.isnan(X)))
# print(dataset_predit.info())
# print(np.where(np.isnan(X_test)))

#Déclaration de DecisionTree.
# regressor = DecisionTreeRegressor(random_state=0)
#Fit the regressor object to the dataset.
# regressor.fit(X,y)

#obtenir les valeurs pour l'années 2020
# y_pred = regressor.predict(X_test)

# dataset_predit['avg_temp (celsius)']= y_pred
# dataset_predit.to_excel("finalPredictionWeather.xlsx","2020-prediction")


#save  decission tree

# text_representation = tree.export_text(regressor)


# with open("decistion_tree.log", "w") as fout:
#     fout.write(text_representation)
# Visualising the Decision Tree Regression results
# dot_data = tree.export_graphviz(regressor, out_file=None,
#                                 feature_names=col_names,
#                                 class_names='avg_temp (celsius)',
#                                 filled=True,)
# graph = graphviz.Source(dot_data, format="SVG")
# graph.render("decision_tree_training")
# print('decision_tree_graphivz.png')
# print(y_pred)



# viz = dtreeviz(regressor, X, y,
#                 target_name="target",
#                 feature_names=col_names,
#                 class_names=list('avg_temp (celsius)'))
# viz.save("decision_tree2.svg")

#5 Visualising the Decision Tree Regression results
# plt.scatter(X[:,0], y, color = 'red')
# plt.plot(X[:,0], regressor.predict(X), color = 'blue')
# plt.show()