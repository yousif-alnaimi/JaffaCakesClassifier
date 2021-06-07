from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from importer import X_orig_std, X_jaffa_std, y  # get the full datasets and jaffa datasets from importer.py

# define our models according to the best performers from classifier.py
forest_model = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True, oob_score=True)
boost_model = GradientBoostingClassifier(loss="exponential", n_estimators=280, learning_rate=0.24, max_depth=8)
knn_model = KNeighborsClassifier(n_neighbors=13, weights="distance", algorithm="ball_tree", p=2)

# fit our models to the whole dataset of all cakes and biscuits
forest_pred_model = forest_model.fit(X_orig_std, y)
boost_pred_model = boost_model.fit(X_orig_std, y)
knn_pred_model = knn_model.fit(X_orig_std, y)

# run predictions on the jaffa cake recipes data set
forest_predictions = forest_pred_model.predict(X_jaffa_std)
boost_predictions = boost_pred_model.predict(X_jaffa_std)
knn_predictions = knn_pred_model.predict(X_jaffa_std)

# augment data with added recipe numbers for added insight in conjunction with the graph
forest_predictions_print = [(forest_predictions[i], i+1) for i in range(12)]
boost_predictions_print = [(boost_predictions[i], i+1) for i in range(12)]
knn_predictions_print = [(knn_predictions[i], i+1) for i in range(12)]

# print the predictions with the recipe numbers
print("Forest", forest_predictions_print, "\nBoosting", boost_predictions_print,
      "\nKNN", knn_predictions_print)
