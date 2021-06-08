from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

from importer import X_orig_std, X_jaffa_std, y  # get the full datasets and jaffa datasets from importer.py

# define our models according to the best performers from classifier.py
forest_model = RandomForestClassifier(n_estimators=80, class_weight="balanced", random_state=0)
boost_model = GradientBoostingClassifier(loss="exponential", n_estimators=240, learning_rate=0.15, max_depth=7,
                                         random_state=0)
knn_model = KNeighborsClassifier(n_neighbors=22, weights="distance", algorithm="ball_tree", p=2)

# fit our models to the whole dataset of all cakes and biscuits
forest_pred_model = forest_model.fit(X_orig_std, y)
boost_pred_model = boost_model.fit(X_orig_std, y)
knn_pred_model = knn_model.fit(X_orig_std, y)

# run predictions on the jaffa cake recipes data set
forest_predictions = forest_pred_model.predict(X_jaffa_std)
boost_predictions = boost_pred_model.predict(X_jaffa_std)
knn_predictions = knn_pred_model.predict(X_jaffa_std)

# augment data with added recipe numbers for added insight in conjunction with the graph
predict_df = pd.DataFrame(
    {"Recipe No.": list(range(1, 13)),
     "Forest": forest_predictions,
     "Boosting": boost_predictions,
     "KNN": knn_predictions,
}
)

# print the predictions with the recipe numbers
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(predict_df)

# find the probabilities for each class prediction for more insight
forest_predictions_probs = forest_pred_model.predict_proba(X_jaffa_std)
boost_predictions_probs = boost_pred_model.predict_proba(X_jaffa_std)
knn_predictions_probs = knn_pred_model.predict_proba(X_jaffa_std)

# put data into DataFrame for easier visualisation
prob_df = pd.DataFrame(
    {"Recipe No.": list(range(1, 13)),
     "Forest p_biscuit": ["{:.4f}".format(float(i[0])) for i in forest_predictions_probs],
     "Forest p_cake": ["{:.4f}".format(float(i[1])) for i in forest_predictions_probs],
     "Boosting p_biscuit": ["{:.4f}".format(float(i[0])) for i in boost_predictions_probs],
     "Boosting p_cake": ["{:.4f}".format(float(i[1])) for i in boost_predictions_probs],
     "KNN p_biscuit": ["{:.4f}".format(float(i[0])) for i in knn_predictions_probs],
     "KNN p_cake": ["{:.4f}".format(float(i[1])) for i in knn_predictions_probs]}
)

# print the prediction probabilities with the recipe numbers
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
    print(prob_df)
