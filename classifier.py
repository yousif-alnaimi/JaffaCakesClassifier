from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# get the main datasets from importer.py
from importer import X_orig_std, y, X_train_std, X_test_std, y_train_std, y_test_std

# baseline logistic regression model
model1 = LogisticRegression()
# cross validation to evaluate the model with less variance
scores1 = cross_val_score(model1, X_train_std, y_train_std, cv=5, scoring="accuracy")
# show the performance calculations
print(scores1.mean(), scores1.std(), "LR")

# random forest model
model2 = RandomForestClassifier(n_estimators=80, random_state=0, class_weight="balanced")
scores2 = cross_val_score(model2, X_train_std, y_train_std, cv=5, scoring="accuracy")
print(scores2.mean(), scores2.std(), "Forest")

# boosting model
model3 = GradientBoostingClassifier(loss="exponential", random_state=0, n_estimators=240, learning_rate=0.15,
                                    max_depth=7)
scores3 = cross_val_score(model3, X_train_std, y_train_std, cv=5, scoring="accuracy")
print(scores3.mean(), scores3.std(), "Boosting")

# KNN model
model4 = KNeighborsClassifier(n_neighbors=22, weights="distance", algorithm="ball_tree", p=2)
scores4 = cross_val_score(model4, X_train_std, y_train_std, cv=5, scoring="accuracy")
print(scores4.mean(), scores4.std(), "KNN")

# Decision tree model
model5 = DecisionTreeClassifier()
scores5 = cross_val_score(model5, X_train_std, y_train_std, cv=5, scoring="accuracy")
print(scores5.mean(), scores5.std(), "Tree")

# SVM model
model6 = svm.SVC()
scores6 = cross_val_score(model6, X_train_std, y_train_std, cv=5, scoring="accuracy")
print(scores6.mean(), scores6.std(), "SVM")

# showing statistics more clearly using predictions and classification reports

# logistic regression
# fit the model to the training sets
pred_model1 = model1.fit(X_train_std, y_train_std)
# predict what the labels should be based off test data
y_pred1 = pred_model1.predict(X_test_std)
# show the statistics more clearly in console
print("LR", classification_report(y_test_std, y_pred1, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred1, labels=["biscuit", "cake"]))

# random forest classification test
pred_model2 = model2.fit(X_train_std, y_train_std)
y_pred2 = pred_model2.predict(X_test_std)
print("Forest", classification_report(y_test_std, y_pred2, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred2, labels=["biscuit", "cake"]))

# boosting classification test
pred_model3 = model3.fit(X_train_std, y_train_std)
y_pred3 = pred_model3.predict(X_test_std)
print("Boost", classification_report(y_test_std, y_pred3, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred3, labels=["biscuit", "cake"]))

# KNN classification test
pred_model4 = model4.fit(X_train_std, y_train_std)
y_pred4 = pred_model4.predict(X_test_std)
print("KNN", classification_report(y_test_std, y_pred4, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred4, labels=["biscuit", "cake"]))

# Tree classification test
pred_model5 = model5.fit(X_train_std, y_train_std)
y_pred5 = pred_model5.predict(X_test_std)
print("Tree", classification_report(y_test_std, y_pred5, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred5, labels=["biscuit", "cake"]))

# SVM classification test
pred_model6 = model6.fit(X_train_std, y_train_std)
y_pred6 = pred_model6.predict(X_test_std)
print("SVM", classification_report(y_test_std, y_pred6, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred6, labels=["biscuit", "cake"]))
