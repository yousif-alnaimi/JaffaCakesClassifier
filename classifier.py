from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# optional imports for the graphing script - not necessary by default
import pandas as pd
import matplotlib.pyplot as plt

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

# # Boxplot code - no output is produced here - testing to try find the best result ended with poorer test and cv means
# # than the mean and sd method done below
# # KNN model performance grapher with respect to k - boxplots
# # initialise list of data for each iteration
# perf_list = []
# # for loop to iterate through n_neighbours (k) from 1 to 100, then add to a list of tuples
# for i in range(3, 101):
#       model4 = KNeighborsClassifier(n_neighbors=i, weights="distance", algorithm="ball_tree", p=2)
#       scores4 = cross_val_score(model4, X_train_std, y_train_std, cv=5, scoring="accuracy")
#       perf_list.append((scores4, i))
#
# results = [i[0] for i in perf_list]
# neighbours = [i[1] for i in perf_list]
# fig, ax = plt.subplots()
# plt.figure(figsize=(21,6))
# plt.boxplot(results)
# ax.set_xticklabels(neighbours)
# plt.savefig("graphs/KNN_boxplot.png", dpi=300)
# plt.show()


# # Code with individual plots instead of box plots
# # KNN model performance grapher with respect to k - disabled by default as graph is already obtained
# # initialise list of data for each iteration
# perf_list = []
# # for loop to iterate through n_neighbours (k) from 1 to 100, then add to a list of tuples
# for i in range(3, 51):
#       model4 = KNeighborsClassifier(n_neighbors=i, weights="distance", algorithm="ball_tree", p=2)
#       scores4 = cross_val_score(model4, X_train_std, y_train_std, cv=5, scoring="accuracy")
#       perf_list.append((scores4.mean(), scores4.std(), i))
#
# # print perf_list for numeric viewing
# print(perf_list)
# # convert the list of tuples into a dataframe for easy plotting
# perf_df = pd.DataFrame({"Mean": [i[0] for i in perf_list], "Standard Deviation": [i[1] for i in perf_list],
#                         "K": [i[2] for i in perf_list]})
#
# # initialise figure
# fig, ax = plt.subplots()
#
# # set x and y labels and colours
# ax.set_xlabel('K')
# ax.set_ylabel('Mean', color='tab:red')
# ax.tick_params(axis='y', labelcolor='tab:red')
# # plot a scatter plot of the means in red
# ax.scatter(perf_df["K"], perf_df["Mean"], color='tab:red')
#
# # initialise second y axis for standard deviation
# ax2 = ax.twinx()
#
# # initialise colour parameters and labels
# ax2.set_ylabel('Standard Deviation', color='tab:blue')
# ax2.tick_params(axis='y', labelcolor='tab:blue')
#
# # Scatter plot the Standard Deviation
# ax2.scatter(perf_df["K"], perf_df["Standard Deviation"], color='tab:blue')
#
# # prevent clipping of right label
# fig.tight_layout()
# # add grid lines for added clarity
# ax.grid(b=True, which='both', axis='both')
#
# # save and show the figure
# plt.savefig("graphs/KNN_comparison.png", dpi=300)
# plt.show()

# Chosen KNN model
model4 = KNeighborsClassifier(n_neighbors=22, weights="distance", algorithm="ball_tree", p=2)
scores4 = cross_val_score(model4, X_train_std, y_train_std, cv=5, scoring="accuracy")
print(scores4.mean(), scores4.std(), "KNN")

# Decision tree model
model5 = DecisionTreeClassifier()
scores5 = cross_val_score(model5, X_train_std, y_train_std, cv=5, scoring="accuracy")
print(scores5.mean(), scores5.std(), "Tree")

# SVM model
model6 = svm.SVC(C=1)
scores6 = cross_val_score(model6, X_train_std, y_train_std, cv=5, scoring="accuracy")
print(scores6.mean(), scores6.std(), "SVM")

# LDA model
model7 = LinearDiscriminantAnalysis()
scores7 = cross_val_score(model7, X_train_std, y_train_std, cv=5, scoring="accuracy")
print(scores7.mean(), scores7.std(), "LDA")

# showing statistics more clearly using predictions and classification reports

# logistic regression
# fit the model to the training sets
pred_model1 = model1.fit(X_train_std, y_train_std)
# predict what the labels should be based off test data
y_pred1 = pred_model1.predict(X_test_std)
# show the statistics more clearly in console
print("LR", classification_report(y_test_std, y_pred1, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred1, labels=["biscuit", "cake"]), "\n", accuracy_score(y_test_std, y_pred1))

# random forest classification test
pred_model2 = model2.fit(X_train_std, y_train_std)
y_pred2 = pred_model2.predict(X_test_std)
print("Forest", classification_report(y_test_std, y_pred2, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred2, labels=["biscuit", "cake"]), "\n", accuracy_score(y_test_std, y_pred2))

# boosting classification test
pred_model3 = model3.fit(X_train_std, y_train_std)
y_pred3 = pred_model3.predict(X_test_std)
print("Boost", classification_report(y_test_std, y_pred3, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred3, labels=["biscuit", "cake"]), "\n", accuracy_score(y_test_std, y_pred3))

# KNN classification test
pred_model4 = model4.fit(X_train_std, y_train_std)
y_pred4 = pred_model4.predict(X_test_std)
print("KNN", classification_report(y_test_std, y_pred4, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred4, labels=["biscuit", "cake"]), "\n", accuracy_score(y_test_std, y_pred4))

# Tree classification test
pred_model5 = model5.fit(X_train_std, y_train_std)
y_pred5 = pred_model5.predict(X_test_std)
print("Tree", classification_report(y_test_std, y_pred5, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred5, labels=["biscuit", "cake"]), "\n", accuracy_score(y_test_std, y_pred5))

# SVM classification test
pred_model6 = model6.fit(X_train_std, y_train_std)
y_pred6 = pred_model6.predict(X_test_std)
print("SVM", classification_report(y_test_std, y_pred6, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred6, labels=["biscuit", "cake"]), "\n", accuracy_score(y_test_std, y_pred6))

# LDA classification test
pred_model7 = model7.fit(X_train_std, y_train_std)
y_pred7 = pred_model7.predict(X_test_std)
print("LDA", classification_report(y_test_std, y_pred7, labels=["biscuit", "cake"]),
      confusion_matrix(y_test_std, y_pred7, labels=["biscuit", "cake"]), "\n", accuracy_score(y_test_std, y_pred7))
