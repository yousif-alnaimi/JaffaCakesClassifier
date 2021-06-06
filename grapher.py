from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from importer import X, df_jaffa, y  # take the pre-processed dataset from the classifier script

# add a label for jaffa cake to make keying in later easier
jaffa_label = ["jaffa" for i in range(len(df_jaffa["sugar"]))]
df_jaffa["label"] = jaffa_label
# get the list of recipe columns - these will be our features (once the label column is removed)
feature_cols = list(df_jaffa.columns.values)
feature_cols.remove('label')

# make the X and y sets for the jaffa cake data set in the same way as we did for the main set
X_jaffa_for_graph = df_jaffa.loc[:, feature_cols]
y_jaffa_for_graph = df_jaffa.label
# normalise the sum of each row to 1
X_jaffa_for_graph = X_jaffa_for_graph.div(X_jaffa_for_graph.sum(axis=1), axis=0)
# concatenate the dataframes together to run PCA on them
X_for_graph = pd.concat([X, X_jaffa_for_graph])
y_for_graph = pd.concat([y, y_jaffa_for_graph])

# initialise and scale total dataset (this centres the mean to 0)
sc = StandardScaler(with_std=False)  # disables scaling of standard deviation to 1 to produce a more pronounced graph
X_std = sc.fit_transform(X_for_graph)
# initalise PCA method and fit the dataset to it
pca = decomposition.PCA(n_components=2, svd_solver="full")
X_std_pca = pca.fit_transform(X_std)  # Run the PCA
# transform the dataframe into a numpy array to remake a data frame with the new PCA elements
y_col = y_for_graph.to_numpy()

# generate new data frame with our new PCA components and the colour labels
df_for_graph = pd.DataFrame({"PCA component 1": X_std_pca[:, 0], "PCA component 2": X_std_pca[:, 1], "colour": y_col})

# optional colour palette setting - done in the colours of chocolate, orange, and tan
# disabled by default as the graph is not very clear

# colours = ["#150a03", "#ffa500", "#d1b26f"]
# sns.set_palette(sns.color_palette(colours))

# generate plots separately, using 3 different plots on the same axis, done by filtering for the label
# (in this case the column "colour") and drawing the graphs separately for each, allowing control over
# transparency and shape for each data set
gfg_pca = sns.scatterplot(x="PCA component 1", y="PCA component 2", label="cake",
                          data=df_for_graph[df_for_graph.colour == "cake"], alpha=0.12)
sns.scatterplot(x="PCA component 1", y="PCA component 2", label="biscuit",
                data=df_for_graph[df_for_graph.colour == "biscuit"], alpha=0.12, ax=gfg_pca)
# colour and size are marked separately in this graph to more clearly show the jaffa cake data points as large stars
sns.scatterplot(x="PCA component 1", y="PCA component 2", label="jaffa cake",
                data=df_for_graph[df_for_graph.colour == "jaffa"], alpha=1, marker="*", s=200, ax=gfg_pca)

# plot legend
lgnd = gfg_pca.legend()
# change point transparency in the legend to make it more readable
lgnd.legendHandles[0].set_alpha(0.5)
lgnd.legendHandles[1].set_alpha(0.5)

# save the figure into a file, then show the graph in the console
plt.savefig("graphs/default_PCA.png", dpi=300)
plt.show()
