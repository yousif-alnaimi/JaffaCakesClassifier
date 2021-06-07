# JaffaCakesClassifier

## Introduction

A Python project to accompany my poster submission for the MATH40008 module. This project scrapes biscuit and cake
recipes from the internet, then does data analysis and classification on the dataset.

## Scraping

The main source for the data is from the cake and biscuit sections of [allrecipes.co.uk](http://allrecipes.co.uk). The
scraper first finds the second index page of the section (this is because the first page is formatted differently, and
so cannot be easily scraped). Then, it takes each entry in the page and fetches the recipe link from each one. Within
each recipe link, the scraper finds every `<li>` tag in the ingredients class on the webpage, then extracts the text and
cleans it for later processing.

The next step is to classify the data into features using regex (ignoring case). This finds the data and converts the
units to be ml or g as appropriate. Broad strokes had to be used to prevent the number of features from becoming too
large, and what goes into each category are defined below in the order they would be accepted - if one is seen before
the other, the latter mention is ignored (e.g., salted butter would be classified as butter, and give 0 to salt):

- Sugar: any string containing the word "sugar"
- Butter: any string containing the words "butter", "margarine", or "oil"
- Egg: any string containing the word "egg" (this includes eggs when separated into yolks and whites, and adjusts
  quantities accordingly)
- Flour: any string containing the words "flour" or "oats"
- Milk: any string containing the word "milk" (this will include vegan milks like soy, milk solids, and milk derivatives
  like condensed milk and milk chocolate)
- Soda: any string containing the phrases "baking powder" or "soda" (Note that, while self-raising flour contains these,
  self-raising flour does not contribute to these categories)
- Water: any string containing the word "water"
- Salt: any string containing the word "salt"
- Syrup: any string containing the words "syrup" or "honey"

The units where then found using regex in the `quantity_finder()` function, then placed into a dictionary for the
current recipe. This could find units denominated in g, ml, l, kg, oz, lb, tsp, tbsp, dessertspoon, and cup. Should the
units not be ascertainable, the unit was defulated to grams (this helps in the case of "a pinch of salt"). In this case,
recipes where a unit was exclusively denominated in ounces was removed, as they had a high chance of subverting the
regex detection due to inaccurate use of spaces, and this amounted to a small portion of total results skipped. Note
that the recorded recipes omit ingredients related to flavouring that do not change the make-up of the cake like nuts,
fruits, and jams. To filter out recipes like cheesecakes and other such irrelevant cakes (these are not comparable to
the sponge in a Jaffa Cake), recipes using less than 50 grams of either sugar or flour were removed from the list.

Finally, this data was written into a csv from the dictionary of the recipe each time, as this means that, in the rare
event of a crash or failure, progress is not lost (at the time of writing, all errors in reading were removed). These
can be found in the `csv_tests` and `data` folders. The final csvs had labels added to them manually, as well as manual
pruning of obviously outlier results (e.g., 75 kg of flour or 600g of salt).

## Importing

These datasets are recorded separately into a cake csv and a biscuit csv. These are imported into pandas DataFrames,
duplicates removed from each, leaving 1479 biscuit recipes and 2804 cake recipes, then concatenated into one larger
DataFrame. The next step was to split these into a feature and label set, then to normalise the data in the feature set
such that the features are proportions of recipes (i.e., the rows sum to one). Then the `train_test_split` occurs to
give us separate datasets in an 80:20 ratio to give us insight later in the classification step. Then, `StandardScaler`
was applied to give the classification algorithms more standardised data to work on.

The Jaffa Cake dataset is also imported in this script, using only the ingredients used in the sponge, and following the
same category rules as in the initial dataset. Additional rules were required for greater accuracy: where they use
"butter for greasing", one tablespoon (17g) is used, and ground almonds were added to the flour category. These recipes
were manually extracted from the following websites, in the order that they appear in the csv file:

- https://www.bbc.co.uk/food/recipes/mary_berrys_jaffa_cakes_58695
- https://www.thespruceeats.com/british-jaffa-cakes-recipe-4143259
- https://marshasbakingaddiction.com/homemade-jaffa-cakes/
- https://sortedfood.com/jaffacakes
- https://www.jamieoliver.com/recipes/chocolate-recipes/jaffa-cakes/
- http://allrecipes.co.uk/recipe/26008/jaffa-cakes.aspx
- http://allrecipes.co.uk/recipe/43983/the-vegan-dad---jaffa-cakes.aspx
- https://www.greatbritishchefs.com/recipes/jaffa-orange-cakes-recipe
- https://www.greatbritishchefs.com/recipes/jaffa-cakes-recipe
- https://www.waitrose.com/home/recipes/recipe_directory/j/jaffa-cakes.html
- https://www.radhidevlukia.co/post/jaffa-cakes-chocolate-orange-cakies-cookie-cake
- https://www.loveoggs.com/recipe/jaffa-cakes/

Care was taken to only choose recipes of normal sized Jaffa Cakes as opposed to giant and loaf cakes, though vegan
recipes were allowed through (more on this later). Note that recipe 11 in this list is outside the training set, as it
contains no sugar and uses maple syrup instead, but it has still been included for sake of completion.

## Graphing

Doing the same procedure as we did in the import step on the Jaffa Cake set, we can then concatenate this onto our full
set of data, giving us a dataset with three possible labels: "biscuit", "cake", or "jaffa". The data is then transformed
by `StandardScaler` which centres the mean to 0 (standard deviation scaling was disabled in this case for a more
pronounced graph, though the default behaviour is to divide the centred data by the standard deviation). Then, PCA is
run on the dataset with labels removed, reducing the number of components to 2 and allowing us to graph it.

The next step was to graph this data, separating them out by label into different colours and marker shapes to give a
graphical view of the data split. The Jaffa Cake points are then labelled with their recipe number (the row they in
which they appear in the Jaffa Cake csv), and results both labelled and unlabelled have been drawn. An optional step is
included which changes the colours of each class to be consistent with the colours of a Jaffa Cake, however this is
disabled by default as the classes are not as clear. These graphs can be found in the `graphs` folder. Notice the point
corresponding to recipe 11 in the far right of the graph - this point is the recipe which contains no sugar mentioned
earlier. We can clearly see this is an outlier, especially as it deviates from the rest of the data as it is the only
recipe to have no sugar in it, though it is still included as a potential recipe.

## Classifying

This step is the crux of the classification problem. A few models were chosen to serve as a basis to see which would be
best to choose for the final model. The initial selection was logistic regression, random forests, gradient boosting, K
nearest-neighbours, decision trees, and support vector machines. These models were run in their default settings and
through a 5-fold cross validation test on the training , then a prediction on the test set with a classification report
and confusion matrix (a table of predicted class in the columns and actual class in the rows - in this case column/row 1
is biscuit and column/row 2 is cake). Comparing their means and standard deviations, as well as the reports and
matrices, it seemed that random forests, gradient boosting, and k nearest-neighbour were the best algorithms in this
case. The reasons for selection and rejection for each model is summarised in the table below:

| Model       | Status | Reasoning |
| ----------- | ----------- | ----------- |
| Logistic Regression | Rejected | Lesser accuracy in detection in the training set, as well as a strong bias towards choosing cakes over biscuits, likely due to the difference in quantity of each data type. |
| Random Forest | Accepted | Strongest accuracy and lowest standard deviation out of all six models, as well as an equal split in false classifications of both biscuits and cakes |
| Gradient Boosting | Accepted | Similarly strong accuracy, but with a middling standard deviation. The confusion matrix, however, showed that it had less of a bias towards cake, than the random forest |
| K Nearest-Neighbour | Accepted | Similarly strong accuracy to the random forest and a similar standard deviation to the gradient boosting model. The confusion matrix here was also showed less of a bias towards cake. The very different algorithm as well provides a good distinction to the random forest and boosting algorithms.
| Decision Tree | Rejected | The lowest accuracy in the cross-validation and the highest standard deviation too. One redeeming quality however, was the close split to what we might expect with the ratio of cake to biscuit data. |
| Support Vector Machines | Rejected | Very high accuracy and low standard deviation, however the bias to cake was too strong to be included. |

Then, for each of these models, the parameters, found on
the [documentation pages for sklearn](https://scikit-learn.org/) were iterated through to maximise the performance,
while trying to keep the standard deviation low. This code is no longer in the script, but an example of the code used
can be found below
(altering the `max_depth` parameter in the gradient boosting model):

 ```angular2html
for i in range(1,15):
model3 = GradientBoostingClassifier(loss="exponential", random_state=0, n_estimators=240,
learning_rate=0.15, max_depth=i)
scores3 = cross_val_score(model3, X_train_std, y_train_std, cv=5, scoring="accuracy")
print(scores3.mean(), scores3.std(), "Boosting", i)
 ```

The output of this script can be found in `classifier.txt` in the `text_results` folder.

## Final Classification

Now we reach the goal of the project - are Jaffa Cakes biscuits or cakes?

This step is somewhat simple: we simply fit the previously tuned models to the scaled data, then run a prediction on the
scaled Jaffa Cake set. After some reformatting to give us a clearer indication of which recipes and classified as what,
we get the output below (from `final.txt`):

```angular2html
Forest [('cake', 1), ('cake', 2), ('cake', 3), ('cake', 4), ('biscuit', 5), ('cake', 6), ('biscuit', 7), ('biscuit', 8), ('biscuit', 9), ('cake', 10), ('cake', 11), ('cake', 12)]
Boosting [('cake', 1), ('cake', 2), ('cake', 3), ('cake', 4), ('biscuit', 5), ('cake', 6), ('cake', 7), ('biscuit', 8), ('biscuit', 9), ('cake', 10), ('cake', 11), ('cake', 12)]
KNN [('cake', 1), ('cake', 2), ('cake', 3), ('cake', 4), ('biscuit', 5), ('cake', 6), ('biscuit', 7), ('cake', 8), ('biscuit', 9), ('cake', 10), ('cake', 11), ('cake', 12)]
```

We can see that four recipes were detected as biscuit in the random forest model, while three were detected as biscuits
in the other two models. Comparing this with our earlier PCA graph, we can see that they roughly match up to what we
would expect - recipes 8 and 9 are firmly in biscuit territory, while recipes 5 and 7 are on the edge. All others had a
consensus that they were, in fact, cakes. Interestingly, while recipes 10 and 11 seem like they would be on the edge,
all the algorithms agreed that they were cakes, perhaps due to the bias towards cakes due to the imbalanced dataset.

With all of these models combined, we have a 72.22% chance across the algorithms that a jaffa cake recipe is considered a cake.
Even discounting recipe 11 for being outside the set we trained on, we still get a 69.69% chance of classifying a jaffa
cake recipe as a cake.