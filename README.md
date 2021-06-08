# JaffaCakesClassifier

## Introduction

A Python project to accompany my poster submission for the MATH40008 module. This project scrapes biscuit and cake
recipes from the internet, then does data analysis and classification on the dataset. We end in a prediction of the
class of a set of Jaffa Cake recipes.

## Scraping

The main source for the data is from the cake and biscuit sections of [allrecipes.co.uk](http://allrecipes.co.uk). The
scraper first finds the second index page of the section (this is because the first page is formatted differently, and
so cannot be easily scraped). Then, it takes each entry in the page and fetches the recipe link from each one. Within
each recipe link, the scraper finds every `<li>` tag in the ingredients class on the webpage, then extracts the text and
cleans it for later processing. Once this has been done for every recipe link in a page, the script moves on to the next
page by changing the index in the URL. The limit for this url has to be set manually and added to the for loop, and the
number can be found at the bottom of the page of any of the index pages.

The next step is to classify the data into features using regex (ignoring case). This finds the data and converts the
units to be ml or g as appropriate. Broad strokes had to be used to prevent the number of features from becoming too
large, and what goes into each category are defined below in the order they would be accepted - if one is seen before
the other, the latter mention is ignored (e.g. salted butter would be classified as butter, and give 0 to salt):

- Sugar: any string containing the word "sugar"
- Butter: any string containing the words "butter", "margarine", or "oil"
- Egg: any string containing the word "egg" (this includes eggs when separated into yolks and whites, and adjusts
  quantities accordingly) - we have used that an egg contains 52ml of egg, split into 22ml of yolk and 30ml of whites
- Flour: any string containing the words "flour" or "oats"
- Milk: any string containing the word "milk" (this will include vegan milks like soy and milk derivatives
  like condensed milk, though solids (anything measured in grams) will be ignored, e.g. milk chocolate)
- Soda: any string containing the phrases "baking powder" or "soda" (Note that, while self-raising flour contains these,
  self-raising flour does not contribute to these categories)
- Water: any string containing the word "water"
- Salt: any string containing the word "salt"
- Syrup: any string containing the words "syrup" or "honey"

The units where then found using regex in the `quantity_finder()` function, then placed into a dictionary for the
current recipe. This could find units denominated in g, ml, l, kg, oz, lb, tsp, tbsp, dessertspoon, and cup, and will
convert them using standard conversions to convert them into grams or milliltres as appropriate. While these are not
100% comparable, as most liquids used will be of similar density to water, which in itself has a density of roughly
1g/ml, this should be close enough for our purposes. Should theunits not be ascertainable, the unit was defaulted to
grams (this helps in the case of "a pinch of salt"). In this case, recipes where a unit was exclusively denominated
in ounces was removed, as they had a high chance of subverting the regex detection due to inconsistent use of spaces,
and this amounted to a small portion of total results skipped. Note that the recorded recipes omit ingredients
related to flavouring that, in most cases, do not change whether the recipe is for a cake or biscuit, like nuts, fruits,
and jams. An exception to this would be ground nuts, which act like flour, though there do not seem to be many recipes
containing these. To filter out recipes like cheesecakes and other such irrelevant cakes and biscuits (these are not
comparable to the sponge in a Jaffa Cake), recipes using less than 50 grams of either sugar or flour were removed
from the list.

Finally, this data was written into a csv from the dictionary of the recipe each time a new recipe was fully detected,
as this means that, in the rare event of a crash or failure, progress is not lost (at the time of writing, the script
could get through both the cake and biscuit recipe lists without suffering a crash provided the URL does not go outside
the maximum index). These csvs can be found in the `csv_tests` and `data` folders. The final csvs had labels added to
them manually, as well as manual pruning of obviously outlier results (e.g. recipes containing 75kg of flour or 600g
of salt).

## Importing

These datasets are recorded separately into a cake csv and a biscuit csv in the `data` folder. These are imported
into pandas DataFrames, duplicates removed from each, leaving 1479 biscuit recipes and 2804 cake recipes, then
concatenated into one larger DataFrame. The next step was to split these into a feature and label set, then to normalise
the data in the feature set such that the features are proportions of recipes (i.e. the rows sum to one). Then the
`train_test_split` occurs to give us separate datasets in an 80:20 ratio to give us insight later in the classification
step. Then, `StandardScaler` is applied to give the classification algorithms more standardised data to work on,
as this shifts and scales the data so that the mean becomes 0 and the standard deviation becomes 1.

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
set of data, giving us a dataset with three possible labels: "biscuit", "cake", or "jaffa". The data is then shifted by
`StandardScaler` so that its mean is 0 (standard deviation scaling was disabled in this case for a more
pronounced graph, though the default behaviour is to normalise standard deviation to 1 as seen earlier). Then, PCA is
run on the dataset with labels removed, reducing the number of components to 2 and allowing us to graph it.

The next step was to graph this data, separating them out by label into different colours and marker shapes to give a
graphical view of the data split. The Jaffa Cake points are then labelled with their recipe number (the row in which
they appear in the Jaffa Cake csv), and results both labelled and unlabelled have been drawn. An optional step is
included which changes the colours of each class to be consistent with the colours of a Jaffa Cake, however this is
disabled by default as the classes are not as clear. These graphs can be found in the `graphs` folder. Here is the
result of the default graph:

![Default PCA graph of the whole dataset](graphs/default_PCA.png)

Notice the point corresponding to recipe 11 in the far right of the graph - this point is the recipe which contains no
sugar mentioned earlier. We can clearly see this is an outlier, especially as it deviates from the rest of the data as
it is the only recipe to have no sugar in it, though it is still included as a potential recipe.

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
the [documentation pages for sklearn](https://scikit-learn.org/), were iterated through to maximise the performance,
while trying to keep the standard deviation low. This code is no longer in the script, but an example of the code used
(in this case altering the `max_depth` parameter in the boosting algorithm)
can be found below, as well as in the comments of classifier.py in the KNN graphing section:

 ```angular2html
for i in range(1,15):
  model3 = GradientBoostingClassifier(loss="exponential", random_state=0, n_estimators=240,
                                      learning_rate=0.15, max_depth=i)
  scores3 = cross_val_score(model3, X_train_std, y_train_std, cv=5, scoring="accuracy")
  print(scores3.mean(), scores3.std(), "Boosting", i)
 ```
 
Similarly, the resulting graph from collecting the mean performance and standard deviations can be found here:

![Graph of mean performance and standard deviation of performance against K](graphs/KNN_comparison.png)

Here we can see that K=22 has the highest mean, as well as a middling standard deviation (the differences in
standard deviation across this graph are very low, so this does not matter too much), and so this was
chosen as the value for n_neighbours in the final model.

The output of this script can be found in `classifier.txt` in the `text_results` folder.

## Final Classification

Now we reach the goal of the project - are Jaffa Cakes biscuits or cakes?

This step is somewhat simple: we simply fit the previously tuned models to the scaled data, then run a prediction on the
scaled Jaffa Cake set. After some reformatting to give us a clearer indication of which recipes and classified as what,
we get the output below (derived from `final.txt`):

| Recipe No. | Forest  | Boosting | KNN     |
|------------|---------|----------|---------|
| 1          | cake    | cake     | cake    |
| 2          | cake    | cake     | cake    |
| 3          | cake    | cake     | cake    |
| 4          | cake    | cake     | cake    |
| 5          | biscuit | biscuit  | biscuit |
| 6          | cake    | cake     | cake    |
| 7          | biscuit | cake     | biscuit |
| 8          | biscuit | biscuit  | cake    |
| 9          | biscuit | biscuit  | biscuit |
| 10         | cake    | cake     | cake    |
| 11         | cake    | cake     | cake    |
| 12         | cake    | cake     | cake    |

| Recipe No. | Forest p_biscuit | Forest p_cake | Boosting p_biscuit | Boosting p_cake | KNN p_biscuit | KNN p_cake |
|------------|------------------|---------------|--------------------|-----------------|---------------|------------|
| 1          | 0.0000           | 1.0000        | 0.0000             | 1.0000          | 0.0793        | 0.9207     |
| 2          | 0.0625           | 0.9375        | 0.0001             | 0.9999          | 0.0000        | 1.0000     |
| 3          | 0.1125           | 0.8875        | 0.0052             | 0.9948          | 0.1272        | 0.8728     |
| 4          | 0.0375           | 0.9625        | 0.0003             | 0.9997          | 0.0000        | 1.0000     |
| 5          | 0.8500           | 0.1500        | 0.9771             | 0.0229          | 0.9576        | 0.0424     |
| 6          | 0.1250           | 0.8750        | 0.0006             | 0.9994          | 0.0994        | 0.9006     |
| 7          | 0.5000           | 0.5000        | 0.3009             | 0.6991          | 0.8078        | 0.1922     |
| 8          | 0.6625           | 0.3375        | 0.8592             | 0.1408          | 0.4315        | 0.5685     |
| 9          | 0.9250           | 0.0750        | 0.9982             | 0.0018          | 0.9064        | 0.0936     |
| 10         | 0.1375           | 0.8625        | 0.0007             | 0.9993          | 0.2094        | 0.7906     |
| 11         | 0.4375           | 0.5625        | 0.1548             | 0.8452          | 0.3719        | 0.6281     |
| 12         | 0.0875           | 0.9125        | 0.0011             | 0.9989          | 0.0379        | 0.9621     |

We can see that four recipes were detected as biscuit in the random forest model, while three were detected as biscuits
in the other two models. Comparing this with our earlier PCA graph, we can see that they roughly match up to what we
would expect - recipes 8 and 9 are firmly in biscuit territory, while recipes 5 and 7 are on the edge. All others had a
consensus that they were, in fact, cakes. Interestingly, while recipes 10 and 11 seem like they would be on the edge,
all the algorithms agreed that they were cakes, perhaps due to the bias towards cakes due to the imbalanced dataset.

With all of these models combined, we have a 72.2% chance across the algorithms that a jaffa cake recipe is considered a cake.
Even discounting recipe 11 for being outside the set we trained on, we still get a 69.7% chance of classifying a jaffa
cake recipe as a cake.
