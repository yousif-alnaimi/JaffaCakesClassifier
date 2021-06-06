# JaffaCakesClassifier

## Introduction
 A Python project to accompany my poster submission for the MATH40008 module. This project scrapes biscuit and cake recipes from the internet, then does data analysis and classification on the dataset.
 
## Scraping
 The main source for the data is from the cake and biscuit sections of [allrecipes.co.uk](http://allrecipes.co.uk). The scraper first finds the second index page of the section (this is because the first page is formatted differently, and so cannot be easily scraped). Then, it takes each entry in the page and fetches the recipe link from each one. Within each recipe link, the scraper finds every `<li>` tag in the ingredients class on the webpage, then extracts the text and cleans it for later processing.
 
 The next step is to classify the data into features using regex (ignoring case). This finds the data and converts the units to be ml or g as appropriate. Broad strokes had to be used to prevent the number of features from becoming too large, and what goes into each category are defined below in the order they would be accepted - if one is seen before the other, the latter mention is ignored (e.g. salted butter would be classified as butter, and give 0 to salt):
 - Sugar: any string containing the word "sugar"
 - Butter: any string containing the words "butter", "margarine", or "oil"
 - Egg: any string containing the word "egg" (this includes eggs when separated into yolks and whites, and adjusts quantities accordingly)
 - Flour: any string containing the words "flour" or "oats"
 - Milk: any string containing the word "milk" (this will include vegan milks like soy, milk solids, and milk derivatives like condensed milk and milk chocolate)
 - Soda: any string containing the phrases "baking powder" or "soda" (Note that, while self-raising flour contains these, self-raising flour does not contribute to these categories)
 - Water: any string containing the word "water"
 - Salt: any string containing the word "salt"
 - Syrup: any string containing the words "syrup" or "honey"
 
 The units where then found using regex in the `quantity_finder()` function, then placed into a dictionary for the current recipe. This could find units demoninated in g, ml, l, kg, oz, lb, tsp, tbsp, dessertspoon, and cup. Should the units not be ascertainable, the unit was defulated to grams (this helps in the case of "a pinch of salt"). In this case, recipes where a unit was exclusively denominated in ounces was removed, as they had a high chance of subverting the regex detection due to inaccurate use of spaces, and this amounted to a small portion of total results skipped. Note that the recorded recipes omit ingredients related to flavouring that do not change the make-up of the cake like nuts, fruits, and jams. To filter out recipes like cheesecakes and other such irrelevant cakes (these are not comparable to the sponge in a jaffa cake), recipes using less than 50 grams of either sugar or flour were removed from the list.
 
 Finally, this data was written into a csv from the dicitonary of the recipe each time, as this means that, in the rare event of a crash or failure, progress is not lost (at the time of writing, all errors in reading were removed). These can be found in the `csv_tests` and `data` folders. The final csvs had labels added to them manually, as well as manual pruning of obviously outlier results (e.g. 75kg of flour or 600g of salt).

## Importing
 These datasets are recorded separately into a cake csv and a biscuit csv. These are imported into pandas DataFrames, duplicates removed from each, leaving 1479 biscuit recipes and 2804 cake recipes, then concatenated into one larger DataFrame. The next step was to split these into a feature and label set, then to normalise the data in the feature set such that the features are proportions of recipes (i.e. the rows sum to one). Then the `train_test_split` occurs to give us separate datasets in an 80:20 ratio to give us insight later in the classification step.
 The jaffa cake dataset is also imported in this set, using only the ingredients used in the sponge using the same rules as in the initial dataset. Additional rules were required for greater accuracy - where they use "butter for greasing", one tablespoon (17g) is used, and ground almonds were added to the flour category. These recipes were manually extracted from the following websites, in the order that they appear in the csv file:
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

 Care was taken to only choose recipes of normal sized jaffa cakes as opposed to giant and loaf cakes, though vegan recipes were allowed through (more on this later). Note that recipe 11 in this list is outside of the training set, as it contains no sugar and uses maple syrup instead, but it has still been included for sake of completion.

## Graphing
 Doing the same procedure as we did in the import step on the jaffa cake set, we can then concatenate this onto our full set of data, giving us a dataset with three possible labels: "biscuit", "cake", or "jaffa". The data is then transformed by `StandardScaler` which centres the mean to 0 (standard deviation scaling was disabled in this case for a more pornounced graph, though the default behaviour is to divide the centred data by the standard deviation). Then, PCA is run on the dataset with labels removed, reducing the number of components to 2 and allowing us to graph it.
 The next step was to graph this data, separating them out by label into different colours and marker shapes to give a graphical view of the data split. An optional step is included which changes the colours of each class to be consistent with the colours of a jaffa cake, however this is disabled by default as the classes are not as clear. These graphs can be found in the graphs folder. Notice the jaffa cake point in the far right of the graph - this point is the recipe which contains no sugar mentioned earlier. We can clearly see this is an outlier, especially as it deviates from the rest of the data as it is the only recipe to have no sugar in it.

## Classifying
 This step is the crux of the classification problem. A few models were chosen to serve as a basis to see which would be best to choose for the final model. The initial selection was logistic regression, random forests, gradient boosting, K nearest-neighbours, decision trees, and support vector machines. Running through these in their default settings and running a 5-fold cross validation test on them on the biscuit and cake dataset, then comparing their means and standard deviations seemed to show that random forests, gradient boosting, and k nearest-neighbour were the best algorithms in this case.
