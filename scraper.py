import time
from bs4 import BeautifulSoup
import requests
import re
from fractions import Fraction
import csv

target_url = "http://allrecipes.co.uk/recipes/cake-recipes.aspx?page="
metric_list = ["g", "ml", "kg", "l"]  # list of acceptable metric units
file_name = "csv_tests/cake-recipes.csv"  # file name to write the finished csv to (has to end in ".csv")


# list of implemented units and their multipliers
# unit_list = [("g", 1), ("ml", 1), ("l", 1000), ("kg", 1000), ("oz", 28), ("lb", 453)]

def imperial_unit_checker(input_string):
    """
    Function to take the input of a string and return a multiplier to return a value in ml or g
    if it contains an imperial measurement.
    """
    if "tsp" in input_string or "teaspoon" in input_string:
        multiplier = 6
    elif "dessertspoon" in input_string:
        multiplier = 7
    elif "tbsp" in input_string or "tablespoon" in input_string:
        multiplier = 17
    elif "cup" in input_string:
        multiplier = 284
    else:
        # this will make unrecognised values 1 (e.g. 1 pinch of salt will be equal to 1 gram)
        multiplier = 1
    return multiplier


def quantity_finder(input_string):
    """
    Function to take in an input string and find the quantity of an ingredient and normalise it to g or ml
    as appropriate. Supports units in unit_list and measurements in imperial_unit_checker.
    """
    # searches for a 1 to 4 digit number sequence followed by a unit
    if re.search(r'\d{1,4}g', input_string, re.IGNORECASE):
        # finds the first instance of the accepted string and extracts the number from it
        quantity = int(re.findall(r'\d{1,3}g', input_string, flags=re.I)[0][:-1])
    elif re.search(r'\d{1,4}ml', input_string, re.IGNORECASE):
        quantity = int(re.findall(r'\d{1,3}ml', input_string, flags=re.I)[0][:-2])
    elif re.search(r'\d{1,3}kg', input_string, re.IGNORECASE):
        quantity = int(re.findall(r'\d{1,3}kg', input_string, flags=re.I)[0][:-2])
        quantity *= 1000  # multiplier to normalise to g or ml
    elif re.search(r'\d{1,3}l', input_string, re.IGNORECASE):
        quantity = int(re.findall(r'\d{1,3}l', input_string, flags=re.I)[0][:-1])
        quantity *= 1000
    elif re.search(r'\d{1,3}oz', input_string, re.IGNORECASE):
        quantity = int(re.findall(r'\d{1,3}oz', input_string, flags=re.I)[0][:-2])
        quantity *= 28
    elif re.search(r'\d{1,3}lb', input_string, re.IGNORECASE):
        quantity = int(re.findall(r'\d{1,3}lb', input_string, flags=re.I)[0][:-2])
        quantity *= 453
    else:
        # takes the float value of the sum of the fractions that can be parsed from the string, done by
        # splitting the string and trying to parse fractions only when there is a number in the split
        try:
            quantity = float(
                sum(Fraction(s) for s in [x for x in input_string.split() if any(char.isdigit() for char in x)]))
            quantity *= imperial_unit_checker(input_string)
        except:
            quantity = "fail"
    return quantity


def get_recipe_links(url):
    """
    Function to take a url input of a page containing recipe links and returning the links to those recipes.
    """
    response = requests.get(url)
    data = response.text
    soup = BeautifulSoup(data, 'lxml')
    # find all href tags containing "recipe" in classes called "row recipe"
    recipes_raw = [i.find_all("a", href=True) for i in
                   soup.find_all("div", {"class": "row recipe"})]
    # flatten the list
    recipes_flattened = [item for sublist in recipes_raw for item in sublist]
    # get the href out of the a tags if they are recipe links
    recipes_filtered2 = [i.get('href') for i in recipes_flattened if "recipe/" in i.get('href')]
    # filter out jaffa cakes (prevents training on data we want to check)
    recipes_filtered = [i for i in recipes_filtered2 if "jaffa" not in i]
    # remove duplicate recipe links
    recipes = list(set(recipes_filtered))
    return recipes


def classify_ingredients(ingred_list):
    """
    Function to take the list scraped from the "li" tags and extract out key ingredients from the strings.
    All units have been converted to ml or g as appropriate (including eggs).
    The filtering methods are only commented for the first time a method appears, onwards they are identical
    except for the different values they are updating.
    """
    # initialise dictionary
    start_dict = {"sugar": 0, "butter": 0, "egg": 0, "flour": 0, "milk": 0, "raising-agent": 0,
                  "water": 0, "salt": 0, "syrup": 0}
    # initialise fail statement
    failed = False
    # loop through all ingredient strings found
    for i in ingred_list:
        if "oz" in i and not any(key_string in i for key_string in metric_list):
            failed = True  # fail if imperial units are used exclusively in an ingredient
            break  # saves time searching through
        elif " g " in i:
            failed = True  # fail if units are spaced
            break
        # go through each ingredient to see if it is present in the string
        elif re.search("sugar", i, re.I):
            quantity = quantity_finder(i)  # run the quantity finder function
            # add the new value found to the old value of sugar in the case multiple types of sugar are present
            if quantity == "fail" or quantity < 50:
                # if statement to account for failed quantity_finder function (very rare, usually due to variable
                # measurements like "1-2 eggs"), as well as less than 50g of
                # sugar (this is only done for sugar and flour, as these are considered core ingredients which are
                # not present in irrelevant recipes like rice krispies biscuits and flourless cakes
                failed = True
                break
            new = start_dict.get("sugar") + quantity
            # update dictionary with the new value for sugar
            start_dict.update({"sugar": new})

        elif re.search("butter", i, re.I) or re.search("margarine", i, re.I):
            quantity = quantity_finder(i)
            if quantity == "fail":
                failed = True
                break
            new = start_dict.get("butter") + quantity
            start_dict.update({"butter": new})

        elif re.search("oil", i, re.I):
            quantity = quantity_finder(i)
            if quantity == "fail":
                failed = True
                break
            new = start_dict.get("butter") + quantity
            start_dict.update({"butter": new})

        elif re.search("egg", i, re.I):
            try:
                # one recipe is breaking the egg detection because of "3-4 eggs"
                quantity = int(re.findall(r'\d', i)[0])
            except:
                failed = True
                break
            if "yolk" in i:
                quantity *= 22  # average mass of a large egg yolk
            elif "white" in i:
                quantity += 30  # average mass of a large egg white
            else:
                quantity *= 52  # average mass of a large egg
            new = start_dict.get("egg") + quantity
            start_dict.update({"egg": new})

        elif re.search("flour", i, re.I) or re.search("oat", i, re.I):
            quantity = quantity_finder(i)
            if quantity == "fail" or quantity < 50:
                failed = True
                break
            new = start_dict.get("flour") + quantity
            start_dict.update({"flour": new})

        elif re.search("milk", i, re.I):
            # this will include buttermilk and other similar liquid milk equivalents, as well as milk chocolate
            # and condensed milk
            quantity = quantity_finder(i)
            if quantity == "fail":
                failed = True
                break
            new = start_dict.get("milk") + quantity
            start_dict.update({"milk": new})

        elif re.search("soda", i, re.I) or re.search("baking powder", i, re.I):
            quantity = quantity_finder(i)
            if quantity == "fail":
                failed = True
                break
            new = start_dict.get("raising-agent") + quantity
            start_dict.update({"raising-agent": new})

        elif re.search("water", i, re.I):
            quantity = quantity_finder(i)
            if quantity == "fail":
                failed = True
                break
            new = start_dict.get("water") + quantity
            start_dict.update({"water": new})

        elif re.search("salt", i, re.I):
            quantity = quantity_finder(i)
            if quantity == "fail" or quantity >= 60:
                # some recipes end up with too much salt - this removes them
                failed = True
                break
            new = start_dict.get("salt") + quantity
            start_dict.update({"salt": new})

        elif re.search("syrup", i, re.I) or re.search("honey", i, re.I):
            quantity = quantity_finder(i)
            if quantity == "fail":
                failed = True
                break
            new = start_dict.get("syrup") + quantity
            start_dict.update({"syrup": new})

    # if one of the fail conditions occur, the recipe will be None, and then filtered out later
    if failed:
        return None
    else:
        return start_dict


def get_ingredients(url):
    """
    Function that takes in a url and returns the cleaned text in a list from all the li tags in the ingredients class.
    """
    response = requests.get(url)  # get the data from the link
    data = response.text
    soup = BeautifulSoup(data, 'lxml')
    # find all "li" tags in the ingredients class (these contain the necessary data)
    ingredients_raw = [i.find_all("li") for i in
                       soup.find_all("section", {"class": "recipeIngredients gridResponsive__module"})][0]
    ingredients = [i.text for i in ingredients_raw]  # extract text from li tags
    ingredients_neat = [i.replace("\n", "") for i in ingredients]  # clean out newlines
    # run the classifier function to return a dictionary of the ingredients
    return classify_ingredients(ingredients_neat)


# initialise recipe_list with example recipe - data is not added to the csv, just the keys
recipe_list = [{'sugar': 425, 'butter': 225, 'egg': 0, 'flour': 375,
                'milk': 0, 'raising-agent': 6.0, 'water': 7.0, 'salt': 3.0, 'syrup': 0}]
# initialise headers for csv
keys = recipe_list[0].keys()
# write headers and first recipe to allow appending
with open(file_name, 'w', newline='') as output_file:
    # write csv according to dictionaries in the list of ingredients
    dict_writer = csv.DictWriter(output_file, keys)
    # write the column labels at the top
    dict_writer.writeheader()
    # close the file so it can be opened differently later
    output_file.close()

for i in range(2, 518):
    # add a number corresponding to page number and iterate through for all recipe pages
    # has to start from 2 as the first page is formatted differently
    recipe_page = target_url + str(i)
    # get the links from each collection of recipes
    for j in get_recipe_links(recipe_page):
        # get ingredients from each recipe
        next_recipe = get_ingredients(j)
        # print statements to show what is happening while it is running
        print(j)
        print(get_ingredients(j))
        if next_recipe is not None:
            with open(file_name, 'a+', newline='') as output_file:
                # write as append - this way a fail or a crash will not erase progress
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writerow(next_recipe)
                output_file.close()
        # sleep for 1 second after every iteration to prevent banning
        time.sleep(1)
