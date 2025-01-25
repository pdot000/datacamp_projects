In this project, you will use regression models to predict the number of days a customer rents DVDs for.

As with most data science projects, you will need to pre-process the data provided, in this case, a csv file called `rental_info.csv`. Specifically, you need to:

- Read in the csv file `rental_info.csv` using ``pandas``.
- Create a column named `"rental_length_days"` using the columns `"return_date"` and `"rental_date"`, and add it to the `pandas` DataFrame. This column should contain information on how many days a DVD has been rented by a customer.
- Create two columns of dummy variables from `"special_features"`, which takes the value of `1` when:
    - The value is `"Deleted Scenes"`, storing as a column called `"deleted_scenes"`.
    - The value is `"Behind the Scenes"`, storing as a column called `"behind_the_scenes"`.
- Make a `pandas` DataFrame called X containing all the appropriate features you can use to run the regression models, avoiding columns that leak data about the target.
- Choose the `"rental_length_days"` as the target column and save it as a `pandas` Series called `y`.
---
Following the preprocessing you will need to:

- Split the data into `X_train`, `y_train`, `X_test`, and `y_test` train and test sets, avoiding any features that leak data about the target variable, and include 20% of the total data in the test set.
- Set __`random_state`__ to __`9`__ whenever you use a function/method involving randomness, for example, when doing a test-train split.

__Recommend a model yielding a mean squared error (MSE) less than 3 on the test set__

Save the model you would recommend as a variable named `best_model`, and save its MSE on the test set as `best_mse`.