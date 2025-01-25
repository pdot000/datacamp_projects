Does Logistic Regression or Random Forest produce a higher accuracy score in predicting telecom churn in India?

- Load the two CSV files into separate DataFrames. Merge them into a DataFrame named `churn_df`. Calculate and print churn rate, and identify the categorical variables in `churn_df`.
- Convert categorical features in `churn_df` into `features_scaled`. Perform feature scaling separating the appropriate features and scale them. Define your scaled features and `target` variable for the churn prediction model.
- Split the processed data into training and testing sets giving names of `X_train`, `X_test`, `y_train`, and `y_test` using an 80-20 split, setting a random state of `42` for reproducibility.
- Train Logistic Regression and Random Forest Classifier models, setting a random seed of `42`. Store model predictions in `logreg_pred` and `rf_pred`.
- Assess the models on test data. Assign the model's name with higher accuracy ("`LogisticRegression`" or "`RandomForest`") to `higher_accuracy`.