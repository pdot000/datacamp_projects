Develop a regression model using the `insurance.csv` dataset to predict `charges`. Evaluate the model's accuracy using the R-Squared Score. Then, apply the model to estimate `predicted_charges` for unseen data in `validation_dataset.csv`.

Build a regression model to predict `charges` using the `insurance.csv` dataset. Evaluate the R-Squared Score of your trained model and save it as a variable named `r2_score`. The model's success will be assessed based on its R-Squared Score, which must exceed a threshold of __0.65__.

Use the trained model to predict charges for the data in `validation_dataset.csv`. Store the predictions in a new column named `predicted_charges` within the validation dataset, and save it as a pandas DataFrame called `validation_data`. Ensure a minimum basic charge of __1000__.

⚠️ __Note__: If you encounter errors during model training, make sure the `insurance` DataFrame is properly cleaned and ready for modeling.