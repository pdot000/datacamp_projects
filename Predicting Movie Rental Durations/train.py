
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error as MSE, make_scorer
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from time import time

rental_info = pd.read_csv('rental_info.csv')
seed = 9

rental_transformed = rental_info.copy()
rental_transformed['rental_length_days'] = (pd.to_datetime(rental_transformed['return_date']) - pd.to_datetime(rental_transformed['rental_date'])).dt.total_seconds() / (24 * 3600)
rental_transformed = rental_transformed.drop(['rental_date', 'return_date'], axis=1)
rental_transformed['special_features'] = rental_transformed['special_features'].replace({'{': '', '}': '', '"': ''}, regex=True)
rental_transformed = rental_transformed.drop(['amount_2', 'rental_rate_2', 'length_2'], axis=1)

X = rental_transformed.drop('rental_length_days', axis=1)
y = rental_transformed['rental_length_days'].values
cat_features = ['special_features']
num_features = [x for x in rental_transformed.columns if x not in ['special_features', 'rental_length_days']]

preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features),
    ]
)
kf = KFold(10, shuffle=True, random_state=seed)

models = {
    'Lasso': {
        'model': Lasso(random_state=seed),
        'params': {
            'alpha': np.linspace(.01, 1., 11)
        }
    },
    'Ridge': {
        'model': Ridge(random_state=seed),
        'params': {
            'alpha': np.linspace(.01, 1., 11)
        }
    },
    'AdaBoostRegressor': {
        'model': AdaBoostRegressor(random_state=seed),
        'params': {
            'n_estimators': np.arange(100, 1000, 100),
            'learning_rate': np.linspace(.01, 1., 11)
        }
    },
    'BaggingRegressor': {
        'model': BaggingRegressor(random_state=seed),
        'params': {
            'n_estimators': np.arange(100, 1000, 100),
            'max_samples': np.linspace(.1, 1., 10),
            'max_features': np.linspace(.1, 1., 10)
        }
    },
    'GradientBoostingRegressor': {
        'model': GradientBoostingRegressor(random_state=seed),
        'params': {
            'n_estimators': np.arange(100, 1000, 100),
            'max_depth': np.arange(1, 10, 2),
            'min_samples_leaf': np.arange(1, 10, 2),
        }
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=seed),
        'params': {
            'n_estimators': np.arange(100, 1000, 100),
            'max_depth': np.arange(1, 10, 2),
            'min_samples_leaf': np.arange(1, 10, 2),
        }
    }    
}


pipelines_and_grids = []
for name, config in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        (name.lower(), config['model'])
    ])
    param_grid = {f'{name.lower()}__{key}': value for key, value in config['params'].items()}
    pipelines_and_grids.append((name, pipeline, param_grid))


scores = []
for name, pipeline, param_grid in pipelines_and_grids:
    print(f'Running RandomizedSearchCV for {name}')
    start = time()
    cv = RandomizedSearchCV(n_jobs=-1, estimator=pipeline, param_distributions=param_grid, cv=kf, scoring='r2', return_train_score=True)
    cv.fit(X, y)
    for mean_score, params in  zip(cv.cv_results_['mean_test_score'], cv.cv_results_['params']):
        scores.append({
            'Model': name,
            'Mean Scores': mean_score,
            'Best Params': params
        })
    duration = round(time() - start, 1)
    m, s = int(duration // 60), round(duration % 60, 1)
    if m != 0:
        duration_printable = f'{m}m {s}s'
    else:
        duration_printable = f'{s}s'
    print(f'Finished {name} model in: {duration_printable}. Best score: {cv.best_score_}')


model_scores = pd.DataFrame(scores)
model_scores.to_csv('model_scores_r2.csv', index=False)
