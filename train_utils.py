import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, BayesianRidge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Tuple, List
# import statsmodels as sm
import statsmodels.api as sm
from scipy.stats import norm
import shap
from eda_utils import min_max_scaling
from sklearn.compose import TransformedTargetRegressor


def get_model(model_name: str, **kwargs):
    if model_name == 'svr':
        return SVR(**kwargs)
    elif model_name == 'rf':
        return RandomForestRegressor(**kwargs)
    elif model_name == 'lr':
        return LinearRegression(**kwargs)
    elif model_name == 'elf':
        return ElasticNet(**kwargs)
    elif model_name == 'ridge':
        return Ridge(**kwargs)
    elif model_name == 'lasso':
        return Lasso(**kwargs)
    elif model_name == 'bayes':
        return BayesianRidge()
    return xgb.XGBRegressor(**kwargs)

def train(model_name: str, df: pd.DataFrame, group_kwargs: Dict={}, test_size=0.2, biomass_factor=10) -> Tuple[Dict, Dict]:
    regression_models = {}
    preds_real_y = {}

    for group_num in df['group_num'].unique():
        group_df = df[df['group_num'] == group_num]
        X = group_df.drop(['sum_biomass_ug_ml', 'group_num'], axis=1)
        y = group_df['sum_biomass_ug_ml'] * biomass_factor
        if test_size == 0:
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = get_model(model_name, **group_kwargs.get(group_num, {}))
        model = Pipeline([('scaler', StandardScaler()), ('model', model)])
        # model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
        model.fit(X_train, y_train)
        regression_models[group_num] = model
        
        if test_size != 0:
            y_pred = model.predict(X_test)
            preds_real_y[group_num] = {'real': y_test, 'preds': y_pred}
        
    return regression_models, preds_real_y

def train_iterative(model_name: str, df: pd.DataFrame, group_order: List[int], group_kwargs: Dict={}, test_size=0.2, biomass_factor=10) -> Tuple[Dict, Dict]:
    regression_models = {}
    preds_real_y = {}

    for group_num in group_order:  # Use the provided order for group iteration
        group_df = df[df['group_num'] == group_num]
        X = group_df.drop(['sum_biomass_ug_ml', 'group_num'], axis=1)
        y = group_df['sum_biomass_ug_ml'] * biomass_factor
        
        # Add previous predictions as features
        for prev_group_num in regression_models.keys():
            prev_model = regression_models[prev_group_num]
            prev_preds = prev_model.predict(X)
            X[f'preds_group_{prev_group_num}'] = prev_preds
        
        if test_size == 0:
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = get_model(model_name, **group_kwargs.get(group_num, {}))
        model = Pipeline([('scaler', MinMaxScaler()), ('model', model)])
        # model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
        model.fit(X_train, y_train)
        regression_models[group_num] = model
        
        if test_size != 0:
            y_pred = model.predict(X_test)
            preds_real_y[group_num] = {'real': y_test, 'preds': y_pred}
        
    return regression_models, preds_real_y

def grid_search_cv(model_name: str, df: pd.DataFrame, test_size=0.2, param_grid: Dict = {}) -> Dict:
    best_params_per_group = {}

    for group_num in df['group_num'].unique():
        group_df = df[df['group_num'] == group_num]
        X = group_df.drop(['sum_biomass_ug_ml', 'group_num'], axis=1)
        y = group_df['sum_biomass_ug_ml']

        # Splitting with random state so the split is permenant but the CV does not see the validation set
        X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=42)

        # Set up the parameter grid for the grid search
        grid_search = GridSearchCV(
            estimator=Pipeline([('scaler', MinMaxScaler()), ('model', get_model(model_name))]),
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  # Negative MSE as scoring metric
            cv=5,
            verbose=10,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_params_per_group[group_num] = best_params
        
    return best_params_per_group


def eval_test_iterative(regression_models: Dict, test_df: pd.DataFrame, group_order: List[int], biomass_factor=10) -> None:
    preds_real_y = {}
    
    for i, group_num in enumerate(group_order):
        group_df = test_df[test_df['group_num'] == group_num]
        model = regression_models[group_num]
        
        # Add previous predictions as features
        X = group_df.drop(['sum_biomass_ug_ml', 'group_num'], axis=1)
        for prev_group_num in group_order[:i]:
            prev_model = regression_models[prev_group_num]
            prev_preds = prev_model.predict(X)
            X[f'preds_group_{prev_group_num}'] = prev_preds
        
        y_pred = model.predict(X)
        y_test = group_df['sum_biomass_ug_ml'] * biomass_factor

        preds_real_y[group_num] = {'real': y_test, 'preds': y_pred}
    
    eval_preds(preds_real_y)
    
def eval_test(regression_models: Dict, test_df: pd.DataFrame, biomass_factor=10) -> None:
    preds_real_y = {}
    for group_num in regression_models.keys():
        group_df = test_df[test_df['group_num'] == group_num]
        model = regression_models[group_num]
        y_pred = model.predict(group_df.drop(['sum_biomass_ug_ml', 'group_num'], axis=1))
        y_test = group_df['sum_biomass_ug_ml'] * biomass_factor

        preds_real_y[group_num] = {'real': y_test, 'preds': y_pred}
    
    eval_preds(preds_real_y)

def eval_preds(preds_real_y: Dict) -> None:
    total_r2 = 0
    total_mse = 0
    for group_num, values in preds_real_y.items():
        y_test = values['real']
        y_pred = values['preds']
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        total_r2 += r2
        total_mse += mse
        
        # Plotting actual values vs. predicted values
        plt.scatter(y_test, y_pred, color='b', alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Group {group_num} - Actual vs. Predicted')
        plt.show()

        print(f"Results for group_num {group_num}:")
        print(f"MSE: {mse}")
        print(f"R-squared: {r2}\n")
    print(f"Total MSE: {total_mse/len(preds_real_y.keys())}, Total R-squared: {total_r2/len(preds_real_y.keys())}")

def residual_analysis(df: pd.DataFrame, regression_models: Dict, biomass_factor=10) -> None:
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15,20))
    for i, group_num in enumerate(regression_models.keys()):
        ax_row = i % 7
        group_df = df[df['group_num'] == group_num]
        model = regression_models[group_num]
        group_y_pred = model.predict(group_df.drop(['sum_biomass_ug_ml', 'group_num'], axis=1))
        group_y_test = group_df['sum_biomass_ug_ml'] * biomass_factor

        residuals = group_y_test - group_y_pred

        # Residuals plot
        ax = axes[ax_row, 0]
        ax.scatter(group_y_pred, residuals)
        ax.axhline(0,0, color='r')
        ax.set_title(f'Residuals analysis for Group {group_num}')
        ax.set_xlabel('y_predicted')
        ax.set_ylabel('residual')

        # QQ-Plot for the residuals
        ax = axes[ax_row, 1]
        sm.qqplot(residuals, norm, fit=True, line="45", ax=ax)
        ax.set_title('Residuals QQ-Plot')

    plt.tight_layout()
    plt.show()

def plot_shap_values(df: pd.DataFrame, models_dict: Dict, df_test: pd.DataFrame) -> Dict:    
    # Initialize a list to store the Shapley values for each model
    shap_values_list = {}
    
    # Iterate through the models in the dictionary
    for group_num, model in models_dict.items():
        # Get the model's predictions for the current group_num dataset
        # Convert the dataframe to a matrix for shap values computation
        group_rows = df[df['group_num'] == group_num].drop(['group_num', 'sum_biomass_ug_ml'], axis=1)
        X = group_rows.values
        
        # Compute the model's predictions for the current group_num dataset
        y_pred = model.predict(X)
        
        # Compute the Shapley values for the model
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        
        # Store the Shapley values for this model in the list
        shap_values_list[group_num] = shap_values
    
    # Plot the Shapley values for each feature and each model
    for group_num, shap_values in shap_values_list.items():
        plt.figure(figsize=(10, 6))
        group_rows = df_test[df_test['group_num'] == group_num].drop(['group_num', 'sum_biomass_ug_ml'], axis=1)
        X = group_rows.values
        shap.summary_plot(shap_values, features=X, feature_names=group_rows.columns, plot_type='bar', show=False)
        plt.title(f"Shapley Values - Group {group_num}")
        plt.show()
        
    return shap_values_list

def compare_to_fluor(regression_models: Dict, df: pd.DataFrame, fluor_groups_map: Dict, fluor_test_df: pd.DataFrame, biomass_factor=10) -> None:
    # Visualize predictions along with test points
    fig, axes = plt.subplots(len(fluor_groups_map), 2, figsize=(13, 20))

    for i, group_num in enumerate(fluor_groups_map.keys()):
        group_X_test = df[df['group_num'] == group_num]
        group_y_test = df[df['group_num'] == group_num]['sum_biomass_ug_ml'] * biomass_factor
        group_y_fluor_pred = fluor_test_df[fluor_test_df['group_num'] == group_num][fluor_groups_map[group_num]]
        
        # Scale 'group_y_test' and 'group_y_fluor_pred' to the same scale
        group_y_test_scaled = min_max_scaling(group_y_test)
        group_y_fluor_pred_scaled = min_max_scaling(group_y_fluor_pred)
    
        model = regression_models[group_num]
        group_y_pred = model.predict(group_X_test.drop(['sum_biomass_ug_ml', 'group_num'], axis=1))

        # Create a scatter plot to compare predicted values and actual test values
        axes[i, 0].scatter(group_y_test, group_y_pred, color='b', alpha=0.5)
        axes[i, 0].plot([group_y_test.min(), group_y_test.max()], [group_y_test.min(), group_y_test.max()], 'r--', lw=2)  # Add a diagonal line for reference
        axes[i, 0].set_xlabel('Actual Test Values')
        axes[i, 0].set_ylabel('Predicted Values')
        axes[i, 0].set_title(f'Group {group_num} - Actual vs. Predicted')

        # Create a scatter plot to compare fluorprobe's predicted values and actual test values
        axes[i, 1].scatter(group_y_test_scaled, group_y_fluor_pred_scaled, color='b', alpha=0.5)
        axes[i, 1].plot([group_y_test_scaled.min(), group_y_test_scaled.max()], [group_y_test_scaled.min(), group_y_test_scaled.max()], 'r--', lw=2)  # Add a diagonal line for reference
        axes[i, 1].set_xlabel('Actual Test Values')
        axes[i, 1].set_ylabel('Fluor Predicted Values')
        axes[i, 1].set_title(f'Group {group_num} - Actual vs. Fluor Predicted')

    plt.show()
