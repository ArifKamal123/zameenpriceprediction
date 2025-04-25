from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
import mlflow
import mlflow.sklearn
from skopt.space import Real, Integer
from custom_transform import AreaUnitConverter
import joblib


def load_data(file_path):
    return pd.read_csv(file_path)


def train_model(df):
    cols_to_drop = [
        'Area Category', 'page_url', 'date_added', 'property_id', 'province_name',
        'Area Size', 'area category', 'Area Type', 'agency', 'agent'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Log transform price
    df['log_price'] = np.log1p(df['price'])
    y = df['log_price']
    X = df.drop(columns=['log_price'])

    categorical_cols = ['city', 'location', 'purpose', 'property_type']
    num_cols = ['area', 'bedrooms', 'baths']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    final_pipeline = Pipeline([
    ('unit_converter', AreaUnitConverter(column='area')),
    ('preprocessing', preprocessor),
    ('regressor', XGBRegressor(
    n_estimators=230,
    learning_rate=0.1696558547641648,
    max_depth=10,
    subsample=0.6,
    colsample_bytree=1.0,
    random_state=42,objective='reg:squarederror'))
    ])
    
    # Define hyperparameter search space
    # param_space = {
    #     'regressor__n_estimators': Integer(100, 300),
    #     'regressor__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    #     'regressor__max_depth': Integer(3, 10),
    #     'regressor__subsample': Real(0.6, 1.0),
    #     'regressor__colsample_bytree': Real(0.6, 1.0)
    # }

    # opt = BayesSearchCV(
    #     estimator=pipeline,
    #     search_spaces=param_space,
    #     n_iter=30,
    #     cv=3,
    #     scoring='neg_mean_squared_error',
    #     n_jobs=-1,
    #     verbose=1,
    #     random_state=42
    # )

    # opt.fit(X, y)
    # print("✅ Best Parameters:", opt.best_params_)
    #Best Parameters: OrderedDict({'regressor__colsample_bytree': 1.0, 'regressor__learning_rate': 0.1696558547641648,
    #'regressor__max_depth': 10, 'regressor__n_estimators': 230, 'regressor__subsample': 0.6})
    # Evaluate on test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Zameen Price Prediction")

    with mlflow.start_run():
        final_pipeline.fit(X_train, y_train)
        y_pred = final_pipeline.predict(X_test)

    # Convert predictions back to original scale
        predicted_vs_actual = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
        })
        predicted_vs_actual['Predicted_price'] = np.expm1(predicted_vs_actual['Predicted'])
        predicted_vs_actual['Actual_price'] = np.expm1(predicted_vs_actual['Actual'])

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_params({
            "n_estimators": 230,
            "learning_rate": 0.1696558547641648,
            "max_depth": 10,
            "subsample": 0.6,
            "colsample_bytree": 1.0,
        })

        mlflow.log_metrics({
            "mse":mse,
            "rmse":rmse,
            "mae":mae,
        })

        mlflow.sklearn.log_model(
            sk_model = final_pipeline,
            artifact_path = "xgb_price_pipeline",
            registered_model_name = 'xgb_price_model'
        )
        print(f"Logged to MLflow run: {mlflow.active_run().info.run_id}")
        mlflow.set_tracking_uri("file:../app/mlruns")


    # Cross-validation
    '''
    cross_val_scores = cross_val_score(final_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    cross_val_rmse = np.sqrt(-cross_val_scores)

    print("Cross-validated RMSE:", cross_val_rmse)
    print("Average RMSE:", np.mean(cross_val_rmse))
    print(predicted_vs_actual.head())
    print("MAE (log):", mae)
    print("RMSE (log):", rmse)

    joblib.dump(final_pipeline,'xgb_price_pipeline.pkl')
    print('pipeline saved successfully')
    '''
    return final_pipeline


if __name__ == "__main__":
    data_path = "../Data/zameen-updated.csv"
    df = load_data(data_path)
    train_model(df)
