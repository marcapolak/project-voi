import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
import joblib

def train_and_evaluate(X_train, y_train, X_test, y_test, output_folder):
    # K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mae_scores = []
    rmse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Train the model
        lgbm_model = lgb.LGBMRegressor()
        lgbm_model.fit(X_train_fold, y_train_fold)

        # Make predictions
        y_pred_fold = lgbm_model.predict(X_test_fold)

        # Evaluate the model
        mae_scores.append(mean_absolute_error(y_test_fold, y_pred_fold))
        rmse_scores.append(mean_squared_error(y_test_fold, y_pred_fold, squared=False))
        r2_scores.append(r2_score(y_test_fold, y_pred_fold))

    # Average scores across all folds
    avg_mae = np.mean(mae_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)

    print(f"Average MAE: {avg_mae}, Average RMSE: {avg_rmse}, Average R²: {avg_r2}")

    # Training and evaluating without cross-validation
    lgbm_model_final = lgb.LGBMRegressor()
    lgbm_model_final.fit(X_train, y_train)
    y_pred = lgbm_model_final.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"Final Model MAE: {mae}, RMSE: {rmse}, R²: {r2}")
    
    # Save the trained LightGBM model
    lgbm_model_path = os.path.join(output_folder, 'trained_model.txt')
    lgbm_model_final.booster_.save_model(lgbm_model_path)
    print(f"Final LightGBM Model saved at: {lgbm_model_path}")

    # Save the trained scikit-learn model
    sklearn_model_path = os.path.join(output_folder, 'model.pkl')
    joblib.dump(lgbm_model_final, sklearn_model_path)
    print(f"Final Scikit-learn Model saved at: {sklearn_model_path}")


