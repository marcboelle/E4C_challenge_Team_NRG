from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from challenge_utils import build_training_data, relative_squared_error, train_test_split, save_onnx, load_onnx
import time 

student_data_path = 'students_drahi_production_consumption_hourly.csv'

target_time, targets, predictors = build_training_data(student_data_path)

ntot = len(targets)
x_all = predictors.reshape(ntot, -1)
y_all = targets

# separating train/test sets
n = 400
test_ind = np.arange(n, len(targets))
print(len(test_ind))

x_train, y_train, x_test, y_test = train_test_split(predictors, targets, test_ind)
print(len(y_train), len(y_test))
#  ──────────────────────────────────────────────────────────────────────────
# Simple modèle de régression RandomForest
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

time_start = time.time()
reg_rf = GradientBoostingRegressor(n_estimators=100, random_state=42, verbose = 1)
reg_rf.fit(x_train, y_train)
y_pred_rf = reg_rf.predict(x_test)
print('time pour entrainement :', time.time() - time_start)

RelativeMSE_rf = relative_squared_error(y_pred_rf, y_test)
print('Gradient Boosting trained model RSE:', RelativeMSE_rf)

#  ──────────────────────────────────────────────────────────────────────────
# Recherche des coefficients optimaux (validation)

custom_scorer = make_scorer(relative_squared_error, greater_is_better=False)

param_grid_gb = {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 4, 5]}

#grid_rf = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1, verbose=1), param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=5, scoring=custom_scorer, verbose=10, n_jobs=-1)

grid_gb.fit(x_all, y_all) # utiliser l'ensemble du jeu de données pour la validation

best_params_gb = grid_gb.best_params_
best_estimator_gb = grid_gb.best_estimator_
y_pred_rf = best_estimator_gb.predict(x_all)
RelativeMSE_gb = relative_squared_error(y_pred_rf, y_all)
print('Best Gradient boosting model RSE:', RelativeMSE_gb)
print('Best parameters from validation:', best_params_gb)

#  ──────────────────────────────────────────────────────────────────────────
# Enregistrez la meilleure version du modèle RandomForest

print('Enregistrement au format ONNX')
save_onnx(best_estimator_gb, 'gradient_boosting_model.onnx', x_train)

#  ──────────────────────────────────────────────────────────────────────────
# Chargez et exécutez le modèle RandomForest enregistré (la fonction combine les actions)

y_pred_gb_onnx = load_onnx('gradient_boosting_model.onnx', x_all)
RelativeMSE_gb_onnx = relative_squared_error(y_pred_gb_onnx, y_all)
print('Chargé depuis le fichier ONNX RSE:', RelativeMSE_gb_onnx)