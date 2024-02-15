import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from challenge_utils import build_training_data, relative_squared_error, train_test_split, save_onnx, load_onnx

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

# Utilisez RandomForestRegressor au lieu de Ridge
reg_rf = RandomForestRegressor(n_estimators=100, random_state=42, verbose = 1)
reg_rf.fit(x_train, y_train)
y_pred_rf = reg_rf.predict(x_test)

RelativeMSE_rf = relative_squared_error(y_pred_rf, y_test)
print('Random Forest trained model RSE:', RelativeMSE_rf)

#  ──────────────────────────────────────────────────────────────────────────
# Recherche des coefficients optimaux (validation)

param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}

grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='neg_mean_squared_error')

grid_rf.fit(x_all, y_all) # utiliser l'ensemble du jeu de données pour la validation

best_params_rf = grid_rf.best_params_
best_estimator_rf = grid_rf.best_estimator_
y_pred_rf = best_estimator_rf.predict(x_all)
RelativeMSE_rf = relative_squared_error(y_pred_rf, y_all)
print('Best Random Forest model RSE:', RelativeMSE_rf)
print('Best parameters from validation:', best_params_rf)

#  ──────────────────────────────────────────────────────────────────────────
# Enregistrez la meilleure version du modèle RandomForest

print('Enregistrement au format ONNX')
save_onnx(best_estimator_rf, 'random_forest_model.onnx', x_train)

#  ──────────────────────────────────────────────────────────────────────────
# Chargez et exécutez le modèle RandomForest enregistré (la fonction combine les actions)

y_pred_rf_onnx = load_onnx('random_forest_model.onnx', x_all)
RelativeMSE_rf_onnx = relative_squared_error(y_pred_rf_onnx, y_all)
print('Chargé depuis le fichier ONNX RSE:', RelativeMSE_rf_onnx)
