# for data manipulation
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os

DATA_DIR = "self_paced_courses_1_mlops/data"
MODEL_DIR = "self_paced_courses_1_mlops/deployment/model"

Xtrain_path = os.path.join(DATA_DIR, "Xtrain.csv")
Xtest_path = os.path.join(DATA_DIR, "Xtest.csv")
ytrain_path = os.path.join(DATA_DIR, "ytrain.csv")
ytest_path = os.path.join(DATA_DIR, "ytest.csv")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# scale numeric features
numeric_features = [
    'preg',
    'plas',
    'pres',
    'skin',
    'test',
    'mass',
    'pedi',
    'age'
]


# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Define GB model
gb_model = GradientBoostingClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'gradientboostingclassifier__n_estimators': [75, 100, 125],
    'gradientboostingclassifier__max_depth': [2, 3, 4],
    'gradientboostingclassifier__subsample': [0.5, 0.6]
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, gb_model)

# Grid search with cross-validation
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)


# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

# Save best model directly into the deployment folder inside the repo
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "best_pima_diabetes_model_v1.joblib")
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")
