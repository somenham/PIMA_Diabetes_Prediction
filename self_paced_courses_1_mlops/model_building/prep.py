# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# Define constants for the dataset and output paths
DATA_DIR = "self_paced_courses_1_mlops/data"
DATASET_PATH = os.path.join(DATA_DIR, "pima.csv")

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

target_col = 'class'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

os.makedirs(DATA_DIR, exist_ok=True)

Xtrain.to_csv(os.path.join(DATA_DIR, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(DATA_DIR, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(DATA_DIR, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(DATA_DIR, "ytest.csv"), index=False)

print(f"Train/test splits written to {DATA_DIR}/")
