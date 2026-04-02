import pandas as pd
from sklearn.model_selection import train_test_split
import os
import wandb

# Init W&B run
run = wandb.init(project="iris-classification", job_type="data_preprocessing")

# Download and load the raw data artifact
artifact = run.use_artifact("iris_raw_data:v0", type="dataset")
artifact_path = artifact.download()
raw_data_path = os.path.join(artifact_path, "iris.csv")
df = pd.read_csv(raw_data_path)

# Preprocessing: features/labels split + train/test split
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the output directory
output_dir = "data/preprocessed"
os.makedirs(output_dir, exist_ok=True)

# Save processed datasets locally
X_train_path = os.path.join(output_dir, "X_train.csv")
X_test_path = os.path.join(output_dir, "X_test.csv")
y_train_path = os.path.join(output_dir, "y_train.csv")
y_test_path = os.path.join(output_dir, "y_test.csv")

X_train.to_csv(X_train_path, index=False)
X_test.to_csv(X_test_path, index=False)
y_train.to_csv(y_train_path, index=False)
y_test.to_csv(y_test_path, index=False)


# Log preprocessed data as a new artifact
artifact = wandb.Artifact(
    name="iris_preprocessed_data",
    type="dataset",
    description="Train/test split of the Iris dataset",
)
artifact.add_file(X_train_path)
artifact.add_file(X_test_path)
artifact.add_file(y_train_path)
artifact.add_file(y_test_path)

run.log_artifact(artifact)
run.finish()