from sklearn.datasets import load_iris
import pandas as pd
import os
import wandb

# Load the Iris dataset from scikit-learn
iris = load_iris(as_frame=True)
df = iris.frame

# Create the output directory if it doesn't exist
path = "data/raw/iris.csv"
os.makedirs(os.path.dirname(path), exist_ok=True)

# Save the dataset locally as a CSV file
df.to_csv(path, index=False)

# Initialize a W&B run
run = wandb.init(project="iris-classification", job_type="data_import")

# Create an artifact for the raw dataset
artifact = wandb.Artifact(
    name="iris_raw_data",
    type="dataset",
    description="Iris raw dataset",
)
artifact.add_file(path)

# Log the artifact to W&B
run.log_artifact(artifact)
run.finish()