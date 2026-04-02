import wandb
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import time

# 1. Initialize a W&B experience
run = wandb.init(
    project="iris-classification",
    name="Iris Decision Tree - From Artifact",
    job_type="training",
    config={"model": "DecisionTreeClassifier", "criterion": "gini", "max_depth": 5},
    tags=["iris", "decision-tree", "artifact"]
)

# 2. Retrieve pre-processed data from W&B
preprocessed_data_artifact = run.use_artifact("iris_preprocessed_data:v0", type="dataset")
artifact_dir = preprocessed_data_artifact.download()

X_train = pd.read_csv(f"{artifact_dir}/X_train.csv")
X_test = pd.read_csv(f"{artifact_dir}/X_test.csv")
y_train = pd.read_csv(f"{artifact_dir}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{artifact_dir}/y_test.csv").values.ravel()

# 3. Training the model
model = DecisionTreeClassifier(criterion="gini", max_depth=5)
model.fit(X_train, y_train)


# 4. Evaluate the model
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
wandb.log({"train_accuracy": train_acc, "test_accuracy": test_acc})

# 5. Visualization with wandb.sklearn
y_pred = model.predict(X_test)
y_probas = model.predict_proba(X_test)

wandb.sklearn.plot_classifier(
    model,
    X_train, X_test, y_train, y_test,
    y_pred=y_pred,
    y_probas=y_probas,
    labels=[0, 1, 2],
    model_name="DecisionTreeClassifier",
    feature_names=X_train.columns
)

# 6. Save and log the model as an artifact
model_path = "iris_decision_tree.joblib"
joblib.dump(model, model_path)

model_artifact = wandb.Artifact(
    name="iris_decision_tree_model",
    type="model",
    description="Decision Tree trained on Iris dataset"
)
model_artifact.add_file(model_path)
run.log_artifact(model_artifact)



# 7. Complete the experiment properly
time.sleep(60)  
run.finish()

