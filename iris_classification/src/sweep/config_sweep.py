import yaml
import joblib
import wandb

run = wandb.init(
    project="iris-classification",
    job_type="sweep_config_upload",
    name="Upload sweep config"
)

sweep_config_artifact = wandb.Artifact(
    name="iris_sweep_config",
    type="config",
    description="Configuration YAML pour l’optimisation des hyperparamètres"
)

config_path = "src/sweep/config_sweep.yaml"
sweep_config_artifact.add_file(config_path)

run.log_artifact(sweep_config_artifact)

run.finish()