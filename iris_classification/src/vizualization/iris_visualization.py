import pandas as pd
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Initialize W&B for this visualization job
run = wandb.init(
    project="iris-classification",
    name="Iris visualizations rapports",
    job_type="visualization"
)

# Use the artifact containing the pre-processed dataset
artifact = run.use_artifact("iris_preprocessed_data:v0", type="dataset")
artifact_dir = artifact.download()

# Load training data from artifact
X_train = pd.read_csv(os.path.join(artifact_dir, "X_train.csv"))
y_train = pd.read_csv(os.path.join(artifact_dir, "y_train.csv"))

df = pd.concat([X_train, y_train], axis=1)

# Log data into W&B
run.log({
    "iris_table": wandb.Table(dataframe=df),
    "summary_stats": wandb.Table(dataframe=df.describe().transpose())
})

summary_stats = df.describe().transpose()
run.log({"summary_stats": wandb.Table(dataframe=summary_stats)})

# 1. Pairplot 
sns_plot = sns.pairplot(df, hue="target", diag_kind="hist")
sns_plot.fig.suptitle("Iris Pairplot - Relations entre caractéristiques",
                      y=1.02)
run.log({"1_pairplot": wandb.Image(sns_plot.fig)})
plt.close(sns_plot.fig)

# 2. Correlation matrix 
plt.figure(figsize=(10, 8))
corr_matrix = df.select_dtypes(include=['float64']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title("Matrice de corrélation des caractéristiques Iris")
run.log({"2_correlation_matrix": wandb.Image(plt.gcf())})
plt.close()

# 3. Boxplots 
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
features = ['sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
            'petal width (cm)']

for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    sns.boxplot(data=df, x='target', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution de {feature}')
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
run.log({"3_boxplots_features": wandb.Image(plt.gcf())})
plt.close()

# 4. Scatter plot 
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)',
                hue='target', s=80)
plt.title('Longueur vs Largeur des pétales')


plt.tight_layout()
run.log({"4_scatter_plots": wandb.Image(plt.gcf())})
plt.close()

# Close W&B session
run.finish()