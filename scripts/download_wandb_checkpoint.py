import argparse
import wandb

def download_artifact(model_name):
    run = wandb.init()
    artifact = run.use_artifact(model_name, type='model')
    artifact_dir = artifact.download()
    print(f"Artifact downloaded to: {artifact_dir}")
    return artifact_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model artifact from Weights & Biases")
    parser.add_argument("model_name", help="The name of the model artifact to download")
    args = parser.parse_args()

    download_artifact(args.model_name)

# Usage example:
# python script_name.py "ml43dss24/Nerfusion/model-unz7vi39:v0"
#
# This will download the specified model artifact and print the directory where it was saved.
# Make sure to replace "script_name.py" with the actual name of your script file.