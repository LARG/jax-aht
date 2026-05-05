import wandb
api = wandb.Api()
runs = api.runs("jeffreychen287-the-university-of-texas-at-austin/aht-benchmark")
for run in runs:
    if "ippo_law_" in run.name and "0.2" not in run.name and "0.0" not in run.name and "0.1" not in run.name:
        print(f"Found run: {run.name}")
        break
