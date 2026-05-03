import wandb
import re

def rename_inconsistent_runs():
    api = wandb.Api()
    entity = "jeffreychen287-the-university-of-texas-at-austin"
    project = "aht-benchmark"
    
    print(f"Fetching runs from {entity}/{project}...")
    runs = api.runs(f"{entity}/{project}")
    
    updated_count = 0
    for run in runs:
        label = run.config.get("label", "")
        
        # Match something like "ippo_law_2_agent" but NOT "ippo_law_0.2_2_agent"
        # We look for (ippo|mappo|ppo)_law_X_agent
        match = re.match(r'^(ippo|mappo|ppo)_law_(\d+)_agent$', label)
        
        if match:
            algo = match.group(1)
            num_agents = match.group(2)
            
            new_label = f"{algo}_law_0.2_{num_agents}_agent"
            print(f"Updating Run {run.id} | Old Label: {label} -> New Label: {new_label}")
            
            # 1. Update config
            run.config["label"] = new_label
            run.update()
            
            # 2. Update tags
            new_tags = [new_label if tag == label else tag for tag in run.tags]
            if label not in run.tags and new_label not in run.tags:
                new_tags.append(new_label)
            
            run.tags = new_tags
            run.update()
            
            updated_count += 1
            
    print(f"Successfully updated {updated_count} runs.")

if __name__ == "__main__":
    rename_inconsistent_runs()
