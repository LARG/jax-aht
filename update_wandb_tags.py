import wandb
import re

def update_agent_tags():
    # Initialize wandb API
    api = wandb.Api()

    # Replace with your wandb entity and project name
    entity = "jeffreychen287-the-university-of-texas-at-austin"
    project = "aht-benchmark"

    print(f"Fetching runs from {entity}/{project}...")
    runs = api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} runs. Processing...")

    updated_count = 0

    for run in runs:
        tags = run.tags
        
        # Check if this run is missing an "X agents" tag
        has_agent_tag = any(tag.endswith(" agents") for tag in tags)
        
        # Also, check if it incorrectly has the "2 agents" tag despite being >2 agents
        # Since we know the previous runs incorrectly defaulted to 2
        incorrect_tag = None
        
        # Determine the TRUE number of agents
        num_agents = None
        
        # Attempt 1: Check config directly if it's there
        if "task" in run.config and "ENV_KWARGS" in run.config["task"] and "num_agents" in run.config["task"]["ENV_KWARGS"]:
            num_agents = run.config["task"]["ENV_KWARGS"]["num_agents"]
        
        # Attempt 2: Try to extract from run name, e.g. "ippo_law_0.2_3_agent"
        if num_agents is None:
            match = re.search(r'_(\d+)_agent', run.name)
            if match:
                num_agents = int(match.group(1))
                
        # Attempt 3: Try from TASK_NAME in tags e.g. CoopRecon_Compare_Law_0.2_3Agent
        if num_agents is None:
            for tag in tags:
                match = re.search(r'_(\d+)Agent', tag)
                if match:
                    num_agents = int(match.group(1))
                    break

        if num_agents is not None:
            correct_tag = f"{num_agents} agents"
            
            # Remove incorrect "2 agents" tag if it exists but the true count is != 2
            if "2 agents" in tags and num_agents != 2:
                tags.remove("2 agents")
                has_agent_tag = False
                incorrect_tag = "2 agents"
                
            # If it's missing the correct tag, add it
            if not has_agent_tag or (has_agent_tag and correct_tag not in tags):
                tags.append(correct_tag)
                run.tags = tags
                run.update()
                
                msg = f"Updated run {run.name} ({run.id}) -> added tag '{correct_tag}'"
                if incorrect_tag:
                    msg += f" (removed incorrect '{incorrect_tag}')"
                print(msg)
                updated_count += 1
        else:
            print(f"Could not determine agent count for run {run.name} ({run.id})")

    print(f"Finished! Successfully updated tags for {updated_count} runs.")

if __name__ == "__main__":
    update_agent_tags()
