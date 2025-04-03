import os
import pickle
import orbax.checkpoint
from flax.training import orbax_utils
import shutil


#################### LOAD SAVED PICKLE PYTREE
basepath = "/scratch/cluster/clw4542/explore_marl/continual-aht/results"
train_state_path = f"{basepath}/lbf/debug/2025-03-17_23-12-43/fcp_train.pkl"
with open(train_state_path, "rb") as f:
    out = pickle.load(f) # out is a pytree
print("Loaded pickle pytree.")
print(out.keys())
print("final_params type:", type(out['final_params']))
print("final_params keys:", out['final_params']['params'].keys())
print("action_body_0 type:", type(out['final_params']['params']['action_body_0']))
print("action_body_0 keys:", out['final_params']['params']['action_body_0'].keys())
print("bias type:", type(out['final_params']['params']['action_body_0']['bias']))

print("metrics keys:", out['metrics'].keys())
print("percent_eaten type:", type(out['metrics']['percent_eaten']))

############### SAVE PYTREE USING ORBAX
# Use a simpler path for testing
new_ckpt_dir = f"{basepath}/pickle2orbax_test"
print(f"Attempting to save to directory: {new_ckpt_dir}")

# Remove existing directory if it exists
if os.path.exists(new_ckpt_dir):
    print(f"Removing existing directory: {new_ckpt_dir}")
    shutil.rmtree(new_ckpt_dir)
    print("Directory removed successfully")

# Initialize the Orbax checkpointer
print("Initializing Orbax checkpointer")
checkpointer = orbax.checkpoint.PyTreeCheckpointer()

# Get save arguments for the pytree
print("Generating save arguments")
save_args = orbax_utils.save_args_from_target(out)

# Save the checkpoint
print("Saving checkpoint...")
checkpointer.save(new_ckpt_dir, out, save_args=save_args)
print(f"Saved checkpoint to {new_ckpt_dir}")

# Optional: Verify the checkpoint can be restored
print("Attempting to restore checkpoint...")
restored = checkpointer.restore(new_ckpt_dir)
print("Successfully restored checkpoint")



