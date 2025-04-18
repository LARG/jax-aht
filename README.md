# jax-aht

## Installation Guide
Follow instructions at `install_instructions.md`

### Project Structure
- `agents/`: Contains heuristic agent implementations
- `common/`: Shared utilities and common code
- `envs/`: Environment implementations and wrappers
- `evaluation/`: Evaluation and visualization scripts
- `examples/`: Example usage scripts
- `fcp/`: Fictitious Co-Play implementation
- `ppo/`: IPPO algorithm implementation
- `tests/`: Test scripts used during development.

## Project Guide

More details about some folders are provided below. 

### Agents

The `agents` directory contains heuristic for each supported environment. 
Currently, only agents for Overcooked have been implemented.
You can run the Overcooked heuristic agent by running, `python tests/test_overcooked_agents.py`.
Later, we would want to add pretrained agents to this directory as well. 

### Envs
#### Jumanji (LBF)
Currently, the only environment supported from the Jumanji suite is Level-Based Foraging (LBF).

The wrapper for the Jumanji LBF environment is stored in the `envs/` directory, at `envs/jumanji_jaxmarl_wrapper.py`. A corresponding test script is stored at `tests/test_jumanji_jaxmarl_wrapper.py`.
`
#### Overcooked-v1
We made some modifications to the JaxMARL Overcooked environment to improve the functionality and ensure environments are solvable.

- Initialization randomization: Previously, setting `random_reset` would lead to random initial agent positions, and randomized initial object states (e.g. pot might be initialized with onions already in it, agents might be initialized holding plates, etc.). We separate the functionality of the argument `random_reset` into two arguments: `random_reset` and `random_obj_state`, where `random_reset` only controls the initial positions of the two agents. 
- Agent initial positions: previously, in a map with disconnected components, it was possible for two agents to be spawned in the same component, making it impossible to solve the task. The Overcooked-v1 environment initializes agents such that one is always spawned on each side of the map.


### Fictitious Co-Play (FCP)
The `fcp/` directory stores our Fictitious Co-Play implementation. This implementation was based on JaxMARL's IPPO implementation. 
Our full implementation can be run via `python fcp/run_fcp_pipeline.py`. Results are logged via wandb, but can also be viewed locally in the `results` directory.

- `fcp_train.py`: Training with S5 actor-critic architecture for the ego agent
- `fcp_eval.py`: Evaluation script for FCP agents
- `train_partners.py`: Script for training partner agents using IPPO.

### PPO
The `ppo/` directory stores our IPPO implementation. 
To run it with wandb logging and using the configs, run `python ppo/run_ppo.py`. 
Results are logged via wandb, but can also be viewed locally in the `results` directory.

The `ppo/ippo.py` script can also be ran on its own for debugging purposes.

### Coding Style Notes
JaxMARL follows a single-script training paradigm, which enables jit-compiling the entire RL training loop and makes it simple for researchers to modify algorithms. 
We follow a similar paradigm, but importing a couple common utility functions and training scripts to avoid code duplication. 
