# continual-aht

## TODOs

Experiment Infrastructure
- Implement wandb logging for FCP

Metrics / Logging: 
- Explore open-ended learning metrics 

Method Exploration: 
- Implement PAIRED
- Implement (L)BR-Div

Environments with Diverse Coordination Conventions: 
- Create LBF task where the objective is for the agent to collect multiple fruits in a certain order. 
    - Hand-code LBF teammate policies that collect the fruits in a certain order. 
- Explore other environments offered by Jumanji and JaxMARL.

## Installation Guide
Follow instructions at `install_instructions.md`

## Project Guide
### Tests
The `tests/` directory stores various test scripts. 

### Envs
#### Jumanji (LBF)
Currently, the only environment supported from the Jumanji suite is Level-Based Foraging (LBF).

The wrapper for the Jumanji LBF environment is stored in the `envs/` directory, at `envs/jumanji_jaxmarl_wrapper.py`. A corresponding test script is stored at `tests/test_jumanji_jaxmarl_wrapper.py`.

#### Overcooked-v2
We made some modifications to the JaxMARL Overcooked environment to improve the functionality and ensure environments are solvable.

- Initialization randomization: Previously, setting `random_reset` would lead to random initial agent positions, and randomized initial object states (e.g. pot might be initialized with onions already in it, agents might be initialized holding plates, etc.). We separate the functionality of the argument `random_reset` into two arguments: `random_reset` and `random_obj_state`, where `random_reset` only controls the initial positions of the two agents. 
- Agent initial positions: previously, in a map with disconnected components, it was possible for two agents to be spawned in the same component, making it impossible to solve the task. The Overcooked-v2 environment initializes agents such that one is always spawned on each side of the map.

#### Fictitious Co-Play (FCP)
The `fcp/` directory stores our Fictitious Co-Play implementation. This implementation was based on JaxMARL's IPPO implementation on Overcooked. Note that all FCP scripts train on LBF (using our wrapper) by default. 
- Our implementation can be run via `python fcp/fcp_pipeline.py`. Perform X-forwarding to see plots.

Note that the individual scripts referenced by `fcp/fcp_pipeline.py` can be ran individually, which is useful when debugging.
- To train IPPO agents, run `python fcp/ippo.py`. 
- To train a FCP agent, run `python fcp/fcp_train.py`. 
- To evaluate FCP agents against a pool of evaluation checkpoints, run `python fcp/fcp_evaluation.py`

### Coding Style Notes
JaxMARL follows a single-script training paradigm, which enables jit-compiling the entire RL training loop. 
We follow a similar paradigm, but importing functions to avoid code duplication. 
