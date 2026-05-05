#!/usr/bin/env python3
"""
Debug script for pizza_v2 action preconditions.

Shows which individual preconditions are satisfied or failing for each action.
Supports manual action selection for testing specific action sequences.

Usage:
    # Analyze at initial state
    python test_pizza_v2_preconditions.py --seed 0 --max-steps 0 --mode summary
    
    # Manually choose actions for 3 steps, then analyze
    python test_pizza_v2_preconditions.py --seed 42 --max-steps 3 --mode detailed
    
    # Full interactive mode - choose actions until episode ends
    python test_pizza_v2_preconditions.py --seed 0 --max-steps -1
    
    # Different instance
    python test_pizza_v2_preconditions.py --instance pizza_v2_instance_0.rddl --max-steps 0
"""


import argparse
import os
from typing import Dict, List, Tuple, Optional, Any
import math

import jax
import jax.numpy as jnp
import numpy as np

from envs import make_env





class PreconditionDebugger:
    """Debugs action preconditions by checking each one individually."""
    
    def __init__(self, env):
        """Initialize with environment.
        
        Args:
            env: PizzaWrapper environment instance
        """
        self.env = env
        self.underlying_env = env.env  # JaxRDDLEnv
        self.num_agents = env.num_agents
        self.agents = env.agents
        self.rddl_action_keys = env.rddl_action_keys
        self.model = self.underlying_env.model
        self.locations = self.model.type_to_objects.get('location', [])
        self.trucks = self.model.type_to_objects.get('truck', [])

    @staticmethod
    def _normalize_error(error: Any) -> str:
        """Convert precondition error payloads (including JAX arrays) to text."""
        if error is None:
            return ""
        if isinstance(error, str):
            return error
        try:
            arr = np.asarray(error)
            if arr.shape == ():
                return str(arr.item())
            return np.array2string(arr, threshold=8)
        except Exception:
            return str(error)
    
    def check_action_preconditions(self, env_state, agent_idx: int, 
                                   action_flat_idx: int) -> Dict[str, Any]:
        """Check which preconditions pass/fail for a specific action.
        
        Args:
            env_state: Current EnvState
            agent_idx: Agent index
            action_flat_idx: Flat action index in agent's action space
            
        Returns:
            Dict with precondition results and state info
        """
        # Extract base subs from state
        base_subs = env_state.subs
        
        # Convert flat action index to RDDL action format
        action_indices = jnp.zeros(self.num_agents, dtype=jnp.int32)
        action_indices = action_indices.at[agent_idx].set(action_flat_idx)
        
        # Action-num is the single action fluent
        action_dict = {self.rddl_action_keys[0]: action_indices}
        
        # Create test subs with this action
        test_subs = {**base_subs, **action_dict}
        
        # Get model_aux (auxiliary parameters) - try multiple possible locations
        model_aux = None
        if hasattr(self.underlying_env, 'model_aux'):
            model_aux = self.underlying_env.model_aux
        elif hasattr(self.underlying_env, '_model_aux'):
            model_aux = self.underlying_env._model_aux
        
        # Get RNG key (doesn't matter much for preconditions)
        key = jax.random.PRNGKey(0)
        
        # Iterate through preconditions and check each one
        precond_results = []
        if hasattr(self.underlying_env, '_jit_preconditions'):
            for precond_idx, precond_fn in enumerate(self.underlying_env._jit_preconditions):
                try:
                    result, key, error, model_aux = precond_fn(test_subs, model_aux, key)
                    is_satisfied = bool(np.asarray(result))
                    precond_results.append({
                        'index': precond_idx,
                        'satisfied': is_satisfied,
                        'error': self._normalize_error(error),
                    })
                except Exception as e:
                    precond_results.append({
                        'index': precond_idx,
                        'satisfied': False,
                        'error': f"Exception: {str(e)[:100]}",
                    })
        
        # Overall validity
        overall_valid = all(p['satisfied'] for p in precond_results)
        
        # Decode which action this is
        action_info = self._decode_action(action_flat_idx, agent_idx)
        
        return {
            'agent_idx': agent_idx,
            'action_flat_idx': action_flat_idx,
            'action_name': action_info['name'],
            'action_params': action_info['params'],
            'overall_valid': overall_valid,
            'preconditions': precond_results,
        }
    
    def _decode_action(self, flat_idx: int, agent_idx: int) -> Dict[str, Any]:
        """Decode flat action index to action name and parameters.
        
        Args:
            flat_idx: Flat action index
            agent_idx: Agent index
            
        Returns:
            Dict with 'name' and 'params' keys
        """
        if flat_idx == 0:
            return {'name': 'noop', 'params': None}
        if flat_idx == 1:
            return {'name': 'load', 'params': None}
        if flat_idx == 2:
            return {'name': 'deliver', 'params': None}
        if flat_idx >= 3:
            return {'name': 'drive', 'params': (flat_idx - 2,)}

        return {'name': 'unknown', 'params': (flat_idx,)}
    
    def analyze_all_actions(self, env_state) -> Dict[str, Any]:
        """Analyze all actions for all agents.
        
        Args:
            env_state: Current EnvState
            
        Returns:
            Analysis results
        """
        results = {'agents': {}}
        
        for agent_idx, agent_name in enumerate(self.agents):
            # Get total actions for this agent
            n_actions = int(self.env.action_space(agent_name).n)
            
            agent_results = []
            for action_idx in range(n_actions):
                result = self.check_action_preconditions(env_state, agent_idx, action_idx)
                agent_results.append(result)
            
            results['agents'][agent_name] = agent_results
        
        return results


class ReportFormatter:
    """Formats precondition check output."""
    
    def __init__(self, debugger: PreconditionDebugger):
        self.debugger = debugger
        self.locations = debugger.locations
    
    def format_action_name(self, action_name: str, params: Optional[Tuple]) -> str:
        """Format action with parameters."""
        if params is None:
            return f"{action_name}(...)"
        
        param_strs = []
        for i, p in enumerate(params):
            if i == 0 and action_name == 'drive':
                param_strs.append(f"conn_{p}")
            else:
                param_strs.append(str(p))
        
        return f"{action_name}({', '.join(param_strs)})"
    
    def format_summary(self, results: Dict[str, Any]) -> str:
        """Format summary output."""
        lines = []
        lines.append("=" * 80)
        lines.append("PIZZA V2 ACTION PRECONDITION SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        for agent_name, agent_actions in results['agents'].items():
            valid_count = sum(1 for a in agent_actions if a['overall_valid'])
            total_count = len(agent_actions)
            
            lines.append(f"{agent_name.upper()}: {valid_count}/{total_count} actions valid")
            lines.append("-" * 80)
            
            # Valid actions
            valid_actions = [a for a in agent_actions if a['overall_valid']]
            if valid_actions:
                lines.append("✓ VALID:")
                for action in valid_actions:
                    action_str = self.format_action_name(action['action_name'], action['action_params'])
                    passed = sum(1 for p in action['preconditions'] if p['satisfied'])
                    total = len(action['preconditions'])
                    lines.append(f"  • {action_str:30s} ({passed}/{total} preconditions pass)")
            
            # Invalid actions
            invalid_actions = [a for a in agent_actions if not a['overall_valid']]
            if invalid_actions:
                lines.append("")
                lines.append("✗ INVALID:")
                for action in invalid_actions:
                    action_str = self.format_action_name(action['action_name'], action['action_params'])
                    lines.append(f"  • {action_str:30s}")
                    
                    # Show which preconditions failed
                    failed_preconds = [p for p in action['preconditions'] if not p['satisfied']]
                    for p in failed_preconds:
                        error_text = str(p.get('error', ''))
                        error_str = error_text[:60] if error_text else "Unknown reason"
                        lines.append(f"      ✗ Precondition #{p['index']}: {error_str}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def format_detailed(self, results: Dict[str, Any]) -> str:
        """Format detailed output."""
        lines = []
        lines.append("=" * 80)
        lines.append("PIZZA V2 ACTION PRECONDITION DETAILED REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        for agent_name, agent_actions in results['agents'].items():
            lines.append(f"{agent_name.upper()}")
            lines.append("-" * 80)
            
            for i, action in enumerate(agent_actions):
                action_str = self.format_action_name(action['action_name'], action['action_params'])
                status = "✓ VALID" if action['overall_valid'] else "✗ INVALID"
                
                lines.append(f"\n[{i:2d}] {status} — {action_str}")
                lines.append(f"      Flat index: {action['action_flat_idx']}")
                
                # List all preconditions
                lines.append(f"      Preconditions ({len(action['preconditions'])} total):")
                for p in action['preconditions']:
                    symbol = "✓" if p['satisfied'] else "✗"
                    error_text = str(p.get('error', ''))
                    error_info = f" — {error_text[:70]}" if error_text else ""
                    lines.append(f"        {symbol} #{p['index']}{error_info}")
            
            lines.append("")
        
        return "\n".join(lines)


def parse_action_input(action_spec: str, agent_idx: int, env) -> Optional[int]:
    """Parse user input into an action index.
    
    Args:
        action_spec: User input (can be flat index or "action_name(param1, param2)")
        agent_idx: Agent index
        env: Environment
        
    Returns:
        Flat action index or None if invalid
    """
    action_spec = action_spec.strip()
    
    # Try parsing as integer directly
    try:
        flat_idx = int(action_spec)
        n_actions = int(env.action_space(env.agents[agent_idx]).n)
        if 0 <= flat_idx < n_actions:
            return flat_idx
        else:
            print(f"  Error: Action index {flat_idx} out of range [0, {n_actions})")
            return None
    except ValueError:
        pass
    
    # Try parsing as action name
    print(f"  Action '{action_spec}' not recognized as index. Please use numeric action indices.")
    return None


def show_available_actions(env, agent_idx: int, debugger: Optional[PreconditionDebugger] = None,
                          env_state = None) -> None:
    """Display available actions for an agent.
    
    Args:
        env: Environment
        agent_idx: Agent index
        debugger: Optional PreconditionDebugger to show validity
        env_state: Optional EnvState to check preconditions
    """
    agent_name = env.agents[agent_idx]
    n_actions = int(env.action_space(agent_name).n)
    
    print(f"\n  Actions for {agent_name} (0 to {n_actions - 1}):")
    
    for flat_idx in range(n_actions):
        if flat_idx == 0:
            action_str = "noop(...)"
        elif flat_idx == 1:
            action_str = "load(...)"
        elif flat_idx == 2:
            action_str = "deliver(...)"
        else:
            action_str = f"drive(conn_{flat_idx - 2})"
        
        # Show validity if available
        validity_str = ""
        if debugger and env_state:
            result = debugger.check_action_preconditions(env_state, agent_idx, flat_idx)
            validity_str = " ✓" if result['overall_valid'] else " ✗"
        
        print(f"    [{flat_idx:2d}] {action_str:40s}{validity_str}")


def interactive_mode(env, debugger: PreconditionDebugger) -> Tuple[Dict, Any]:
    """Interactive step-by-step action selection.
    
    Args:
        env: Environment
        debugger: PreconditionDebugger
        
    Returns:
        Tuple of (state, env_state) at the end
    """
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    step = 0
    
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE - Choose actions at each step")
    print("=" * 80)
    print("Commands: Enter action index, 'show' to list actions, 'check' to analyze")
    print("          'step <n>' to take n steps, 'quit' to stop")
    print()
    
    while True:
        print(f"\n--- STEP {step} ---")
        
        # Get actions from user for each agent
        actions = {}
        all_valid = True
        
        for agent_idx, agent_name in enumerate(env.agents):
            while True:
                prompt = f"  {agent_name} action (0-{int(env.action_space(agent_name).n) - 1}, or 'show'): "
                user_input = input(prompt).strip().lower()
                
                if user_input == "show":
                    show_available_actions(env, agent_idx, debugger, state.env_state)
                    continue
                elif user_input == "check":
                    # Analyze preconditions at current state
                    print("\n  Analyzing preconditions at current state...")
                    results = debugger.analyze_all_actions(state.env_state)
                    formatter = ReportFormatter(debugger)
                    print(formatter.format_summary(results))
                    continue
                elif user_input == "quit":
                    return state, state.env_state
                
                action_idx = parse_action_input(user_input, agent_idx, env)
                if action_idx is not None:
                    actions[agent_name] = action_idx
                    break
                print("  Please enter a valid action index.")
        
        # Take step
        rng, step_key = jax.random.split(rng)
        obs, state, rewards, dones, info = env.step(step_key, state, actions)
        step += 1
        
        print(f"  Step taken. Done={dones['__all__']}")
        print(f"  Rewards: {rewards}, Info: {info}")
        
        if dones['__all__']:
            print("  Episode finished!")
            break
    
    return state, state.env_state


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Debug action preconditions in pizza_v2"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=0,
                       help="Steps to roll forward before checking (use -1 for interactive mode)")
    parser.add_argument("--instance", type=str, default="pizza_v2_instance_all.rddl",
                       help="Instance file")
    parser.add_argument("--mode", choices=["summary", "detailed"], default="summary",
                       help="Output format")
    parser.add_argument("--output-dir", type=str, default="results/pizza_v2_preconditions",
                       help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PIZZA V2 PRECONDITION DEBUGGER")
    print("=" * 80)
    print(f"Seed: {args.seed}, Mode: {args.mode}\n")
    
    # Load environment
    print("Loading environment...")
    env = make_env("rddl/pizza_v2", {"render": False, "instance": args.instance})
    
    # Initialize debugger
    debugger = PreconditionDebugger(env)
    
    # Interactive mode
    if args.max_steps == -1:
        state, final_env_state = interactive_mode(env, debugger)
    else:
        # Reset
        rng = jax.random.PRNGKey(args.seed)
        obs, state = env.reset(rng)
        
        # Roll forward if needed
        if args.max_steps > 0:
            print(f"Rolling forward {args.max_steps} steps manually...")
            for step_num in range(args.max_steps):
                print(f"\n--- STEP {step_num} ---")
                
                # Get actions from user for each agent
                actions = {}
                for agent_idx, agent_name in enumerate(env.agents):
                    while True:
                        prompt = f"  {agent_name} action (0-{int(env.action_space(agent_name).n) - 1}, or 'show'): "
                        user_input = input(prompt).strip().lower()
                        
                        if user_input == "show":
                            show_available_actions(env, agent_idx, debugger, state.env_state)
                            continue
                        
                        action_idx = parse_action_input(user_input, agent_idx, env)
                        if action_idx is not None:
                            actions[agent_name] = action_idx
                            break
                        print("  Please enter a valid action index.")
                
                # Take step
                rng, step_key = jax.random.split(rng)
                obs, state, rewards, dones, info = env.step(step_key, state, actions)
                print(f"  Step taken.")
                print(f"  Rewards: {rewards}, Info: {info}")
                
                if dones['__all__']:
                    print("  Episode finished!")
                    break
        
        final_env_state = state.env_state
    
    # Analyze
    print("\nAnalyzing preconditions at final state...")
    results = debugger.analyze_all_actions(final_env_state)
    
    # Format and output
    formatter = ReportFormatter(debugger)
    if args.mode == "detailed":
        output = formatter.format_detailed(results)
    else:
        output = formatter.format_summary(results)
    
    print(output)
    
    # Save
    output_file = os.path.join(args.output_dir, f"preconditions_seed{args.seed}.txt")
    with open(output_file, 'w') as f:
        f.write(output)
    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
