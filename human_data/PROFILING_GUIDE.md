# Profiling Guide for app.py

This guide explains how to profile the `step()` function in the Flask application.

## Quick Start

### Option 1: Enable Detailed Profiling (cProfile)

Run the Flask app with profiling enabled:

```bash
ENABLE_PROFILING=true python app.py
```

This will:
- Profile every call to `GameSession.step()`
- Print a summary of the top 20 functions by cumulative time to the console
- Save detailed profile data to `profile_results/step_TIMESTAMP.prof`

### Option 2: Simple Timing (Always On)

The `simple_timer` decorator is also available if you just want timing information without the overhead of full profiling.

To use it, change the decorator on the `step()` method from `@profile_function` to `@simple_timer`.

## Understanding the Output

### Console Output

When profiling is enabled, you'll see output like this after each `step()` call:

```
================================================================================
PROFILE: step (Total time: 0.0234s)
================================================================================
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      1    0.000    0.000    0.023    0.023 app.py:145(step)
      1    0.003    0.003    0.015    0.015 env.py:234(step)
     42    0.002    0.000    0.008    0.000 {jax operations}
    ...
Full profile saved to: profile_results/step_20251118_143022.prof
================================================================================
```

### Column Meanings

- **ncalls**: Number of times the function was called
- **tottime**: Total time spent in the function (excluding subfunctions)
- **percall**: tottime / ncalls
- **cumtime**: Cumulative time (including subfunctions)
- **percall**: cumtime / ncalls
- **filename:lineno(function)**: Where the function is defined

## Analyzing Saved Profiles

### Interactive Web Viewer (Recommended)

The easiest way to analyze profiles is using the interactive web viewer:

```bash
python analyze_profile.py
```

This will:
1. Start a web server on `http://localhost:9999`
2. Automatically open your browser
3. Display an interactive interface with:
   - Dropdown to select different profile files
   - Sortable tables showing functions by cumulative time, total time, and call count
   - Search/filter functionality
   - Visual bar charts showing time distribution
   - Download option for `.prof` files

**Custom Port:**
```bash
python analyze_profile.py --port 8080
```

### Console Mode

For command-line analysis, use the `--console` flag:

```bash
# Analyze most recent profile
python analyze_profile.py --console

# Analyze specific profile
python analyze_profile.py --console step_20251118_143022.prof

# Analyze all profiles
python analyze_profile.py --console --all
```

### Python API

The `.prof` files can also be analyzed using Python's `pstats` module:

```python
import pstats

# Load the profile
p = pstats.Stats('profile_results/step_20251118_143022.prof')

# Sort by cumulative time and print top 30 functions
p.sort_stats('cumulative').print_stats(30)

# Sort by total time (time in function, not including subcalls)
p.sort_stats('tottime').print_stats(30)

# Show only functions from a specific module
p.print_stats('agents')

# Show callers and callees of a specific function
p.print_callers('step')
p.print_callees('step')
```

## Advanced: Using snakeviz for Visualization

Install snakeviz for interactive profile visualization:

```bash
pip install snakeviz
```

Then visualize a profile:

```bash
snakeviz profile_results/step_20251118_143022.prof
```

This opens an interactive browser-based visualization showing:
- Call hierarchy
- Time distribution
- Hotspots in your code

**Note:** The built-in web viewer (`python analyze_profile.py`) is simpler and doesn't require additional dependencies.

## Advanced: Using line_profiler

For line-by-line profiling, install `line_profiler`:

```bash
pip install line_profiler
```

Then add the `@profile` decorator to the `step()` method and run:

```bash
kernprof -l -v app.py
```

## Performance Tips

1. **JAX Compilation**: The first call to JAX functions is slow due to JIT compilation. Profile after the first few calls.

2. **Disable Profiling in Production**: Profiling adds overhead. Only enable when needed.

3. **Focus on Bottlenecks**: Look for functions with high `cumtime` that are called frequently.

4. **Check for Unexpected Calls**: If a function is called more times than expected, investigate why.

## Profiling Specific Scenarios

### Profile Only First Call
```python
@profile_function
def step(self, human_action):
    if self.step_count == 0:
        # First step is profiled
        ...
```

### Profile Every Nth Call
```python
@profile_function
def step(self, human_action):
    if self.step_count % 10 == 0:
        # Profile every 10th step
        ...
```

### Conditional Profiling
Set environment variable dynamically:
```python
import os
os.environ['ENABLE_PROFILING'] = 'true'
# ... make some API calls to /api/step ...
os.environ['ENABLE_PROFILING'] = 'false'
```

## Troubleshooting

### "profile_results directory not writable"
Ensure the human_data directory has write permissions:
```bash
chmod +w profile_results
```

### "No output from profiler"
- Check that `ENABLE_PROFILING=true` is set
- Verify the `step()` function is actually being called
- Check console output for error messages

### "Profiling slows down the application too much"
- Use `simple_timer` instead for lightweight timing
- Profile only specific scenarios (e.g., first call only)
- Use sampling profilers like `py-spy` for production profiling
