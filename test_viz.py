import os
from envs.coop_recon_continuous.CoopReconContinuousViz import CoopReconContinuousViz

def test_viz():
    print("Testing visualizer locally...")
    
    # Create dummy states
    frames = []
    viz = CoopReconContinuousViz()
    
    # Simulate an episode of 10 steps
    for t in range(5):
        # Move agents closer to tasks
        pos_0 = (0.2 * t, 0.2 * t)
        pos_1 = (0.8, 0.2 * t)
        
        # Simulate tasks completing
        w = t >= 3
        l = t >= 4
        p = t >= 4
        
        state_layout = {
            'agent_positions': [pos_0, pos_1],
            'detected_water': w,
            'detected_life': l,
            'picture_taken': p
        }
        
        frame = viz.render(state_layout)
        frames.append(frame)
        
    os.makedirs("test_render", exist_ok=True)
    frames[0].save(
        "test_render/dummy_ep.gif",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )
    print("Test GIF generated successfully!")

if __name__ == "__main__":
    test_viz()
