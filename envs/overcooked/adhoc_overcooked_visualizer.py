import os
import numpy as np
from moviepy import ImageSequenceClip

from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer, TILE_PIXELS

class AdHocOvercookedVisualizer(OvercookedVisualizer):
    '''
    Allows highlighting a specified agent and saving MP4 videos of Overcooked episodes. 
    The original OvercookedVisualizer class only allows for saving GIFs, 
    which are not ideal for creating videos as GIFs sometimes drop frames 
    in order to optimize for file size. 
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    
    def create_agent_highlight_mask(self, grid, state, highlight_agent_idx=None):
        """Create a highlight mask for the specified agent.
        """
        h, w = grid.shape[:2]
        
        # Create a highlight mask for the specific agent
        highlight_mask = np.zeros(shape=(h, w), dtype=bool)
                # Get the agent's position from the state
        agent_pos = state.agent_pos[highlight_agent_idx]
        
        # Convert agent position to grid coordinates (accounting for padding)
        x, y = agent_pos[0] + 1, agent_pos[1] + 1
        
        # Highlight just the agent's grid space
        highlight_mask[y, x] = True
        return highlight_mask


    def render(self, agent_view_size, state, highlight_agent_idx=None, tile_size=TILE_PIXELS):
        """Override the parent's render method to add agent-specific highlighting.
        
        Args:
            agent_view_size: Size of the agent's view
            state: The environment state to render
            highlight: Whether to apply highlighting
            tile_size: Size of each tile in pixels
            
        Returns:
            The rendered image
        """
        # If no specific agent is highlighted, just use the parent's render method
        if highlight_agent_idx is None:
            # Call parent's render method which displays the image
            return super().render(agent_view_size, state, highlight=False)
        
        # Get the grid from the state
        padding = agent_view_size - 2  # show
        grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
        highlight_mask = self.create_agent_highlight_mask(grid, state, highlight_agent_idx)
        # Render the grid with the highlight mask
        img = OvercookedVisualizer._render_grid(
            grid,
            tile_size,
            highlight_mask=highlight_mask,
            agent_dir_idx=state.agent_dir_idx,
            agent_inv=state.agent_inv
        )
        
        # Display the image
        self._lazy_init_window()
        self.window.show_img(img)        
    
    def animate_mp4(self, state_seq, agent_view_size, highlight_agent_idx=None, 
                    filename="animation.mp4", fps=5, pixels_per_tile=TILE_PIXELS):
        """Animate a sequence of states and save as an MP4 file.
        
        Args:
            state_seq: Sequence of environment states to animate
            agent_view_size: Size of the agent's view
            filename: Output filename
            fps: Frames per second
            pixels_per_tile: Size of each tile in pixels
        """
        frames = []
        for state in state_seq:
            # Get the grid from the state
            padding = agent_view_size - 2  # show
            grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
            h, w = grid.shape[:2]
            
            if highlight_agent_idx is not None:
                highlight_mask = self.create_agent_highlight_mask(grid, state, highlight_agent_idx)
            
            # Render the grid with the highlight mask
            frame = OvercookedVisualizer._render_grid(
                grid,
                tile_size=pixels_per_tile,
                highlight_mask=highlight_mask if highlight_agent_idx is not None else None,
                agent_dir_idx=state.agent_dir_idx,
                agent_inv=state.agent_inv
            )
            frames.append(frame)
        
        # Check if basename directory exists, if not create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create video clip
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(filename, fps=fps, codec='libx264', audio=False, 
                             bitrate='8000k', preset='slow')

