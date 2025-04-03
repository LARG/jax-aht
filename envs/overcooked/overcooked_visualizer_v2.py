import os
import numpy as np
from moviepy import ImageSequenceClip

from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer

class OvercookedVisualizerV2(OvercookedVisualizer):
    '''
    This class implements saving MP4 videos of Overcooked episodes. 
    The original OvercookedVisualizer class only allows for saving GIFs, 
    which are not ideal for creating videos as GIFs sometimes drop frames 
    in order to optimize for file size. 
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def animate_mp4(self, state_seq, agent_view_size, filename="animation.mp4", 
                    pixels_per_tile=32, fps=5):
        '''
        Animate an MP4 video of the episode and save it to file.
        '''
        padding = agent_view_size - 2  # show
        def get_frame(state):
            grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
            # Render the state
            frame = OvercookedVisualizer._render_grid(
                grid,
                tile_size=pixels_per_tile,
                highlight_mask=None,
                agent_dir_idx=state.agent_dir_idx,
                agent_inv=state.agent_inv
            )
            return frame

        frame_seq = [get_frame(state) for state in state_seq]
        # Check if basename directory exists, if not create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Create video clip
        clip = ImageSequenceClip(frame_seq, fps=fps)
        clip.write_videofile(filename, codec='libx264', audio=False, 
                             bitrate='8000k', preset='slow', threads=4)

