from jumanji.environments.routing.lbf.viewer import LevelBasedForagingViewer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Tuple


class AdHocLBFViewer(LevelBasedForagingViewer):
    '''Custom viewer for the LevelBasedForaging environment that highlights the specified agent.'''
    def __init__(self,
        grid_size: int,
        name: str = "LevelBasedForaging",
        render_mode: str = "human",
        highlight_agent_idx: Optional[int] = None
    ):
        super().__init__(grid_size, name, render_mode)
        self.highlighted_agent_idx = highlight_agent_idx
    
    def set_highlighted_agent(self, agent_idx: Optional[int]):
        """Set which agent to highlight with a box.
        
        Args:
            agent_idx: Index of the agent to highlight, or None to remove highlighting
        """
        self.highlighted_agent_idx = agent_idx
    
    def _draw_agents(self, agents, ax):
        """Override the parent method to add a box around the highlighted agent."""
        # First draw all agents normally
        super()._draw_agents(agents, ax)
        
        # If we have a highlighted agent, draw a box around it
        if self.highlighted_agent_idx is not None and self.highlighted_agent_idx < len(agents.level):
            from jumanji.tree_utils import tree_slice
            
            # Get the highlighted agent
            agent = tree_slice(agents, self.highlighted_agent_idx)
            cell_center = self._entity_position(agent)
            
            # Calculate box dimensions
            box_size = self.grid_size * 0.8  # Slightly smaller than the grid cell
            box_x = cell_center[0] - box_size/2
            box_y = cell_center[1] - box_size/2
            
            # Create and add the box
            rect = patches.Rectangle(
                (box_x, box_y), 
                box_size, 
                box_size, 
                linewidth=2, 
                edgecolor='red', 
                facecolor='none',
                zorder=5  # Ensure it's drawn on top of the agent
            )
            ax.add_patch(rect)

