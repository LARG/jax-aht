"""
Visualizer for the Continuous Cooperative Reconnaissance Domain

This visualizer displays:
- A 1.0 x 1.0 continuous observation field
- Multiple rover agents plotted at their (x, y) coordinates
- Static feature locations (Water: 1.0, 1.0) and (Life: 0.0, 1.0)
- A scoreboard showing the status of Water, Life, and Picture tasks
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

class CoopReconContinuousViz:
    # Colors for different agents
    AGENT_COLORS = [
        '#4169E1',  # Royal blue
        '#FF6347',  # Tomato red
        '#32CD32',  # Lime green
        '#FFD700',  # Gold
    ]

    def __init__(self, dpi=100, fontsize=10) -> None:
        """
        Initialize the Continuous CoopRecon Visualizer.

        Args:
            dpi: Dots per inch for the figure
            fontsize: Font size for text labels
        """
        self._dpi = dpi
        self._fontsize = fontsize
        self._fig, self._ax = None, None
        self._render_count = 0
        
        # Static task locations
        self.WATER_POS = (1.0, 1.0)
        self.LIFE_POS = (0.0, 1.0)

    def init_canvas(self, figure_size, dpi):
        """Initialize the matplotlib figure and axes."""
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.gca()

        # Set up the continuous field [0, 1] x [0, 1]
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        
        # Draw grid lines for visual reference
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        
        return fig, ax

    def render_field(self, state_layout):
        """Render the observation field with features and agents."""
        ax = self._ax
        
        # Render goals
        for goal_idx, (goal_x, goal_y) in enumerate(state_layout['goal_positions']):
            water_done = state_layout['detected_water'][goal_idx]
            life_done = state_layout['detected_life'][goal_idx]
            pic_done = state_layout['picture_taken'][goal_idx]
            
            # Goal color indicates progression: Light Blue -> Dark Blue (Water) -> Green (Life)
            goal_color = '#32CD32' if life_done else ('#4169E1' if water_done else '#ADD8E6')
            
            ax.plot(goal_x, goal_y, marker='s', markersize=25,
                    color=goal_color, markeredgecolor='black', markeredgewidth=2, zorder=2)
            
            label = f"Goal {goal_idx + 1}"
            
            # Camera Flash if Picture Taken
            if pic_done:
                label += " 📸"
                ax.plot(goal_x, goal_y, marker='*', markersize=60,
                        color='yellow', alpha=0.5, zorder=1)
                        
            ax.text(goal_x, goal_y - 0.04, label,
                    ha='center', va='top', fontsize=8, fontweight='bold', color='black')

        # Draw agents
        for agent_idx, (pos_x, pos_y) in enumerate(state_layout['agent_positions']):
            agent_color = self.AGENT_COLORS[agent_idx % len(self.AGENT_COLORS)]
            
            # Draw collision radius (light red circle)
            collision_circle = mpatches.Circle(
                (pos_x, pos_y),
                0.1, # collision radius
                facecolor='red',
                edgecolor='none',
                zorder=1,
                alpha=0.1
            )
            ax.add_patch(collision_circle)
            
            # Draw detection radius (light blue circle)
            detection_circle = mpatches.Circle(
                (pos_x, pos_y),
                0.15, # default detection radius
                facecolor='blue',
                edgecolor='blue',
                linewidth=1,
                linestyle='--',
                zorder=1,
                alpha=0.05
            )
            ax.add_patch(detection_circle)

            # Draw the agent rover
            circle = mpatches.Circle(
                (pos_x, pos_y),
                0.03, # rover size
                facecolor=agent_color,
                edgecolor='black',
                linewidth=2,
                zorder=5
            )
            ax.add_patch(circle)

            # Add agent label
            ax.text(pos_x, pos_y, str(agent_idx + 1),
                   ha='center', va='center',
                   fontsize=self._fontsize,
                   fontweight='bold',
                   color='white',
                   zorder=6)

    def add_legend_and_scoreboard(self, state_layout):
        """Add legend and task scoreboard to the visualization."""
        # Add tasks scoreboard on the right
        score_text = "Task Status:"
        for i in range(2):
            w = "✅" if state_layout['detected_water'][i] else "❌"
            l = "✅" if state_layout['detected_life'][i] else "❌"
            p = "✅" if state_layout['picture_taken'][i] else "❌"
            score_text += f"\n\nGoal {i+1}:\nWater: {w}\nLife:  {l}\nPhoto: {p}"

        self._ax.text(1.05, 0.85, score_text,
                     transform=self._ax.transAxes,
                     fontsize=self._fontsize + 2,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1),
                     family='monospace')

        # Add step counter
        steps_text = f"Step: {self._render_count}"
        self._ax.text(0.02, 0.98, steps_text,
                     transform=self._ax.transAxes,
                     fontsize=self._fontsize,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                     
        # Add title
        title = "Continuous Cooperative Reconnaissance"
        if all(state_layout['picture_taken']):
            title += " - 🎉 MISSION ACCOMPLISHED! 🎉"
        self._ax.set_title(title, fontsize=self._fontsize + 2, fontweight='bold', pad=15)

    def convert2img(self, fig):
        """Convert matplotlib figure to numpy array/PIL Image."""
        fig.tight_layout()
        fig.canvas.draw()

        # Convert to numpy array
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]  # Remove alpha channel

        # Convert to PIL Image
        img = Image.fromarray(data)
        return img

    def render(self, state_layout):
        """
        Main render method called by the environment wrapper.

        Args:
            state_layout: Dictionary containing agent_positions and task flags
        Returns:
            PIL Image of the rendered visualization
        """
        # Initialize canvas
        figure_size = (7, 5) # 7x5 inches
        self._fig, self._ax = self.init_canvas(figure_size, self._dpi)

        # Render field
        self.render_field(state_layout)

        # Add scoreboard
        self.add_legend_and_scoreboard(state_layout)

        # Convert to image
        img = self.convert2img(self._fig)

        # Increment render count
        self._render_count += 1

        # Clean up
        plt.close(self._fig)

        return img

    def reset(self):
        """Reset the visualizer state."""
        self._render_count = 0
