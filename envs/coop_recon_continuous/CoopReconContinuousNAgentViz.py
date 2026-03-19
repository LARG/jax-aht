"""
Visualizer for the Continuous Cooperative Reconnaissance Domain (N-Agent Generic)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

class CoopReconContinuousNAgentViz:
    # Colors for different agents
    AGENT_COLORS = [
        '#4169E1',  # Royal blue
        '#FF6347',  # Tomato red
        '#32CD32',  # Lime green
        '#FFD700',  # Gold
        '#FF69B4',  # Hot Pink
        '#8A2BE2',  # Blue Violet
    ]

    def __init__(self, grid_size=1.0, dpi=100, fontsize=10) -> None:
        self.grid_size = grid_size
        self._dpi = dpi
        self._fontsize = fontsize
        self._fig, self._ax = None, None
        self._render_count = 0

    def init_canvas(self, figure_size, dpi):
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.gca()

        # Set up the continuous field [0, grid_size] x [0, grid_size]
        ax.set_xlim(-0.05, self.grid_size + 0.05)
        ax.set_ylim(-0.05, self.grid_size + 0.05)
        ax.set_aspect('equal')
        
        # Draw grid lines for visual reference
        ticks = np.arange(0, self.grid_size + 0.1, 0.2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        
        return fig, ax

    def render_field(self, state_layout):
        ax = self._ax
        
        # Render goals
        for goal_idx, (goal_x, goal_y) in enumerate(state_layout['goal_positions']):
            water_done = state_layout['detected_water'][goal_idx]
            life_done = state_layout['detected_life'][goal_idx]
            pic_done = state_layout['picture_taken'][goal_idx]
            
            goal_color = '#32CD32' if life_done else ('#4169E1' if water_done else '#ADD8E6')
            
            ax.plot(goal_x, goal_y, marker='s', markersize=25,
                    color=goal_color, markeredgecolor='black', markeredgewidth=2, zorder=2)
            
            label = f"Goal {goal_idx + 1}"
            
            if pic_done:
                label += " 📸"
                ax.plot(goal_x, goal_y, marker='*', markersize=60,
                        color='yellow', alpha=0.5, zorder=1)
                        
            ax.text(goal_x, goal_y - 0.04, label,
                    ha='center', va='top', fontsize=8, fontweight='bold', color='black')

        # Draw agents
        for agent_idx, (pos_x, pos_y) in enumerate(state_layout['agent_positions']):
            agent_color = self.AGENT_COLORS[agent_idx % len(self.AGENT_COLORS)]
            
            collision_circle = mpatches.Circle(
                (pos_x, pos_y), 0.05, facecolor='red', edgecolor='none', zorder=1, alpha=0.1)
            ax.add_patch(collision_circle)
            
            detection_circle = mpatches.Circle(
                (pos_x, pos_y), 0.20, facecolor='blue', edgecolor='blue',
                linewidth=1, linestyle='--', zorder=1, alpha=0.05)
            ax.add_patch(detection_circle)

            circle = mpatches.Circle(
                (pos_x, pos_y), 0.03, facecolor=agent_color, edgecolor='black', linewidth=2, zorder=5)
            ax.add_patch(circle)

            ax.text(pos_x, pos_y, str(agent_idx + 1),
                   ha='center', va='center', fontsize=self._fontsize, fontweight='bold', color='white', zorder=6)

    def add_legend_and_scoreboard(self, state_layout):
        score_text = "Task Status:"
        num_goals = len(state_layout['detected_water'])
        for i in range(num_goals):
            w = "✅" if state_layout['detected_water'][i] else "❌"
            l = "✅" if state_layout['detected_life'][i] else "❌"
            p = "✅" if state_layout['picture_taken'][i] else "❌"
            score_text += f"\n\nGoal {i+1}:\nWater: {w}\nLife:  {l}\nPhoto: {p}"

        self._ax.text(1.05, 0.85, score_text, transform=self._ax.transAxes,
                     fontsize=self._fontsize + 2, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1),
                     family='monospace')

        steps_text = f"Step: {self._render_count}"
        self._ax.text(0.02, 0.98, steps_text, transform=self._ax.transAxes,
                     fontsize=self._fontsize, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                     
        title = f"Continuous Cooperative Reconnaissance ({num_goals} Agents)"
        if all(state_layout['picture_taken']):
            title += " - 🎉 MISSION ACCOMPLISHED! 🎉"
        self._ax.set_title(title, fontsize=self._fontsize + 2, fontweight='bold', pad=15)

    def convert2img(self, fig):
        fig.tight_layout()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        img = Image.fromarray(data)
        return img

    def render(self, state_layout):
        # dynamic width based on N to fit scoreboard
        num_goals = len(state_layout['detected_water'])
        figure_size = (7 + num_goals * 1.0, 6)
        
        self._fig, self._ax = self.init_canvas(figure_size, self._dpi)
        self.render_field(state_layout)
        self.add_legend_and_scoreboard(state_layout)
        img = self.convert2img(self._fig)
        self._render_count += 1
        plt.close(self._fig)
        return img

    def reset(self):
        self._render_count = 0
