"""
Visualizer for the Multi-Agent 4x4 Grid Navigation Domain

This visualizer displays:
- Multiple agents with different colors
- Individual goal locations for each agent
- Controllable vs non-controllable agents (different markers)
- Obstacles
- Grid cells with coordinates
- Paths taken by each agent
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class Grid4x4MultiAgentVisualizer(BaseViz):

    # Colors for different agents
    AGENT_COLORS = [
        '#4169E1',  # Royal blue
        '#FF6347',  # Tomato red
        '#32CD32',  # Lime green
        '#FFD700',  # Gold
        '#9370DB',  # Medium purple
        '#FF69B4',  # Hot pink
    ]

    GOAL_COLORS = [
        '#1E90FF',  # Dodger blue
        '#FF4500',  # Orange red
        '#228B22',  # Forest green
        '#DAA520',  # Goldenrod
        '#8B008B',  # Dark magenta
        '#FF1493',  # Deep pink
    ]

    def __init__(self, model: RDDLPlanningModel,
                 dpi=100,
                 fontsize=10,
                 cell_size=100) -> None:
        """
        Initialize the Multi-Agent Grid4x4 Visualizer.

        Args:
            model: The RDDL planning model
            dpi: Dots per inch for the figure
            fontsize: Font size for text labels
            cell_size: Size of each grid cell in pixels
        """
        self._model = model
        self._objects = model.type_to_objects
        self._dpi = dpi
        self._fontsize = fontsize
        self._cell_size = cell_size

        # Get grid dimensions
        self._x_positions = sorted(self._objects['xpos'])
        self._y_positions = list(reversed(sorted(self._objects['ypos'])))
        self._grid_width = len(self._x_positions)
        self._grid_height = len(self._y_positions)

        # Get agents
        self._agents = sorted(self._objects['agent'])

        self._fig, self._ax = None, None
        self._path_history = {agent: [] for agent in self._agents}
        self._render_count = 0

    def build_nonfluents_layout(self):
        """Extract non-fluent information from the model."""
        obstacles = []
        controllable = {}  # {agent: bool}

        non_fluents = self._model.ground_vars_with_values(self._model.non_fluents)

        for k, v in non_fluents.items():
            var, objects = self._model.parse_grounded(k)

            if var == 'OBSTACLE' and v:
                obstacles.append(tuple(objects))
            elif var == 'CONTROLLABLE':
                agent = objects[0]
                controllable[agent] = v

        return {
            'obstacles': obstacles,
            'controllable': controllable
        }

    def build_states_layout(self, state):
        """Extract state information."""
        agent_positions = {}  # {agent: (x, y)}
        goal_positions = {}  # {agent: (x, y)}
        goal_reached = {}  # {agent: bool}
        collisions = {}  # {agent: bool}

        for k, v in state.items():
            var, objects = self._model.parse_grounded(k)

            if var == 'agent-at' and v:
                agent, x, y = objects
                agent_positions[agent] = (x, y)
            elif var == 'goal-at' and v:
                agent, x, y = objects
                goal_positions[agent] = (x, y)
            elif var == 'goal-reached':
                agent = objects[0]
                goal_reached[agent] = v
            elif var == 'collision':
                agent = objects[0]
                collisions[agent] = v

        return {
            'agent_positions': agent_positions,
            'goals': goal_positions,
            'goal_reached': goal_reached,
            'collisions': collisions
        }

    def _pos_to_grid_coords(self, x_name, y_name):
        """Convert position names (e.g., 'x0', 'y1') to grid coordinates."""
        x_idx = self._x_positions.index(x_name)
        y_idx = self._y_positions.index(y_name)
        # Flip y-axis so y0 is at bottom
        y_idx = self._grid_height - 1 - y_idx
        return x_idx, y_idx

    def init_canvas(self, figure_size, dpi):
        """Initialize the matplotlib figure and axes."""
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.gca()

        # Set up the grid
        ax.set_xlim(-0.5, self._grid_width - 0.5)
        ax.set_ylim(-0.5, self._grid_height - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(self._grid_width))
        ax.set_yticks(range(self._grid_height))
        ax.set_xticklabels(self._x_positions)
        ax.set_yticklabels(self._y_positions[::-1])  # Flip for display
        ax.grid(True, linewidth=2, color='black', alpha=0.3)

        return fig, ax

    def render_cell(self, x_idx, y_idx, x_name, y_name,
                   nonfluent_layout, state_layout):
        """Render a single grid cell."""
        ax = self._ax

        # Check if this is an obstacle
        is_obstacle = (x_name, y_name) in nonfluent_layout['obstacles']

        # Draw cell background
        cell_color = 'white'
        if is_obstacle:
            cell_color = '#FF6B6B'  # Light red for obstacles

        rect = mpatches.Rectangle(
            (x_idx - 0.45, y_idx - 0.45),
            0.9, 0.9,
            facecolor=cell_color,
            edgecolor='none',
            zorder=0
        )
        ax.add_patch(rect)

        # Draw obstacle marker
        if is_obstacle:
            ax.plot(x_idx, y_idx, marker='x', markersize=25,
                   color='darkred', markeredgewidth=4, zorder=3)
            ax.text(x_idx, y_idx - 0.35, 'WALL',
                   ha='center', va='top', fontsize=7,
                   fontweight='bold', color='darkred', zorder=4)

        # Draw goals for each agent
        for agent_idx, (agent, (goal_x, goal_y)) in enumerate(state_layout['goals'].items()):
            if (goal_x, goal_y) == (x_name, y_name):
                goal_color = self.GOAL_COLORS[agent_idx % len(self.GOAL_COLORS)]
                # Draw goal as a star
                ax.plot(x_idx, y_idx, marker='*', markersize=20,
                       color=goal_color, markeredgecolor='orange',
                       markeredgewidth=1.5, zorder=2, alpha=0.7)
                # Label
                ax.text(x_idx, y_idx + 0.35, f'G{agent_idx+1}',
                       ha='center', va='bottom', fontsize=7,
                       fontweight='bold', color=goal_color, zorder=4)

        # Draw agents
        for agent_idx, (agent, (pos_x, pos_y)) in enumerate(state_layout['agent_positions'].items()):
            if (pos_x, pos_y) == (x_name, y_name):
                # Add to path history
                if len(self._path_history[agent]) == 0 or self._path_history[agent][-1] != (x_idx, y_idx):
                    self._path_history[agent].append((x_idx, y_idx))

                agent_color = self.AGENT_COLORS[agent_idx % len(self.AGENT_COLORS)]
                is_controllable = nonfluent_layout['controllable'].get(agent, False)
                is_colliding = state_layout['collisions'].get(agent, False)

                # Draw collision indicator (red glow) if agent is colliding
                if is_colliding:
                    collision_circle = mpatches.Circle(
                        (x_idx, y_idx),
                        0.35,
                        facecolor='none',
                        edgecolor='red',
                        linewidth=4,
                        zorder=4,
                        alpha=0.8
                    )
                    ax.add_patch(collision_circle)

                # Draw the agent - different shapes for controllable vs non-controllable
                if is_controllable:
                    # Controllable: filled circle
                    circle = mpatches.Circle(
                        (x_idx, y_idx),
                        0.25,
                        facecolor=agent_color,
                        edgecolor='red' if is_colliding else 'black',
                        linewidth=3 if is_colliding else 2,
                        zorder=5
                    )
                    ax.add_patch(circle)
                else:
                    # Non-controllable: empty circle
                    circle = mpatches.Circle(
                        (x_idx, y_idx),
                        0.25,
                        facecolor='white',
                        edgecolor='red' if is_colliding else agent_color,
                        linewidth=4 if is_colliding else 3,
                        zorder=5
                    )
                    ax.add_patch(circle)

                # Add agent label
                ax.text(x_idx, y_idx, str(agent_idx + 1),
                       ha='center', va='center',
                       fontsize=self._fontsize,
                       fontweight='bold',
                       color='black' if not is_controllable else 'white',
                       zorder=6)

    def render_paths(self):
        """Draw the paths taken by each agent."""
        for agent_idx, (agent, path) in enumerate(self._path_history.items()):
            if len(path) > 1:
                path_color = self.AGENT_COLORS[agent_idx % len(self.AGENT_COLORS)]
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                self._ax.plot(path_x, path_y,
                             linestyle='--', linewidth=2,
                             color=path_color, alpha=0.4, zorder=1,
                             marker='o', markersize=3,
                             markerfacecolor=path_color, markeredgecolor='none')

    def add_legend_and_title(self, state_layout, nonfluent_layout, subs=None):
        """Add legend and title to the visualization."""
        # Create legend elements
        legend_elements = []

        # Add agents to legend
        for agent_idx, agent in enumerate(self._agents):
            agent_color = self.AGENT_COLORS[agent_idx % len(self.AGENT_COLORS)]
            is_controllable = nonfluent_layout['controllable'].get(agent, False)
            label = f'Agent {agent_idx + 1}'
            if is_controllable:
                label += ' (controllable)'
                patch = mpatches.Patch(facecolor=agent_color, edgecolor='black', label=label)
            else:
                label += ' (auto)'
                patch = mpatches.Patch(facecolor='white', edgecolor=agent_color,
                                      linewidth=2, label=label)
            legend_elements.append(patch)

        # Add obstacle to legend
        legend_elements.append(
            mpatches.Patch(facecolor='#FF6B6B', edgecolor='none', label='Obstacle')
        )

        self._ax.legend(handles=legend_elements,
                       loc='upper left',
                       bbox_to_anchor=(1.05, 1),
                       fontsize=self._fontsize - 1)

        # Add actions display below legend
        if subs is not None:
            action_mapping = {0: 'NOOP', 1: 'WEST', 2: 'EAST', 3: 'NORTH', 4: 'SOUTH'}
            actions_text = "Actions:\n"

            # Parse actions from the state format
            # Actions come in format like {'move': [0, 2]} where values are action indices
            if 'move' in subs:
                move_actions = subs['move']
                for agent_idx, action_val in enumerate(move_actions):
                    if agent_idx < len(self._agents):
                        action_name = action_mapping.get(int(action_val), f'Unknown({action_val})')
                        actions_text += f"  Agent {agent_idx + 1}: {action_name}\n"

            # Place the actions text below the legend
            self._ax.text(1.05, 0.65, actions_text,
                         transform=self._ax.transAxes,
                         fontsize=self._fontsize,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1),
                         family='monospace')

        # Add actions display below legend
        if subs is not None:
            action_mapping = {0: 'NOOP', 1: 'WEST', 2: 'EAST', 3: 'NORTH', 4: 'SOUTH'}
            actions_text = "Effective Actions:\n"

            # Parse actions from the state format
            # Actions come in format like {'move': [0, 2]} where values are action indices
            if 'effective-move' in subs:
                move_actions = subs['effective-move']
                for agent_idx, action_val in enumerate(move_actions):
                    if agent_idx < len(self._agents):
                        action_name = action_mapping.get(int(action_val), f'Unknown({action_val})')
                        actions_text += f"  Agent {agent_idx + 1}: {action_name}\n"

            # Place the actions text below the legend
            self._ax.text(1.05, 0.45, actions_text,
                         transform=self._ax.transAxes,
                         fontsize=self._fontsize,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1),
                         family='monospace')

        # Add title
        num_reached = sum(1 for reached in state_layout['goal_reached'].values() if reached)
        num_collisions = sum(1 for collision in state_layout['collisions'].values() if collision)
        num_controllable = sum(1 for c in nonfluent_layout['controllable'].values() if c)

        title = f"Multi-Agent Grid Navigation ({num_controllable} controllable)"
        if num_reached > 0:
            title += f" - {num_reached} goal(s) reached!"
        # if num_collisions > 0:
        #     title += f" ⚠️ {num_collisions} collision(s)!"

        self._ax.set_title(title, fontsize=self._fontsize + 4,
                          fontweight='bold', pad=20)

        # Add step counter
        steps_text = f"Step: {self._render_count}"
        self._ax.text(0.02, 0.98, steps_text,
                     transform=self._ax.transAxes,
                     fontsize=self._fontsize,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def convert2img(self, fig, ax):
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

    def render(self, state, subs):
        """
        Main render method called by the environment.

        Args:
            state: Dictionary of current state values
            subs: Dictionary of current subs values (e.g., effective actions)
            PIL Image of the rendered visualization
        """
        # Extract layouts
        nonfluent_layout = self.build_nonfluents_layout()

        state_layout = self.build_states_layout(state)

        # Initialize canvas
        figure_size = (self._grid_width + 3, self._grid_height + 1)
        self._fig, self._ax = self.init_canvas(figure_size, self._dpi)

        # Render each cell
        for x_name in self._x_positions:
            for y_name in self._y_positions:
                x_idx, y_idx = self._pos_to_grid_coords(x_name, y_name)
                self.render_cell(x_idx, y_idx, x_name, y_name,
                               nonfluent_layout, state_layout)

        # Render the paths taken
        self.render_paths()

        # Add legend and title
        self.add_legend_and_title(state_layout, nonfluent_layout, subs)

        # Convert to image
        img = self.convert2img(self._fig, self._ax)

        # Increment render count
        self._render_count += 1

        # Clean up
        plt.close(self._fig)

        return img

    def reset(self):
        """Reset the visualizer state."""
        self._path_history = {agent: [] for agent in self._agents}
        self._render_count = 0
