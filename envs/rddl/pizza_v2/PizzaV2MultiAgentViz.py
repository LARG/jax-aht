"""
Visualizer for the Multi-Agent Pizza V2 RDDL domain.

Scene style:
- Node-link map of locations based on CONNECTED non-fluents
- Shop/customer semantic node styling
- Truck markers offset around each node when multiple trucks share a location
- Right-side telemetry panel for actions and delivery state
"""

from typing import Dict, Any, List, Tuple
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class PizzaV2MultiAgentVisualizer(BaseViz):
    TRUCK_COLORS = [
        "#2E86AB",
        "#E76F51",
        "#2A9D8F",
        "#7B2CBF",
        "#F4A261",
        "#3A86FF",
    ]

    def __init__(self, model: RDDLPlanningModel, dpi: int = 100, fontsize: int = 10) -> None:
        self._model = model
        self._objects = model.type_to_objects
        self._dpi = dpi
        self._fontsize = fontsize

        self._locations = list(self._objects["location"])
        self._trucks = list(self._objects["truck"])

        self._fig = None
        self._ax = None
        self._render_count = 0

        self._nonfluents_layout = self.build_nonfluents_layout()
        self._node_positions = self._build_node_positions()
        self._path_history = {truck: [] for truck in self._trucks}

    def build_nonfluents_layout(self) -> Dict[str, Any]:
        shops = set()
        connected = []
        connection_num = {}
        controllable = {}
        can_deliver = {truck: set() for truck in self._trucks}
        capacities = {}
        max_connections = 0

        non_fluents = self._model.ground_vars_with_values(self._model.non_fluents)

        for k, v in non_fluents.items():
            var, objects = self._model.parse_grounded(k)

            if var == "SHOP" and bool(v):
                shops.add(objects[0])
            elif var == "CONNECTED" and bool(v):
                connected.append((objects[0], objects[1]))
            elif var == "CONNECTION-NUM":
                src, dst = objects
                idx = int(v)
                if idx > 0:
                    connected.append((src, dst))
                    connection_num[(src, idx)] = dst
            elif var == "CONTROLLABLE":
                controllable[objects[0]] = bool(v)
            elif var == "CAN-DELIVER" and bool(v):
                truck, loc = objects
                can_deliver[truck].add(loc)
            elif var == "CAPACITY":
                capacities[objects[0]] = int(v)
            elif var == "MAX-CONNECTIONS":
                max_connections = int(v)

        return {
            "shops": shops,
            "connected": connected,
            "connection_num": connection_num,
            "controllable": controllable,
            "can_deliver": can_deliver,
            "capacities": capacities,
            "max_connections": max_connections,
        }

    def build_states_layout(self, state: Dict[str, Any]) -> Dict[str, Any]:
        truck_at = {}
        num_shop_pizzas = 0
        num_orders_remaining = {}
        num_pizzas_in_truck = {}
        collisions = {}
        done_delivering = {}

        for k, v in state.items():
            var, objects = self._model.parse_grounded(k)

            if var == "truckAt" and bool(v):
                truck, location = objects
                truck_at[truck] = location
            elif var == "numShopPizzas":
                num_shop_pizzas = int(v)
            elif var == "numOrdersRemaining":
                num_orders_remaining[objects[0]] = int(v)
            elif var == "numPizzasInTruck":
                num_pizzas_in_truck[objects[0]] = int(v)
            elif var == "collision":
                collisions[objects[0]] = bool(v)
            elif var == "doneDelivering":
                done_delivering[objects[0]] = bool(v)

        return {
            "truck_at": truck_at,
            "num_shop_pizzas": num_shop_pizzas,
            "num_orders_remaining": num_orders_remaining,
            "num_pizzas_in_truck": num_pizzas_in_truck,
            "collisions": collisions,
            "done_delivering": done_delivering,
        }

    def _build_node_positions(self) -> Dict[str, Tuple[float, float]]:
        shops = [loc for loc in self._locations if loc in self._nonfluents_layout["shops"]]
        others = [loc for loc in self._locations if loc not in self._nonfluents_layout["shops"]]

        positions = {}

        if shops:
            for idx, shop in enumerate(shops):
                if len(shops) == 1:
                    positions[shop] = (0.0, 0.0)
                else:
                    angle = 2 * math.pi * idx / len(shops)
                    positions[shop] = (0.5 * math.cos(angle), 0.5 * math.sin(angle))
        elif self._locations:
            positions[self._locations[0]] = (0.0, 0.0)
            others = self._locations[1:]

        if others:
            radius = 1.6
            for idx, loc in enumerate(others):
                angle = 2 * math.pi * idx / max(1, len(others))
                positions[loc] = (radius * math.cos(angle), radius * math.sin(angle))

        return positions

    def _extract_actions_layout(self, subs: Dict[str, Any], state_layout: Dict[str, Any]) -> Dict[str, Any]:
        actions = {truck: "none" for truck in self._trucks}
        action_meta = {truck: "" for truck in self._trucks}

        if not isinstance(subs, dict):
            return {"actions": actions, "meta": action_meta}

        action_num_vals = subs.get("action-num")
        picked_vals = subs.get("pizzaPickedUp")
        reward_vals = subs.get("deliveryRewarded")

        def _as_array(value):
            if value is None:
                return None
            return np.asarray(value)

        def _normalize_bool_action_tensor(value):
            arr = _as_array(value)
            if arr is None:
                return None
            # Only collapse a trailing [false, true] axis when we truly have
            # an extra boolean-pair dimension (e.g., (num_agents, num_locs, 2)).
            # Do NOT collapse plain per-truck vectors like (num_agents,).
            if arr.ndim >= 2 and arr.shape[-1] == 2:
                return arr[..., 1]
            return arr

        action_num_vals = _as_array(action_num_vals)
        picked_vals = _normalize_bool_action_tensor(picked_vals)
        reward_vals = _normalize_bool_action_tensor(reward_vals)

        for t_idx, truck in enumerate(self._trucks):
            if action_num_vals is None or action_num_vals.ndim < 1:
                continue

            action_num = int(action_num_vals[t_idx])
            if action_num == 0:
                actions[truck] = "noop"
            elif action_num == 1:
                actions[truck] = "load"
            elif action_num == 2:
                actions[truck] = "deliver"
            elif action_num >= 3:
                conn_idx = action_num - 2
                # Resolve drive destination based on current truck location.
                src = state_layout.get("truck_at", {}).get(truck)
                dest = None
                if src is not None:
                    dest = self._nonfluents_layout["connection_num"].get((src, conn_idx))
                if dest is not None:
                    actions[truck] = f"drive->{dest}"
                else:
                    actions[truck] = f"drive#{conn_idx}"

        if picked_vals is not None and picked_vals.ndim >= 1:
            for t_idx, truck in enumerate(self._trucks):
                if bool(picked_vals[t_idx]):
                    action_meta[truck] = "picked-up"

        if reward_vals is not None and reward_vals.ndim >= 2:
            for t_idx, truck in enumerate(self._trucks):
                loc_idxs = np.where(reward_vals[t_idx])[0]
                if loc_idxs.size > 0:
                    loc_name = self._locations[int(loc_idxs[0])]
                    suffix = f"delivered:{loc_name}"
                    action_meta[truck] = f"{action_meta[truck]}, {suffix}" if action_meta[truck] else suffix

        return {"actions": actions, "meta": action_meta}

    def init_canvas(self, figure_size: Tuple[float, float], dpi: int):
        fig = plt.figure(figsize=figure_size, dpi=dpi)
        ax = plt.gca()

        xs = [p[0] for p in self._node_positions.values()] or [0.0]
        ys = [p[1] for p in self._node_positions.values()] or [0.0]
        pad = 0.9
        ax.set_xlim(min(xs) - pad, max(xs) + pad + 2.15)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect("equal")
        ax.axis("off")

        return fig, ax

    def render_edges(self):
        edge_set = set(self._nonfluents_layout["connected"])
        node_radius = 0.22

        for src, dst in self._nonfluents_layout["connected"]:
            if src not in self._node_positions or dst not in self._node_positions:
                continue

            x0, y0 = self._node_positions[src]
            x1, y1 = self._node_positions[dst]

            dx = x1 - x0
            dy = y1 - y0
            dist = math.hypot(dx, dy)
            if dist < 1e-8:
                continue

            ux = dx / dist
            uy = dy / dist

            # Start/end on node boundaries so arrowheads are visible and directional.
            sx = x0 + ux * (node_radius + 0.03)
            sy = y0 + uy * (node_radius + 0.03)
            ex = x1 - ux * (node_radius + 0.04)
            ey = y1 - uy * (node_radius + 0.04)

            # If both directions exist, curve opposite directions so arrows are distinct.
            reverse_exists = (dst, src) in edge_set
            if reverse_exists:
                rad = 0.18 if src < dst else -0.18
            else:
                rad = 0.0

            arrow = mpatches.FancyArrowPatch(
                (sx, sy),
                (ex, ey),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2.0,
                color="#5C5C5C",
                alpha=0.9,
                zorder=1,
                connectionstyle=f"arc3,rad={rad}",
            )
            self._ax.add_patch(arrow)

    def render_nodes(self, state_layout: Dict[str, Any]):
        for loc in self._locations:
            x, y = self._node_positions[loc]
            is_shop = loc in self._nonfluents_layout["shops"]
            node_color = "#F4A261" if is_shop else "#8ECAE6"
            edge_color = "#7F5539" if is_shop else "#1D3557"

            circle = mpatches.Circle(
                (x, y),
                0.22,
                facecolor=node_color,
                edgecolor=edge_color,
                linewidth=2.0,
                zorder=2,
            )
            self._ax.add_patch(circle)

            if is_shop:
                sub = f"shop={state_layout['num_shop_pizzas']}"
            else:
                rem = state_layout["num_orders_remaining"].get(loc, 0)
                sub = f"orders={rem}"

            self._ax.text(
                x,
                y + 0.34,
                loc,
                ha="center",
                va="bottom",
                fontsize=self._fontsize,
                fontweight="bold",
                color="#111111",
                zorder=4,
            )
            self._ax.text(
                x,
                y - 0.34,
                sub,
                ha="center",
                va="top",
                fontsize=self._fontsize - 1,
                color="#222222",
                zorder=4,
            )

    def render_trucks(self, state_layout: Dict[str, Any]):
        trucks_at_location = {loc: [] for loc in self._locations}
        for truck, loc in state_layout["truck_at"].items():
            if loc in trucks_at_location:
                trucks_at_location[loc].append(truck)

        for loc, trucks_here in trucks_at_location.items():
            if not trucks_here:
                continue
            node_x, node_y = self._node_positions[loc]
            n = len(trucks_here)
            for idx, truck in enumerate(trucks_here):
                angle = 2 * math.pi * idx / max(1, n)
                tx = node_x + 0.34 * math.cos(angle)
                ty = node_y + 0.34 * math.sin(angle)

                truck_idx = self._trucks.index(truck)
                color = self.TRUCK_COLORS[truck_idx % len(self.TRUCK_COLORS)]
                controllable = self._nonfluents_layout["controllable"].get(truck, False)
                colliding = state_layout["collisions"].get(truck, False)

                if colliding:
                    glow = mpatches.Circle(
                        (tx, ty),
                        0.16,
                        facecolor="none",
                        edgecolor="#D00000",
                        linewidth=3,
                        alpha=0.9,
                        zorder=5,
                    )
                    self._ax.add_patch(glow)

                marker = mpatches.Circle(
                    (tx, ty),
                    0.12,
                    facecolor=color if controllable else "white",
                    edgecolor="#111111" if controllable else color,
                    linewidth=2.4,
                    zorder=6,
                )
                self._ax.add_patch(marker)

                self._ax.text(
                    tx,
                    ty,
                    truck.replace("t", ""),
                    ha="center",
                    va="center",
                    fontsize=self._fontsize - 1,
                    fontweight="bold",
                    color="white" if controllable else color,
                    zorder=7,
                )

                # Track movement history for trails.
                prev = self._path_history[truck]
                if len(prev) == 0 or prev[-1] != (tx, ty):
                    prev.append((tx, ty))

    def render_paths(self):
        for idx, truck in enumerate(self._trucks):
            pts = self._path_history[truck]
            if len(pts) < 2:
                continue
            color = self.TRUCK_COLORS[idx % len(self.TRUCK_COLORS)]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self._ax.plot(
                xs,
                ys,
                linestyle="--",
                linewidth=1.5,
                color=color,
                alpha=0.35,
                zorder=3,
            )

    def add_legend_and_panels(self, state_layout: Dict[str, Any], action_layout: Dict[str, Any]):
        legend_elements = [
            mpatches.Patch(facecolor="#F4A261", edgecolor="#7F5539", label="Shop"),
            mpatches.Patch(facecolor="#8ECAE6", edgecolor="#1D3557", label="Customer"),
            mpatches.Patch(facecolor="white", edgecolor="#D00000", label="Collision"),
        ]

        for idx, truck in enumerate(self._trucks):
            color = self.TRUCK_COLORS[idx % len(self.TRUCK_COLORS)]
            ctrl = self._nonfluents_layout["controllable"].get(truck, False)
            label = f"{truck} ({'ctrl' if ctrl else 'auto'})"
            legend_elements.append(
                mpatches.Patch(facecolor=color if ctrl else "white", edgecolor=color, label=label)
            )

        self._ax.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=self._fontsize - 1,
            frameon=True,
        )

        totals = [state_layout["num_orders_remaining"].get(loc, 0) for loc in self._locations if loc not in self._nonfluents_layout["shops"]]
        total_remaining = int(np.sum(np.asarray(totals, dtype=np.int32))) if totals else 0

        status_lines = [
            f"Step: {self._render_count}",
            f"Shop Pizzas: {state_layout['num_shop_pizzas']}",
            f"Orders Left: {total_remaining}",
            "",
            "Truck Status:",
        ]

        for truck in self._trucks:
            loc = state_layout["truck_at"].get(truck, "?")
            inv = state_layout["num_pizzas_in_truck"].get(truck, 0)
            done = state_layout["done_delivering"].get(truck, False)
            col = state_layout["collisions"].get(truck, False)
            action = action_layout["actions"].get(truck, "none")
            meta = action_layout["meta"].get(truck, "")
            row = f"- {truck}@{loc} inv={inv} act={action}"
            status_lines.append(row)
            status_lines.append(f"  done={done} collision={col}")
            if meta:
                status_lines.append(f"  event={meta}")

        self._ax.text(
            0.67,
            0.52,
            "\n".join(status_lines),
            transform=self._ax.transAxes,
            fontsize=self._fontsize,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#444444", alpha=0.95),
            family="monospace",
        )

        self._ax.set_title("Pizza V2 Delivery - Node Map", fontsize=self._fontsize + 5, fontweight="bold", pad=12)

    def convert2img(self, fig):
        fig.tight_layout()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state: Dict[str, Any], subs: Dict[str, Any]):
        state_layout = self.build_states_layout(state)
        action_layout = self._extract_actions_layout(subs, state_layout)

        self._fig, self._ax = self.init_canvas((11.0, 6.0), self._dpi)

        self.render_edges()
        self.render_nodes(state_layout)
        self.render_trucks(state_layout)
        self.render_paths()
        self.add_legend_and_panels(state_layout, action_layout)

        image = self.convert2img(self._fig)
        self._render_count += 1

        plt.close(self._fig)
        return image

    def reset(self):
        self._path_history = {truck: [] for truck in self._trucks}
        self._render_count = 0
