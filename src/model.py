"""
Mountain Rescue Simulation Model with Multi-Mode Support

This module implements the central MountainRescueModel class that supports three operational modes:
- Basic Mode: Traditional agent-based simulation
- Extended Mode: Enhanced communication and coordination
- Novel Mode: GNN-MARL implementation with PyTorch and PyTorch Geometric

The model provides dynamic agent coordination, terrain simulation, and performance metrics.
"""

import numpy as np
from mesa import Model
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.datacollection import DataCollector
from typing import Dict, Any, Tuple, List, Optional

from agents import FirstAidRobot, ExplorerDrone, Person, MobileCharger, RobotState, DroneState, MobileChargerState
from messaging import MessageSystem
from gnn_marl import GNN_MARL_System, GNNConfig, MARLConfig


class MountainRescueModel(Model):
    """
    Enhanced Mountain Rescue Model with Multi-Mode Support.

    Supports three operational modes:
    - Basic: Traditional agent-based simulation
    - Extended: Enhanced with communication and dynamic spawning
    - Novel: GNN-MARL enhanced decision making (extends Extended mode)
    """

    def __init__(self, width: int = 20, height: int = 20, n_robots: int = 5,
                 n_drones: int = 2, n_persons: int = 10, n_chargers: int = 1, 
                 mode: str = "basic", seed: Optional[int] = None, spawn_interval: int = 50,
                 max_persons: int = 30, use_gnn_marl: bool = False,
                 use_commander: bool = False, device=None, quiet: bool = False):
        """
        Initialize the Mountain Rescue Model.

        Args:
            width: Grid width dimension
            height: Grid height dimension
            n_robots: Number of FirstAidRobots
            n_drones: Number of ExplorerDrones
            n_persons: Initial number of persons to rescue
            n_chargers: Number of MobileCharger agents (Novel mode only)
            mode: Operation mode ("basic", "extended", or "novel")
            seed: Random seed for reproducibility
            spawn_interval: Steps between spawning new persons (Extended/Novel Mode)
            max_persons: Maximum persons in environment at once
            use_gnn_marl: Whether to use GNN-MARL for agent decision making
            use_commander: Whether to include CommanderAgent for strategic oversight
            device: PyTorch device for GPU acceleration (cuda/cpu)
        """
        super().__init__(seed=seed)

        # Core model parameters
        self.width = width
        self.height = height
        self.mode = mode
        self.spawn_interval = spawn_interval
        self.max_persons = max_persons
        self.spawn_timer = 0
        self.use_gnn_marl = use_gnn_marl
        self.use_commander = use_commander
        self.device = device
        self.quiet = quiet

        # Initialize random number generator
        if seed is not None:
            self.random.seed(seed)
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Model state tracking
        self.rescued_count = 0
        self.total_persons_spawned = n_persons
        self.persons_spawned_dynamically = 0
        self.rescue_log = []

        # Base station location (bottom-left corner)
        self.base_position = (0, 0)

        # Initialize core components
        self._initialize_grid()
        self._create_mountain_terrain()
        self._initialize_communication_system()
        self._initialize_gnn_marl_system()
        self._create_agents(n_robots, n_drones, n_persons, n_chargers)
        self._setup_data_collection()

        # Collect initial data
        self.datacollector.collect(self)
        self.running = True

        self._print_initialization_summary(n_robots, n_drones, n_persons, n_chargers)

    # =============================================================================
    # INITIALIZATION METHODS
    # =============================================================================

    def _initialize_grid(self):
        """Initialize the grid space for agent movement."""
        self.grid = OrthogonalMooreGrid(
            dimensions=(self.width, self.height),
            torus=False,  # No wrapping - finite mountain area
            capacity=None  # Multiple agents can occupy same cell
        )

    def _create_mountain_terrain(self):
        """Create a mountain environment with different altitude levels (1K, 2K, 3K MASL)."""
        self.terrain = np.zeros((self.width, self.height))
        center_x, center_y = self.width // 2, self.height // 2

        for x in range(self.width):
            for y in range(self.height):
                # Calculate distance from center
                dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

                # Create mountain profile - higher in center
                elevation_factor = 1 - (dist_from_center / max_dist)

                # Assign altitude levels (0=0MASL, 1=1K MASL, 2=2K MASL, 3=3K MASL)
                if elevation_factor > 0.7:
                    self.terrain[x, y] = 3  # 3K MASL
                elif elevation_factor > 0.4:
                    self.terrain[x, y] = 2  # 2K MASL
                elif elevation_factor > 0.2:
                    self.terrain[x, y] = 1  # 1K MASL
                else:
                    self.terrain[x, y] = 0  # 0 MASL (base level)

    def _initialize_communication_system(self):
        """Initialize message system for Extended and Novel Modes."""
        self.message_system = MessageSystem() if self.mode in ["extended", "novel"] else None

    def _initialize_gnn_marl_system(self):
        """Initialize GNN-MARL system for Novel Mode."""
        self.gnn_marl_system = None
        if self.use_gnn_marl or self.mode == "novel":
            gnn_config = GNNConfig()
            marl_config = MARLConfig()
            self.gnn_marl_system = GNN_MARL_System(gnn_config, marl_config, device=self.device)
            if self.mode == "novel":
                self.gnn_marl_system.enable_training()

    def _create_agents(self, n_robots: int, n_drones: int, n_persons: int, n_chargers: int):
        """Create and place all agents in the simulation."""
        # Get base cell for robots and drones
        base_cell = self._get_base_cell()

        # Create terrain robots at base
        FirstAidRobot.create_agents(
            self,
            n_robots,
            cell=[base_cell] * n_robots,
            battery_capacity=[100] * n_robots
        )

        # Create explorer drones at base
        ExplorerDrone.create_agents(
            self,
            n_drones,
            cell=[base_cell] * n_drones,
            battery_capacity=[150] * n_drones
        )

        # Create mobile chargers at base (Novel mode only)
        if self.mode == "novel" and n_chargers > 0:
            MobileCharger.create_agents(
                self,
                n_chargers,
                cell=[base_cell] * n_chargers,
                battery_capacity=[1000] * n_chargers
            )

        # Create initial persons
        self._spawn_persons(n_persons)

    def _get_base_cell(self):
        """Get the base cell for agent initialization."""
        for cell in self.grid.all_cells:
            if cell.coordinate == self.base_position:
                return cell
        # Fallback - use first available cell
        return list(self.grid.all_cells.cells)[0]

    def _setup_data_collection(self):
        """Setup data collection for performance monitoring."""
        self.datacollector = DataCollector(
            model_reporters={
                # Basic metrics
                "Rescued": lambda m: m.rescued_count,
                "Total Persons": lambda m: len([a for a in m.agents if isinstance(a, Person)]),
                "Active Robots": lambda m: len([a for a in m.agents if isinstance(a, FirstAidRobot) and a.battery > 0]),
                "Active Drones": lambda m: len([a for a in m.agents if isinstance(a, ExplorerDrone) and a.battery > 0]),
                "Rescue Rate": lambda m: m.rescued_count / len([a for a in m.agents if isinstance(a, Person)]) if len(
                    [a for a in m.agents if isinstance(a, Person)]) > 0 else 0,

                # Extended/Novel mode metrics
                "Messages_Sent": lambda m: m.message_system.total_messages_sent if m.message_system else 0,
                "Robots_Waiting": lambda m: len([r for r in m.agents if isinstance(r, FirstAidRobot) and hasattr(r,
                                                                                                                 'state') and r.state.value == "waiting_for_mission"]) if m.mode in [
                    "extended", "novel"] else 0,
                "Robots_On_Mission": lambda m: len([r for r in m.agents if isinstance(r, FirstAidRobot) and hasattr(r,
                                                                                                                    'state') and r.state.value == "moving_to_target"]) if m.mode in [
                    "extended", "novel"] else 0,
                "Average_Response_Time": lambda m: m.calculate_average_response_time(),
                "Communication_Efficiency": lambda m: m.calculate_communication_efficiency(),

                # Enhanced rescue metrics
                "Successful_Rescues": lambda m: len(m.rescue_log),
                "Average_Rescue_Time": lambda m: sum(rescue["rescue_time"] for rescue in m.rescue_log) / max(1,
                                                                                                             len(m.rescue_log)),
                "High_Urgency_Rescues": lambda m: len([r for r in m.rescue_log if r["urgency"] > 0.7]),

                # Novel Mode (GNN-MARL) metrics
                "Novel_Robot_Epsilon": lambda m: m.gnn_marl_system.robot_policy.epsilon if m.gnn_marl_system else 0,
                "Novel_Drone_Epsilon": lambda m: m.gnn_marl_system.drone_policy.epsilon if m.gnn_marl_system else 0,
                "Novel_Training_Steps": lambda m: m.gnn_marl_system.total_steps if m.gnn_marl_system else 0,
            },
            agent_reporters={
                "State": lambda a: getattr(a, 'state', None),
                "Battery": lambda a: getattr(a, 'battery', None),
                "Position": lambda a: getattr(a.cell, 'coordinate', None) if hasattr(a, 'cell') else None,
            }
        )

    def _print_initialization_summary(self, n_robots: int, n_drones: int, n_persons: int, n_chargers: int):
        """Print initialization summary."""
        if not self.quiet:
            print(f"🏔️ Mountain Rescue Model initialized:")
            print(f"   Mode: {self.mode}")
            print(f"   Grid: {self.width}x{self.height}")
            agent_str = f"   Agents: {n_robots} robots, {n_drones} drones, {n_persons} persons"
            if self.mode == "novel" and n_chargers > 0:
                agent_str += f", {n_chargers} chargers"
            print(agent_str)
            print(f"   Base: {self.base_position}")
            if self.mode in ["extended", "novel"]:
                print(f"   Dynamic spawning: Every {self.spawn_interval} steps (max {self.max_persons} persons)")

    # =============================================================================
    # TERRAIN AND ENVIRONMENT METHODS
    # =============================================================================

    def get_altitude_cost(self, coordinate: Tuple[int, int]) -> int:
        """
        Get movement cost based on altitude.

        Args:
            coordinate: (x, y) position on grid

        Returns:
            Movement cost (higher altitude = more energy)
        """
        x, y = coordinate
        if 0 <= x < self.width and 0 <= y < self.height:
            altitude = self.terrain[x, y]
            return int(1 + altitude)  # Base cost 1, +1 for each 1K altitude
        return 1

    def get_altitude(self, coordinate: Tuple[int, int]) -> int:
        """
        Get altitude at given coordinate.

        Args:
            coordinate: (x, y) position on grid

        Returns:
            Altitude level (0-3)
        """
        x, y = coordinate
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.terrain[x, y]
        return 0

    # =============================================================================
    # AGENT SPAWNING AND MANAGEMENT
    # =============================================================================

    def _spawn_persons(self, n_persons: int):
        """
        Spawn persons randomly across the mountain.

        Args:
            n_persons: Number of persons to spawn
        """
        # Get available cells (avoid base station)
        available_cells = [cell for cell in self.grid.all_cells.cells
                           if cell.coordinate != self.base_position]

        if len(available_cells) < n_persons:
            if not self.quiet:
                print(f"Warning: Not enough cells for {n_persons} persons. Placing {len(available_cells)} instead.")
            n_persons = len(available_cells)

        if n_persons > 0:
            selected_cells = self.random.choices(available_cells, k=n_persons)

            Person.create_agents(
                self,
                n_persons,
                cell=selected_cells,
                age=self.rng.integers(18, 80, n_persons),
                health=self.rng.uniform(0.1, 1.0, n_persons)
            )

            self.total_persons_spawned += n_persons
            if not self.quiet:
                print(f"🆘 Spawned {n_persons} persons across the mountain")

    def spawn_new_persons_extended_mode(self):
        """Spawn new persons dynamically during Extended and Novel Modes."""
        if self.mode not in ["extended", "novel"]:
            return

        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0

            # Check if we're at max capacity
            current_persons = len([p for p in self.agents if isinstance(p, Person)])
            if current_persons >= self.max_persons:
                if not self.quiet:
                    print(f"⚠️ Maximum persons limit ({self.max_persons}) reached, skipping spawn")
                return

            # Spawn 1-3 new persons randomly
            max_spawn = min(3, self.max_persons - current_persons)
            num_new = self.random.randint(1, max_spawn + 1)

            self._spawn_persons(num_new)
            self.persons_spawned_dynamically += num_new

    # =============================================================================
    # MODE SWITCHING AND CONFIGURATION
    # =============================================================================

    def switch_mode(self, new_mode: str):
        """
        Switch between operation modes.

        Args:
            new_mode: Target mode ("basic", "extended", or "novel")
        """
        if new_mode not in ["basic", "extended", "novel"]:
            raise ValueError("Mode must be 'basic', 'extended', or 'novel'")

        old_mode = self.mode
        self.mode = new_mode

        # Handle message system
        self._handle_message_system_transition(new_mode)

        # Handle GNN-MARL system
        self._handle_gnn_marl_transition(new_mode)

        if not self.quiet:
            print(f"🔄 Mode switched from {old_mode} to {new_mode}")

    def _handle_message_system_transition(self, new_mode: str):
        """Handle message system during mode transitions."""
        if new_mode in ["extended", "novel"] and self.message_system is None:
            self.message_system = MessageSystem()
            if not self.quiet:
                print(f"📡 Message system activated for {new_mode} Mode")
        elif new_mode == "basic" and self.message_system is not None:
            # Reset robot states that might be waiting for messages
            for agent in self.agents:
                if isinstance(agent, FirstAidRobot) and hasattr(agent, 'state'):
                    if agent.state.value == "waiting_for_mission":
                        agent.state = RobotState.IDLE
                        agent.target_coordinate = None
                        agent.target_person_id = None
            if not self.quiet:
                print(f"📡 Message system deactivated for Basic Mode")

    def _handle_gnn_marl_transition(self, new_mode: str):
        """Handle GNN-MARL system during mode transitions."""
        if new_mode == "novel" and self.gnn_marl_system is None:
            gnn_config = GNNConfig()
            marl_config = MARLConfig()
            self.gnn_marl_system = GNN_MARL_System(gnn_config, marl_config, device=self.device)
            self.gnn_marl_system.enable_training()
            if not self.quiet:
                print(f"🧠 Novel Mode (GNN-MARL) system initialized and training enabled")
        elif new_mode == "novel" and self.gnn_marl_system is not None:
            self.gnn_marl_system.enable_training()
            if not self.quiet:
                print(f"🧠 Novel Mode training enabled")
        elif new_mode != "novel" and self.gnn_marl_system is not None:
            self.gnn_marl_system.disable_training()
            if not self.quiet:
                print(f"🧠 Novel Mode training disabled")

    # =============================================================================
    # METRICS AND STATISTICS
    # =============================================================================

    def calculate_average_response_time(self) -> float:
        """
        Calculate average time from person discovery to rescue initiation.

        Returns:
            Average response time in steps
        """
        if not hasattr(self, '_response_times'):
            self._response_times = []

        rescued_persons = [p for p in self.agents if isinstance(p, Person) and p.is_rescued]
        if not rescued_persons:
            return 0.0

        # Estimate based on current step and rescue count
        return self.steps / max(1, len(rescued_persons))

    def calculate_communication_efficiency(self) -> float:
        """
        Calculate messages sent per successful rescue.

        Returns:
            Communication efficiency ratio
        """
        if self.message_system is None or self.rescued_count == 0:
            return 0.0

        return self.message_system.total_messages_sent / max(1, self.rescued_count)

    def get_mode_specific_stats(self) -> Dict[str, Any]:
        """
        Get statistics specific to current operation mode.

        Returns:
            Dictionary of mode-specific statistics
        """
        stats = {
            "mode": self.mode,
            "steps": self.steps,
            "rescued_count": self.rescued_count,
            "total_persons_spawned": self.total_persons_spawned,
        }

        if self.mode in ["extended", "novel"]:
            stats.update({
                "persons_spawned_dynamically": self.persons_spawned_dynamically,
                "communication_stats": self.message_system.get_communication_stats() if self.message_system else {},
                "robots_waiting_for_missions": len([r for r in self.agents
                                                    if isinstance(r, FirstAidRobot) and
                                                    hasattr(r, 'state') and r.state.value == "waiting_for_mission"]),
                "active_missions": len([r for r in self.agents
                                        if isinstance(r, FirstAidRobot) and
                                        hasattr(r, 'state') and r.state.value == "moving_to_target"]),
            })

        return stats

    def get_rescue_status(self) -> Dict[str, Any]:
        """
        Get current rescue status summary.

        Returns:
            Dictionary of current rescue operation status
        """
        persons = [a for a in self.agents if isinstance(a, Person)]
        robots = [a for a in self.agents if isinstance(a, FirstAidRobot)]
        drones = [a for a in self.agents if isinstance(a, ExplorerDrone)]

        status = {
            "mode": self.mode,
            "step": self.steps,
            "total_persons": len(persons),
            "rescued": len([p for p in persons if p.is_rescued]),
            "missing": len([p for p in persons if not p.is_rescued]),
            "active_robots": len([r for r in robots if r.battery > 0]),
            "active_drones": len([d for d in drones if d.battery > 0]),
            "total_persons_spawned": self.total_persons_spawned,
        }

        if self.mode in ["extended", "novel"]:
            status.update({
                "persons_spawned_dynamically": self.persons_spawned_dynamically,
                "robots_waiting": len(
                    [r for r in robots if hasattr(r, 'state') and r.state.value == "waiting_for_mission"]),
                "robots_on_mission": len(
                    [r for r in robots if hasattr(r, 'state') and r.state.value == "moving_to_target"]),
                "messages_sent": self.message_system.total_messages_sent if self.message_system else 0,
            })

        return status

    # =============================================================================
    # SIMULATION EXECUTION
    # =============================================================================

    def step(self):
        """Execute one step of the simulation."""
        # Handle dynamic person spawning in Extended and Novel Modes
        if self.mode in ["extended", "novel"]:
            self.spawn_new_persons_extended_mode()

        # Execute agent actions based on mode
        if self.gnn_marl_system and self.mode == "novel":
            # Novel Mode: GNN-MARL takes over agent actions while preserving Extended Mode features
            self.gnn_marl_system.train_step(self)
            
            # Ensure MobileCharger agents still get stepped (not handled by GNN-MARL)
            for agent in self.agents:
                if isinstance(agent, MobileCharger):
                    agent.step()
        else:
            # Standard agent stepping for basic/extended modes
            self.agents.shuffle_do("step")

        # Clean up old messages in Extended and Novel Modes
        if self.message_system:
            self.message_system.clear_old_messages(self.steps)

        # Collect performance data
        self.datacollector.collect(self)

        # Check completion conditions
        self._check_completion_conditions()

    def _check_completion_conditions(self):
        """Check if simulation should end based on completion conditions."""
        persons = [a for a in self.agents if isinstance(a, Person)]
        if persons and all(person.is_rescued for person in persons):
            if self.mode == "basic":
                if not self.quiet:
                    print(f"🎉 All persons rescued in {self.steps} steps!")
                # Model can stop or continue
            # In Extended and Novel Modes, new persons keep spawning, so don't stop


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_model_with_mode(mode: str = "basic", **kwargs) -> MountainRescueModel:
    """
    Factory function to create model with specified mode.

    Args:
        mode: Operation mode ("basic", "extended", or "novel")
        **kwargs: Additional arguments for MountainRescueModel

    Returns:
        Configured MountainRescueModel instance
    """
    return MountainRescueModel(mode=mode, **kwargs)