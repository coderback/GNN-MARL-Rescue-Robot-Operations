# tests/tests_basic.py

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import MountainRescueModel
from agents import FirstAidRobot, ExplorerDrone, Person, PersonState, RobotState, DroneState


class TestMountainRescueModel(unittest.TestCase):
    """Test the MountainRescueModel class"""

    def setUp(self):
        """Set up test model before each test"""
        self.model = MountainRescueModel(
            width=10, height=10,
            n_robots=2, n_drones=1, n_persons=2,
            seed=42
        )

    def test_model_initialization(self):
        """Test that model initializes correctly"""
        self.assertEqual(self.model.width, 10)
        self.assertEqual(self.model.height, 10)
        self.assertEqual(self.model.mode, "basic")
        self.assertEqual(self.model.rescued_count, 0)
        self.assertIsNotNone(self.model.grid)
        self.assertEqual(self.model.base_position, (0, 0))

    def test_agent_creation(self):
        """Test that agents are created correctly"""
        # Get all agents using the model's agents attribute
        all_agents = list(self.model.agents)

        robots = [a for a in all_agents if isinstance(a, FirstAidRobot)]
        drones = [a for a in all_agents if isinstance(a, ExplorerDrone)]
        persons = [a for a in all_agents if isinstance(a, Person)]

        self.assertEqual(len(robots), 2)
        self.assertEqual(len(drones), 1)
        self.assertEqual(len(persons), 2)

        # Check that robots and drones start at base
        for robot in robots:
            self.assertEqual(robot.cell.coordinate, (0, 0))
        for drone in drones:
            self.assertEqual(drone.cell.coordinate, (0, 0))

    def test_terrain_creation(self):
        """Test that terrain is created properly"""
        self.assertEqual(self.model.terrain.shape, (10, 10))

        # Check altitude cost function
        cost_base = self.model.get_altitude_cost((0, 0))
        cost_center = self.model.get_altitude_cost((5, 5))

        self.assertGreaterEqual(cost_base, 1)
        self.assertGreaterEqual(cost_center, cost_base)

    def test_data_collection(self):
        """Test that data collection works"""
        # Run a few steps
        for _ in range(3):
            self.model.step()

        # Check that data was collected
        model_data = self.model.datacollector.get_model_vars_dataframe()
        self.assertGreater(len(model_data), 0)

        agent_data = self.model.datacollector.get_agent_vars_dataframe()
        self.assertGreater(len(agent_data), 0)


class TestPerson(unittest.TestCase):
    """Test the Person agent class"""

    def setUp(self):
        """Set up test person"""
        self.model = MountainRescueModel(width=5, height=5, n_robots=0, n_drones=0, n_persons=1, seed=42)
        # Safely get the person agent
        all_agents = list(self.model.agents)
        persons = [a for a in all_agents if isinstance(a, Person)]
        self.assertGreater(len(persons), 0, "No Person agents were created")
        self.person = persons[0]

    def test_person_initialization(self):
        """Test person initialization"""
        self.assertEqual(self.person.state, PersonState.MISSING)
        self.assertFalse(self.person.is_rescued)
        self.assertIsNotNone(self.person.age)
        self.assertIsNotNone(self.person.health)
        self.assertIsNotNone(self.person.urgency)

        # Test age and health ranges
        self.assertGreaterEqual(self.person.age, 18)
        self.assertLessEqual(self.person.age, 80)
        self.assertGreaterEqual(self.person.health, 0.1)
        self.assertLessEqual(self.person.health, 1.0)

    def test_person_rescue(self):
        """Test person rescue functionality"""
        self.assertFalse(self.person.is_rescued)

        self.person.rescue()

        self.assertTrue(self.person.is_rescued)
        self.assertEqual(self.person.state, PersonState.RESCUED)

    def test_urgency_calculation(self):
        """Test urgency calculation"""
        # Get a cell from the model to create test persons
        test_cell = list(self.model.grid.all_cells.cells)[0]

        # Test with specific values
        person_young_healthy = Person(self.model, test_cell, age=30, health=0.9)
        person_old_sick = Person(self.model, test_cell, age=70, health=0.2)

        # Older, sicker person should have higher urgency
        self.assertGreater(person_old_sick.urgency, person_young_healthy.urgency)


class TestFirstAidRobot(unittest.TestCase):
    """Test the FirstAidRobot agent class"""

    def setUp(self):
        """Set up test robot"""
        self.model = MountainRescueModel(width=5, height=5, n_robots=1, n_drones=0, n_persons=1, seed=42)

        # Safely get the robot and person agents
        all_agents = list(self.model.agents)
        robots = [a for a in all_agents if isinstance(a, FirstAidRobot)]
        persons = [a for a in all_agents if isinstance(a, Person)]

        self.assertGreater(len(robots), 0, "No FirstAidRobot agents were created")
        self.assertGreater(len(persons), 0, "No Person agents were created")

        self.robot = robots[0]
        self.person = persons[0]

    def test_robot_initialization(self):
        """Test robot initialization"""
        self.assertEqual(self.robot.state, RobotState.IDLE)
        self.assertEqual(self.robot.battery, 100)
        self.assertEqual(self.robot.cell.coordinate, (0, 0))
        self.assertIsNone(self.robot.target_person)

    def test_robot_movement(self):
        """Test robot movement and battery consumption"""
        initial_battery = self.robot.battery

        self.robot.move_random()

        # Battery should decrease
        self.assertLess(self.robot.battery, initial_battery)

    def test_robot_rescue(self):
        """Test robot rescue functionality"""
        # Move robot to same cell as person
        self.robot.cell = self.person.cell

        # Simulate rescue process
        found = self.robot.search_for_person()
        if found:
            rescued = self.robot.deliver_aid()
            self.assertTrue(rescued)
            self.assertTrue(self.person.is_rescued)

    def test_robot_state_transitions(self):
        """Test robot state machine"""
        # Start idle
        self.assertEqual(self.robot.state, RobotState.IDLE)

        # First step should go to exploring (based on the actual implementation)
        self.robot.step()
        self.assertEqual(self.robot.state, RobotState.EXPLORING)

        # Move robot away from base to test battery depletion properly
        # Find a cell that's not the base position
        target_cell = None
        for cell in self.model.grid.all_cells:
            if cell.coordinate != self.model.base_position:
                target_cell = cell
                break
        
        if target_cell:
            self.robot.cell = target_cell
        
        # Test battery depletion (should stay OUT_OF_BATTERY when not at base)
        self.robot.battery = 0
        self.robot.step()
        self.assertEqual(self.robot.state, RobotState.OUT_OF_BATTERY)


class TestExplorerDrone(unittest.TestCase):
    """Test the ExplorerDrone agent class"""

    def setUp(self):
        """Set up test drone"""
        self.model = MountainRescueModel(width=5, height=5, n_robots=0, n_drones=1, n_persons=1, seed=42)

        # Safely get the drone and person agents
        all_agents = list(self.model.agents)
        drones = [a for a in all_agents if isinstance(a, ExplorerDrone)]
        persons = [a for a in all_agents if isinstance(a, Person)]

        self.assertGreater(len(drones), 0, "No ExplorerDrone agents were created")
        self.assertGreater(len(persons), 0, "No Person agents were created")

        self.drone = drones[0]
        self.person = persons[0]

    def test_drone_initialization(self):
        """Test drone initialization"""
        self.assertEqual(self.drone.state, DroneState.IDLE)
        self.assertEqual(self.drone.battery, 150)
        self.assertEqual(self.drone.cell.coordinate, (0, 0))
        self.assertIsNone(self.drone.target_person)

    def test_drone_movement(self):
        """Test drone 3D movement"""
        initial_battery = self.drone.battery

        self.drone.move_3d()

        # Battery should decrease more than robots (flight cost)
        self.assertLess(self.drone.battery, initial_battery)

    def test_drone_search(self):
        """Test drone person search and assessment"""
        # Move drone to same cell as person
        self.drone.cell = self.person.cell

        found = self.drone.search_for_person()
        if found:
            self.assertEqual(self.drone.target_person, self.person)

            # Test urgency assessment
            urgency = self.drone.assess_urgency()
            self.assertGreaterEqual(urgency, 0)
            self.assertLessEqual(urgency, 1)

    def test_drone_state_transitions(self):
        """Test drone state machine"""
        # Start idle
        self.assertEqual(self.drone.state, DroneState.IDLE)

        # First step should go to flying (based on the actual implementation)
        self.drone.step()
        self.assertEqual(self.drone.state, DroneState.FLYING)

        # Move drone away from base to test battery depletion properly
        # Find a cell that's not the base position
        target_cell = None
        for cell in self.model.grid.all_cells:
            if cell.coordinate != self.model.base_position:
                target_cell = cell
                break
        
        if target_cell:
            self.drone.cell = target_cell
        
        # Test battery depletion (should stay OUT_OF_BATTERY when not at base)
        self.drone.battery = 0
        self.drone.step()
        self.assertEqual(self.drone.state, DroneState.OUT_OF_BATTERY)


class TestEnvironmentCreation(unittest.TestCase):
    """Test environment creation and key functions"""

    def test_small_environment(self):
        """Test creation of small environment"""
        model = MountainRescueModel(width=3, height=3, n_robots=1, n_drones=1, n_persons=1, seed=42)

        self.assertIsNotNone(model.grid)
        self.assertEqual(len(list(model.grid.all_cells.cells)), 9)  # 3x3 = 9 cells

        # Check agent counts
        all_agents = list(model.agents)
        robots = [a for a in all_agents if isinstance(a, FirstAidRobot)]
        drones = [a for a in all_agents if isinstance(a, ExplorerDrone)]
        persons = [a for a in all_agents if isinstance(a, Person)]

        self.assertEqual(len(robots), 1)
        self.assertEqual(len(drones), 1)
        self.assertEqual(len(persons), 1)

    def test_large_environment(self):
        """Test creation of larger environment"""
        model = MountainRescueModel(width=50, height=50, n_robots=10, n_drones=5, n_persons=20, seed=42)

        self.assertIsNotNone(model.grid)
        self.assertEqual(len(list(model.grid.all_cells.cells)), 2500)  # 50x50 = 2500 cells

        # Check agent counts
        all_agents = list(model.agents)
        robots = [a for a in all_agents if isinstance(a, FirstAidRobot)]
        drones = [a for a in all_agents if isinstance(a, ExplorerDrone)]
        persons = [a for a in all_agents if isinstance(a, Person)]

        self.assertEqual(len(robots), 10)
        self.assertEqual(len(drones), 5)
        self.assertEqual(len(persons), 20)


if __name__ == '__main__':
    unittest.main()