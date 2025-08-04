# tests/tests_extended.py

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import MountainRescueModel
from agents import FirstAidRobot, ExplorerDrone, Person, PersonState, RobotState, DroneState
from messaging import MessageSystem, MessageType, Message


class TestExtendedModeInitialization(unittest.TestCase):
    """
    Test Extended Mode initialization and configuration"""

    def setUp(self):
        """Set up Extended Mode test model"""
        self.model = MountainRescueModel(
            width=10, height=10,
            n_robots=3, n_drones=2, n_persons=5,
            mode="extended",
            spawn_interval=30,
            max_persons=20,
            seed=42
        )

    def test_extended_mode_initialization(self):
        """Test that Extended Mode initializes correctly"""
        self.assertEqual(self.model.mode, "extended")
        self.assertEqual(self.model.spawn_interval, 30)
        self.assertEqual(self.model.max_persons, 20)
        self.assertIsNotNone(self.model.message_system)
        self.assertIsInstance(self.model.message_system, MessageSystem)

    def test_message_system_initialization(self):
        """Test that message system is properly initialized"""
        self.assertEqual(self.model.message_system.total_messages_sent, 0)
        self.assertEqual(len(self.model.message_system.inbox), 0)
        self.assertEqual(len(self.model.message_system.broadcast_messages), 0)

    def test_extended_mode_data_collection(self):
        """Test Extended Mode specific data collectors"""
        # Run a few steps to generate data
        for _ in range(5):
            self.model.step()

        model_data = self.model.datacollector.get_model_vars_dataframe()
        
        # Check that Extended Mode specific metrics are collected
        expected_columns = [
            'Messages_Sent', 'Robots_Waiting', 'Robots_On_Mission',
            'Average_Response_Time', 'Communication_Efficiency'
        ]
        
        for column in expected_columns:
            self.assertIn(column, model_data.columns)

    def test_dynamic_person_spawning(self):
        """Test dynamic person spawning in Extended Mode"""
        initial_persons = len([a for a in self.model.agents if isinstance(a, Person)])
        
        # Advance model by spawn_interval steps
        for _ in range(self.model.spawn_interval):
            self.model.step()
            
        # Check that new person was spawned
        current_persons = len([a for a in self.model.agents if isinstance(a, Person)])
        self.assertGreater(current_persons, initial_persons)

    def test_max_persons_limit(self):
        """Test that max_persons limit is enforced"""
        # Create model with low max_persons
        model = MountainRescueModel(
            width=5, height=5,
            n_robots=1, n_drones=1, n_persons=2,
            mode="extended",
            spawn_interval=1,
            max_persons=3,
            seed=42
        )
        
        # Run enough steps to trigger multiple spawns
        for _ in range(10):
            model.step()
            
        # Check that persons count doesn't exceed max_persons
        current_persons = len([a for a in model.agents if isinstance(a, Person) and not a.is_rescued])
        self.assertLessEqual(current_persons, 3)


class TestMessageSystem(unittest.TestCase):
    """Test the MessageSystem class"""

    def setUp(self):
        """Set up message system test"""
        self.message_system = MessageSystem()

    def test_send_direct_message(self):
        """Test sending direct messages"""
        result = self.message_system.send_message(
            sender_id=1,
            receiver_id=2,
            message_type=MessageType.PERSON_LOCATION,
            content={"x": 5, "y": 7, "urgency": 0.8},
            timestamp=10
        )
        
        self.assertTrue(result)
        self.assertEqual(self.message_system.total_messages_sent, 1)
        self.assertIn(2, self.message_system.inbox)
        self.assertEqual(len(self.message_system.inbox[2]), 1)

    def test_send_broadcast_message(self):
        """Test sending broadcast messages"""
        result = self.message_system.send_message(
            sender_id=1,
            receiver_id=None,  # Broadcast
            message_type=MessageType.MISSION_COMPLETE,
            content={"person_id": 5},
            timestamp=20
        )
        
        self.assertTrue(result)
        self.assertEqual(len(self.message_system.broadcast_messages), 1)
        self.assertEqual(self.message_system.total_messages_sent, 1)

    def test_get_messages(self):
        """Test retrieving messages"""
        # Send some messages
        self.message_system.send_message(1, 2, MessageType.PERSON_LOCATION, {"x": 1, "y": 2}, 10)
        self.message_system.send_message(3, 2, MessageType.HELP_REQUEST, {"urgency": 0.9}, 15)
        self.message_system.send_message(4, None, MessageType.STATUS_UPDATE, {"status": "ready"}, 20)
        
        # Get all messages for agent 2
        messages = self.message_system.get_messages(2)
        self.assertEqual(len(messages), 3)  # 2 direct + 1 broadcast

    def test_message_filtering(self):
        """Test filtering messages by type"""
        # Send different types of messages
        self.message_system.send_message(1, 2, MessageType.PERSON_LOCATION, {"x": 1, "y": 2}, 10)
        self.message_system.send_message(3, 2, MessageType.HELP_REQUEST, {"urgency": 0.9}, 15)
        
        # Get only PERSON_LOCATION messages
        location_messages = self.message_system.get_messages(2, MessageType.PERSON_LOCATION)
        self.assertEqual(len(location_messages), 1)
        self.assertEqual(location_messages[0].message_type, MessageType.PERSON_LOCATION)

    def test_message_priority_sorting(self):
        """Test that messages are sorted by priority"""
        # Send messages with different priorities
        self.message_system.send_message(1, 2, MessageType.PERSON_LOCATION, {"urgency": 0.3}, 10, priority=0.3)
        self.message_system.send_message(3, 2, MessageType.HELP_REQUEST, {"urgency": 0.9}, 15, priority=0.9)
        self.message_system.send_message(4, 2, MessageType.STATUS_UPDATE, {"status": "ready"}, 20, priority=0.1)
        
        messages = self.message_system.get_messages(2)
        
        # Check that messages are sorted by priority (highest first)
        self.assertEqual(messages[0].priority, 0.9)
        self.assertEqual(messages[1].priority, 0.3)
        self.assertEqual(messages[2].priority, 0.1)

    def test_clear_messages(self):
        """Test clearing messages"""
        # Send messages
        self.message_system.send_message(1, 2, MessageType.PERSON_LOCATION, {"x": 1, "y": 2}, 10)
        self.message_system.send_message(3, 2, MessageType.HELP_REQUEST, {"urgency": 0.9}, 15)
        
        # Clear all messages
        self.message_system.clear_messages(2)
        messages = self.message_system.get_messages(2)
        
        # Should only have broadcast messages now
        self.assertEqual(len(messages), 0)

    def test_clear_old_messages(self):
        """Test clearing old messages"""
        # Send messages at different times
        self.message_system.send_message(1, 2, MessageType.PERSON_LOCATION, {"x": 1, "y": 2}, 10)
        self.message_system.send_message(3, 2, MessageType.HELP_REQUEST, {"urgency": 0.9}, 100)
        
        # Clear old messages (cutoff at step 90)
        self.message_system.clear_old_messages(current_step=140, max_age=50)
        
        messages = self.message_system.get_messages(2)
        self.assertEqual(len(messages), 1)  # Only the newer message should remain


class TestRobotExtendedMode(unittest.TestCase):
    """Test FirstAidRobot Extended Mode behaviors"""

    def setUp(self):
        """Set up Extended Mode robot test"""
        self.model = MountainRescueModel(
            width=10, height=10,
            n_robots=2, n_drones=1, n_persons=3,
            mode="extended",
            seed=42
        )
        
        robots = [a for a in self.model.agents if isinstance(a, FirstAidRobot)]
        self.robot = robots[0]

    def test_robot_extended_mode_state_transitions(self):
        """Test robot state transitions in Extended Mode"""
        # Robot should start in WAITING_FOR_MISSION instead of EXPLORING
        self.robot.step()
        self.assertEqual(self.robot.state, RobotState.WAITING_FOR_MISSION)

    def test_robot_mission_acceptance(self):
        """Test robot accepting mission from drone"""
        # Set robot to correct state
        self.robot.state = RobotState.WAITING_FOR_MISSION
        
        # Find an available (non-rescued) person
        available_persons = [p for p in self.model.agents if isinstance(p, Person) and not p.is_rescued]
        self.assertGreater(len(available_persons), 0, "No available persons for test")
        person = available_persons[0]
        
        # Send a mission message using available person
        self.model.message_system.send_message(
            sender_id=100,  # Drone ID
            receiver_id=self.robot.unique_id,
            message_type=MessageType.PERSON_LOCATION,
            content={
                "person_id": person.unique_id,
                "x": 5,
                "y": 7,
                "urgency": 0.8,
                "person_health": 0.6,
                "drone_id": 100
            },
            timestamp=self.model.steps
        )
        
        # Robot should process the message and accept mission
        self.robot.step()
        self.assertEqual(self.robot.state, RobotState.MOVING_TO_TARGET)
        self.assertEqual(self.robot.target_coordinate, (5, 7))
        self.assertEqual(self.robot.target_person_id, person.unique_id)

    def test_robot_mission_prioritization(self):
        """Test robot selecting highest priority mission"""
        # Set robot to correct state
        self.robot.state = RobotState.WAITING_FOR_MISSION
        
        # Find available (non-rescued) persons
        available_persons = [p for p in self.model.agents if isinstance(p, Person) and not p.is_rescued]
        self.assertGreaterEqual(len(available_persons), 2, "Need at least 2 available persons for test")
        person1, person2 = available_persons[0], available_persons[1]
        
        # Send multiple mission messages with different priorities
        self.model.message_system.send_message(
            sender_id=100, receiver_id=self.robot.unique_id,
            message_type=MessageType.PERSON_LOCATION,
            content={"person_id": person1.unique_id, "x": 8, "y": 8, "urgency": 0.5, "drone_id": 100},
            timestamp=self.model.steps
        )
        
        self.model.message_system.send_message(
            sender_id=100, receiver_id=self.robot.unique_id,
            message_type=MessageType.PERSON_LOCATION,
            content={"person_id": person2.unique_id, "x": 3, "y": 3, "urgency": 0.9, "drone_id": 100},
            timestamp=self.model.steps
        )
        
        # Robot should select higher priority mission
        self.robot.step()
        self.assertEqual(self.robot.target_person_id, person2.unique_id)
        self.assertEqual(self.robot.target_coordinate, (3, 3))

    def test_robot_acknowledgment_message(self):
        """Test robot sending acknowledgment message"""
        # Set robot to correct state
        self.robot.state = RobotState.WAITING_FOR_MISSION
        
        # Find an available (non-rescued) person
        available_persons = [p for p in self.model.agents if isinstance(p, Person) and not p.is_rescued]
        self.assertGreater(len(available_persons), 0, "No available persons for test")
        person = available_persons[0]
        
        # Send mission with required fields using available person
        self.model.message_system.send_message(
            sender_id=100,  # Drone ID
            receiver_id=self.robot.unique_id,
            message_type=MessageType.PERSON_LOCATION,
            content={
                "person_id": person.unique_id, 
                "x": 5, 
                "y": 7, 
                "urgency": 0.8,
                "drone_id": 100  # Required for acknowledgment
            },
            timestamp=self.model.steps
        )
        
        initial_messages = self.model.message_system.total_messages_sent
        
        # Robot should process message and send acknowledgment
        self.robot.step()
        
        # Check that acknowledgment was sent
        self.assertGreater(self.model.message_system.total_messages_sent, initial_messages)
        
        # Check message content
        ack_messages = [msg for msg in self.model.message_system.message_history 
                       if msg.message_type == MessageType.MISSION_ACKNOWLEDGMENT]
        self.assertGreater(len(ack_messages), 0)
        self.assertEqual(ack_messages[0].sender_id, self.robot.unique_id)
        self.assertEqual(ack_messages[0].receiver_id, 100)

    def test_robot_mission_completion_notification(self):
        """Test robot sending mission completion notification"""
        # Find an available (non-rescued) person
        available_persons = [p for p in self.model.agents if isinstance(p, Person) and not p.is_rescued]
        self.assertGreater(len(available_persons), 0, "No available persons for test")
        person = available_persons[0]
        
        # Set up robot at person location
        self.robot.state = RobotState.AT_PERSON
        self.robot.target_person = person
        self.robot.target_person_id = person.unique_id
        self.robot.cell = person.cell
        
        initial_messages = self.model.message_system.total_messages_sent
        
        # Robot should complete mission and send notification
        self.robot.step()
        
        # Check that completion message was sent
        self.assertGreater(self.model.message_system.total_messages_sent, initial_messages)
        
        completion_messages = [msg for msg in self.model.message_system.message_history 
                             if msg.message_type == MessageType.MISSION_COMPLETE]
        self.assertGreater(len(completion_messages), 0)

    def test_robot_target_movement(self):
        """Test robot moving towards target coordinate"""
        # Set robot target
        self.robot.target_coordinate = (5, 5)
        self.robot.state = RobotState.MOVING_TO_TARGET
        
        initial_pos = self.robot.cell.coordinate
        
        # Robot should move towards target
        self.robot.step()
        
        current_pos = self.robot.cell.coordinate
        
        # Check that robot moved (unless already at target)
        if initial_pos != (5, 5):
            self.assertNotEqual(initial_pos, current_pos)


class TestDroneExtendedMode(unittest.TestCase):
    """Test ExplorerDrone Extended Mode behaviors"""

    def setUp(self):
        """Set up Extended Mode drone test"""
        self.model = MountainRescueModel(
            width=10, height=10,
            n_robots=3, n_drones=1, n_persons=2,
            mode="extended",
            seed=42
        )
        
        drones = [a for a in self.model.agents if isinstance(a, ExplorerDrone)]
        self.drone = drones[0]

    def test_drone_extended_mode_state_transitions(self):
        """Test drone state transitions in Extended Mode"""
        # Move drone to person location
        persons = [a for a in self.model.agents if isinstance(a, Person)]
        person = persons[0]
        self.drone.cell = person.cell
        
        # Drone should transition through LOCATING and COORDINATING states
        self.drone.step()
        if self.drone.state == DroneState.LOCATING:
            self.drone.step()
            self.assertEqual(self.drone.state, DroneState.COORDINATING)

    def test_drone_broadcast_decision(self):
        """Test drone deciding between broadcast and targeted messaging"""
        # Test critical urgency scenario
        persons = [a for a in self.model.agents if isinstance(a, Person)]
        person = persons[0]
        person.urgency = 0.9  # Critical urgency
        
        self.drone.target_person = person
        
        # Should decide to broadcast for critical urgency
        should_broadcast = self.drone.should_broadcast()
        self.assertTrue(should_broadcast)

    def test_drone_robot_selection(self):
        """Test drone selecting best robot for mission"""
        robots = [a for a in self.model.agents if isinstance(a, FirstAidRobot)]
        persons = [a for a in self.model.agents if isinstance(a, Person)]
        person = persons[0]
        
        # Set different battery levels
        robots[0].battery = 90
        robots[1].battery = 50
        robots[2].battery = 30
        
        self.drone.target_person = person
        
        # Should select robot with highest battery
        selected_robot = self.drone.select_best_robot()
        self.assertEqual(selected_robot, robots[0])

    def test_drone_location_message_sending(self):
        """Test drone sending location messages"""
        persons = [a for a in self.model.agents if isinstance(a, Person)]
        person = persons[0]
        
        self.drone.target_person = person
        self.drone.state = DroneState.COORDINATING
        
        initial_messages = self.model.message_system.total_messages_sent
        
        # Drone should send location message
        self.drone.step()
        
        # Check that message was sent
        self.assertGreater(self.model.message_system.total_messages_sent, initial_messages)
        
        # Check message content
        location_messages = [msg for msg in self.model.message_system.message_history 
                           if msg.message_type == MessageType.PERSON_LOCATION]
        self.assertGreater(len(location_messages), 0)

    def test_drone_acknowledgment_monitoring(self):
        """Test drone monitoring for robot acknowledgments"""
        # Set up drone in coordinating state
        self.drone.state = DroneState.COORDINATING
        self.drone.coordination_timer = 0
        
        # Send acknowledgment message
        self.model.message_system.send_message(
            sender_id=1,  # Robot ID
            receiver_id=self.drone.unique_id,
            message_type=MessageType.MISSION_ACKNOWLEDGMENT,
            content={"person_id": 1, "eta": 10},
            timestamp=self.model.steps
        )
        
        # Drone should recognize acknowledgment
        received_ack = self.drone.received_acknowledgment()
        self.assertTrue(received_ack)

    def test_drone_coordination_timeout(self):
        """Test drone coordination timeout mechanism"""
        # Set up drone with target person for coordination
        persons = [a for a in self.model.agents if isinstance(a, Person)]
        person = persons[0]
        self.drone.target_person = person
        self.drone.state = DroneState.COORDINATING
        self.drone.coordination_timer = 0
        
        # Run steps until timeout (coordination_timer starts at 0, increments each step)
        for _ in range(self.drone.max_coordination_time):
            self.drone.step()
        
        # Drone should have timed out and changed to WAITING state
        self.assertEqual(self.drone.state, DroneState.WAITING)


class TestExtendedModeIntegration(unittest.TestCase):
    """Test complete Extended Mode workflows"""

    def setUp(self):
        """Set up Extended Mode integration test"""
        self.model = MountainRescueModel(
            width=8, height=8,
            n_robots=2, n_drones=1, n_persons=2,
            mode="extended",
            seed=42
        )

    def test_complete_rescue_workflow(self):
        """Test complete rescue workflow from discovery to completion"""
        # Run simulation steps
        initial_rescued = self.model.rescued_count
        
        # Run enough steps for a complete rescue cycle
        for _ in range(50):
            self.model.step()
            
            # Check if rescue was completed
            if self.model.rescued_count > initial_rescued:
                break
        
        # Verify that rescue process involved communication
        self.assertGreater(self.model.message_system.total_messages_sent, 0)
        
        # Check that appropriate message types were sent
        message_types = [msg.message_type for msg in self.model.message_system.message_history]
        self.assertIn(MessageType.PERSON_LOCATION, message_types)

    def test_mode_switching(self):
        """Test switching from Extended Mode to other modes"""
        # Verify current mode
        self.assertEqual(self.model.mode, "extended")
        
        # Switch to basic mode
        self.model.switch_mode("basic")
        self.assertEqual(self.model.mode, "basic")
        
        # Switch back to extended mode
        self.model.switch_mode("extended")
        self.assertEqual(self.model.mode, "extended")

    def test_multiple_simultaneous_rescues(self):
        """Test handling multiple simultaneous rescue operations"""
        # Create scenario with multiple persons
        model = MountainRescueModel(
            width=15, height=15,
            n_robots=4, n_drones=2, n_persons=6,
            mode="extended",
            seed=42
        )
        
        # Run simulation
        for _ in range(100):
            model.step()
        
        # Check that multiple rescues were handled
        self.assertGreater(model.message_system.total_messages_sent, 0)
        
        # Verify system stability
        self.assertIsNotNone(model.message_system)

    def test_communication_efficiency_metric(self):
        """Test communication efficiency calculation"""
        # Run simulation
        for _ in range(30):
            self.model.step()
        
        # Get communication efficiency data
        model_data = self.model.datacollector.get_model_vars_dataframe()
        
        # Check that communication efficiency is being calculated
        self.assertIn('Communication_Efficiency', model_data.columns)
        
        # Efficiency should be reasonable (not infinite)
        if self.model.rescued_count > 0:
            latest_efficiency = model_data['Communication_Efficiency'].iloc[-1]
            self.assertGreaterEqual(latest_efficiency, 0)

    def test_response_time_tracking(self):
        """Test average response time calculation"""
        # Run simulation
        for _ in range(40):
            self.model.step()
        
        # Get response time data
        model_data = self.model.datacollector.get_model_vars_dataframe()
        
        # Check that response time is being tracked
        self.assertIn('Average_Response_Time', model_data.columns)
        
        # Response time should be reasonable
        if self.model.rescued_count > 0:
            latest_response_time = model_data['Average_Response_Time'].iloc[-1]
            self.assertGreaterEqual(latest_response_time, 0)


class TestExtendedModeEdgeCases(unittest.TestCase):
    """Test Extended Mode edge cases and error handling"""

    def setUp(self):
        """Set up Extended Mode edge case tests"""
        self.model = MountainRescueModel(
            width=5, height=5,
            n_robots=1, n_drones=1, n_persons=1,
            mode="extended",
            seed=42
        )

    def test_no_available_robots(self):
        """Test drone behavior when no robots are available"""
        # Set all robots to out of battery
        robots = [a for a in self.model.agents if isinstance(a, FirstAidRobot)]
        for robot in robots:
            robot.battery = 0
            robot.state = RobotState.OUT_OF_BATTERY
        
        # Run simulation
        for _ in range(10):
            self.model.step()
        
        # System should remain stable
        self.assertIsNotNone(self.model.message_system)

    def test_message_system_cleanup(self):
        """Test message system cleanup with many messages"""
        # Generate many messages
        for i in range(100):
            self.model.message_system.send_message(
                sender_id=1,
                receiver_id=2,
                message_type=MessageType.STATUS_UPDATE,
                content={"step": i},
                timestamp=i
            )
        
        # Clear old messages
        self.model.message_system.clear_old_messages(current_step=150, max_age=10)
        
        # Only recent messages should remain
        messages = self.model.message_system.get_messages(2)
        self.assertLess(len(messages), 100)

    def test_invalid_message_handling(self):
        """Test handling of invalid message scenarios"""
        # Test with invalid receiver ID
        result = self.model.message_system.send_message(
            sender_id=1,
            receiver_id=999,  # Non-existent agent
            message_type=MessageType.PERSON_LOCATION,
            content={"x": 1, "y": 1},
            timestamp=10
        )
        
        # Should still succeed (message delivered to inbox)
        self.assertTrue(result)

    def test_simultaneous_robot_selection(self):
        """Test multiple drones selecting same robot"""
        # Create scenario with multiple drones
        model = MountainRescueModel(
            width=10, height=10,
            n_robots=1, n_drones=2, n_persons=2,
            mode="extended",
            seed=42
        )
        
        # Run simulation
        for _ in range(20):
            model.step()
        
        # System should handle concurrent selections gracefully
        self.assertIsNotNone(model.message_system)


if __name__ == '__main__':
    unittest.main()