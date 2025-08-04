"""Agent classes for Mountain Rescue Simulation.

This module contains the core agent implementations for the mountain rescue simulation:
- Person: Static agents representing people to rescue
- FirstAidRobot: Ground-based rescue robots with terrain navigation
- ExplorerDrone: Aerial drones for reconnaissance and coordination
- MobileCharger: Ground-based rovers for recharging agents when power depletes
"""

import random
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any

from mesa import Agent
from mesa.discrete_space import CellAgent

# Import MessageType with fallback if messaging.py doesn't exist yet
try:
    from messaging import MessageType
except ImportError:
    # Fallback if messaging.py doesn't exist yet
    class MessageType:
        PERSON_LOCATION = "person_location"
        MISSION_ACKNOWLEDGMENT = "mission_acknowledgment"
        MISSION_COMPLETE = "mission_complete"
        HELP_REQUEST = "help_request"
        STATUS_UPDATE = "status_update"


# ================================
# AGENT STATE ENUMS
# ================================

class PersonState(Enum):
    """States for Person agents."""
    MISSING = "missing"
    RESCUED = "rescued"


class RobotState(Enum):
    """States for FirstAidRobot agents."""
    IDLE = "idle"
    WAITING_FOR_MISSION = "waiting_for_mission"  # Extended Mode - waiting for drone signal
    EXPLORING = "exploring"  # Basic Mode - random exploration
    MOVING_TO_TARGET = "moving_to_target"  # Extended Mode - moving to coordinates
    AT_PERSON = "at_person"
    RETURNING = "returning"
    OUT_OF_BATTERY = "out_of_battery"


class DroneState(Enum):
    """States for ExplorerDrone agents."""
    IDLE = "idle"
    FLYING = "flying"
    LOCATING = "locating"  # Found person, assessing urgency
    COORDINATING = "coordinating"  # Sending messages to robots
    WAITING = "waiting"
    RETURNING = "returning"
    OUT_OF_BATTERY = "out_of_battery"


class MobileChargerState(Enum):
    """States for MobileCharger agents."""
    IDLE_AT_BASE = "idle_at_base"
    MOVING_TO_AGENT = "moving_to_agent"
    CHARGING_AGENT = "charging_agent"
    RETURNING_TO_BASE = "returning_to_base"
    CHARGING_SELF = "charging_self"


# ================================
# AGENT CLASSES
# ================================

class Person(CellAgent):
    """Static agent representing someone to rescue.
    
    Attributes:
        age: Age of the person (affects urgency calculation)
        health: Health level 0.0-1.0 (affects urgency calculation)
        urgency: Calculated urgency level 0.0-1.0
        discovered_at_step: Step when person was discovered
    """

    def __init__(self, model, cell, age=None, health=None):
        super().__init__(model)
        self.cell = cell
        self.state = PersonState.MISSING
        self.age = age or self.random.randint(18, 80)
        self.health = health or self.random.uniform(0.1, 1.0)
        self.urgency = self._calculate_urgency()
        self.discovered_at_step = model.steps

    def _calculate_urgency(self) -> float:
        """Calculate urgency based on age and health (computer vision simulation)."""
        age_factor = 1.0 if self.age > 65 or self.age < 25 else 0.7
        health_factor = 1.0 - self.health
        return min(1.0, (age_factor + health_factor) / 2)

    def rescue(self) -> None:
        """Mark person as rescued."""
        self.state = PersonState.RESCUED

    @property
    def is_rescued(self) -> bool:
        """Check if person has been rescued."""
        return self.state == PersonState.RESCUED

    def step(self) -> None:
        """Person agents don't move or take actions."""
        pass

    @classmethod
    def create_agents(cls, model, count, cell, age=None, health=None):
        """Create multiple Person agents and add them to the model.
        
        Args:
            model: The model instance
            count: Number of persons to create
            cell: List of cells where persons should be placed
            age: List of ages for each person
            health: List of health values for each person
        """
        if age is None:
            age = [None] * count
        if health is None:
            health = [None] * count
        
        for i in range(count):
            person = cls(
                model=model,
                cell=cell[i],
                age=age[i],
                health=health[i]
            )
            model.agents.add(person)


class FirstAidRobot(CellAgent):
    """Ground-based rescue robot with terrain navigation and communication capabilities.
    
    Supports both Basic Mode (random exploration) and Extended/Novel Mode (coordinated missions).
    
    Attributes:
        battery: Current battery level
        battery_capacity: Maximum battery capacity
        target_person: Currently targeted person to rescue
        target_coordinate: Target coordinates for Extended Mode
        target_person_id: ID of assigned person
        mission_start_step: Step when current mission started
    """

    def __init__(self, model, cell, battery_capacity=100):
        super().__init__(model)
        self.cell = cell
        self.state = RobotState.IDLE
        self.battery = battery_capacity
        self.battery_capacity = battery_capacity
        self.target_person = None
        self.target_coordinate = None
        self.target_person_id = None
        self.steps_since_last_action = 0
        self.mission_start_step = None

    def _should_print(self) -> bool:
        """Check if verbose output should be printed."""
        return not getattr(self.model, 'quiet', False)

    # ================================
    # BASIC MOVEMENT METHODS
    # ================================

    def move_random(self) -> None:
        """Move to a random neighboring cell (Basic Mode)."""
        # Check if being charged - don't move during charging
        if hasattr(self, '_being_charged') and self._being_charged:
            if self._should_print():
                print(f"🔒 Robot {self.unique_id}: Cannot move - being charged")
            return
            
        if self.battery <= 0:
            print(f"🔋 Robot {self.unique_id} at {self.cell.coordinate}: Cannot move - battery depleted!")
            return

        old_pos = self.cell.coordinate
        neighbors = list(self.cell.neighborhood.cells)
        if neighbors:
            # Find a neighbor we can afford to move to
            affordable_neighbors = []
            for neighbor in neighbors:
                altitude_cost = self.model.get_altitude_cost(neighbor.coordinate)
                if self.battery >= altitude_cost:
                    affordable_neighbors.append((neighbor, altitude_cost))
            
            if affordable_neighbors:
                new_cell, altitude_cost = self.random.choice(affordable_neighbors)
                self.cell = new_cell
                self.battery -= altitude_cost
                altitude = self.model.get_altitude(new_cell.coordinate)
                if self._should_print():
                    print(
                        f"🚶 Robot {self.unique_id}: {old_pos} → {new_cell.coordinate} (Alt: {altitude}K MASL, Cost: {altitude_cost}, Battery: {self.battery})")
            else:
                if self._should_print():
                    print(f"🔋 Robot {self.unique_id} at {self.cell.coordinate}: Insufficient battery for any movement")
                # CRITICAL FIX: Request charging when unable to move due to battery
                if (self.model.mode == "novel" and 
                    not (hasattr(self, '_charging_requested') and self._charging_requested)):
                    self.request_charging()
        else:
            if self._should_print():
                print(f"🚫 Robot {self.unique_id} at {self.cell.coordinate}: No valid neighbors to move to")

    def move_towards_target(self) -> None:
        """Move one step towards target coordinate (Extended Mode)."""
        # Check if being charged - don't move during charging
        if hasattr(self, '_being_charged') and self._being_charged:
            if self._should_print():
                print(f"🔒 Robot {self.unique_id}: Cannot move - being charged")
            return
            
        if self.battery <= 0 or not self.target_coordinate:
            return

        current_x, current_y = self.cell.coordinate
        target_x, target_y = self.target_coordinate

        # Simple pathfinding towards target
        dx = 1 if target_x > current_x else (-1 if target_x < current_x else 0)
        dy = 1 if target_y > current_y else (-1 if target_y < current_y else 0)
        target_coord = (current_x + dx, current_y + dy)

        # Check bounds and find target cell
        if (0 <= target_coord[0] < self.model.width and
                0 <= target_coord[1] < self.model.height):

            target_cell = None
            for cell in self.model.grid.all_cells:
                if cell.coordinate == target_coord:
                    target_cell = cell
                    break

            if target_cell:
                altitude_cost = self.model.get_altitude_cost(target_coord)
                if self.battery >= altitude_cost:
                    old_coord = self.cell.coordinate
                    self.cell = target_cell
                    self.battery -= altitude_cost
                    distance = abs(target_x - target_coord[0]) + abs(target_y - target_coord[1])
                    if self._should_print():
                        print(f"🎯 Robot {self.unique_id}: Moving to target {old_coord} → {target_coord} "
                              f"(Distance remaining: {distance}, Battery: {self.battery})")
                else:
                    if self._should_print():
                        print(f"🔋 Robot {self.unique_id}: Low battery ({self.battery}), cannot reach target (cost: {altitude_cost})")
                    # CRITICAL FIX: Request charging when unable to move to target due to battery
                    if (self.model.mode == "novel" and 
                        not (hasattr(self, '_charging_requested') and self._charging_requested)):
                        self.request_charging()
                    # Don't decrease battery when we can't move

    def move_towards_base(self) -> None:
        """Move one step towards the base station."""
        # Check if being charged - don't move during charging
        if hasattr(self, '_being_charged') and self._being_charged:
            if self._should_print():
                print(f"🔒 Robot {self.unique_id}: Cannot move - being charged")
            return
            
        if self.battery <= 0:
            print(f"🔋 Robot {self.unique_id} at {self.cell.coordinate}: Cannot return - battery depleted!")
            return

        base_x, base_y = self.model.base_position
        curr_x, curr_y = self.cell.coordinate
        old_pos = self.cell.coordinate

        # Simple pathfinding towards base
        dx = 1 if base_x > curr_x else (-1 if base_x < curr_x else 0)
        dy = 1 if base_y > curr_y else (-1 if base_y < curr_y else 0)
        target_coord = (curr_x + dx, curr_y + dy)

        # Check if target coordinate is valid and within bounds
        if (0 <= target_coord[0] < self.model.width and
                0 <= target_coord[1] < self.model.height):

            target_cell = None
            for cell in self.model.grid.all_cells:
                if cell.coordinate == target_coord:
                    target_cell = cell
                    break

            if target_cell:
                move_cost = self.model.get_altitude_cost(target_coord)
                if self.battery >= move_cost:
                    self.cell = target_cell
                    self.battery -= move_cost
                    dist_to_base = abs(target_coord[0] - base_x) + abs(target_coord[1] - base_y)
                    if self._should_print():
                        print(
                            f"🏠 Robot {self.unique_id}: Returning {old_pos} → {target_coord} (Distance to base: {dist_to_base}, Battery: {self.battery})")
                else:
                    print(f"🔋 Robot {self.unique_id}: Insufficient battery ({self.battery}) for return movement (cost: {move_cost})")
                    # CRITICAL FIX: Request charging when unable to move due to battery
                    if (self.model.mode == "novel" and 
                        not (hasattr(self, '_charging_requested') and self._charging_requested)):
                        self.request_charging()
            else:
                print(f"🚫 Robot {self.unique_id} at {self.cell.coordinate}: Cannot find target cell, staying in place")
        else:
            print(f"🚫 Robot {self.unique_id} at {self.cell.coordinate}: Target out of bounds, staying in place")

    # ================================
    # RESCUE OPERATIONS
    # ================================

    def search_for_person(self) -> bool:
        """Look for people to rescue in current cell."""
        for agent in self.cell.agents:
            if isinstance(agent, Person) and not agent.is_rescued:
                self.target_person = agent
                if self._should_print():
                    print(
                        f"🔍 Robot {self.unique_id} at {self.cell.coordinate}: Found person! Age: {agent.age}, Health: {agent.health:.2f}, Urgency: {agent.urgency:.2f}")
                return True
        return False

    def deliver_aid(self) -> bool:
        """Deliver first aid to the target person."""
        if self.target_person and not self.target_person.is_rescued:
            # Mark person as rescued
            self.target_person.rescue()
            self.model.rescued_count += 1
            rescue_time = self.model.steps - self.target_person.discovered_at_step
            
            # Enhanced success tracking
            success_data = {
                "robot_id": self.unique_id,
                "person_id": self.target_person.unique_id,
                "rescue_time": rescue_time,
                "person_age": self.target_person.age,
                "person_health": self.target_person.health,
                "urgency": self.target_person.urgency,
                "rescue_location": list(self.cell.coordinate),
                "rescue_step": self.model.steps,
                "battery_remaining": self.battery
            }
            
            # Store rescue data for analysis
            if not hasattr(self.model, 'rescue_log'):
                self.model.rescue_log = []
            self.model.rescue_log.append(success_data)
            
            if self._should_print():
                print(f"💊 Robot {self.unique_id} at {self.cell.coordinate}: Successfully delivered aid! "
                      f"Person rescued in {rescue_time} steps. Total rescued: {self.model.rescued_count}")
                print(f"   📊 Rescue #{self.model.rescued_count}: Age {self.target_person.age}, "
                      f"Health {self.target_person.health:.2f}, Urgency {self.target_person.urgency:.2f}")
            return True
        else:
            print(
                f"❌ Robot {self.unique_id} at {self.cell.coordinate}: Cannot deliver aid - person already rescued or not found")
            return False

    # ================================
    # EXTENDED MODE COMMUNICATION
    # ================================

    def send_acknowledgment(self, drone_id: int) -> bool:
        """Send detailed acknowledgment message to drone."""
        if self.model.mode in ["extended", "novel"] and hasattr(self.model, 'message_system') and self.model.message_system:
            content = {
                "robot_id": self.unique_id,
                "status": "mission_accepted",
                "estimated_arrival": self._estimate_arrival_time(),
                "current_battery": self.battery,
                "current_position": list(self.cell.coordinate),
                "robot_state": self.state.value,
                "acknowledgment_time": self.model.steps
            }

            success = self.model.message_system.send_message(
                self.unique_id, drone_id, MessageType.MISSION_ACKNOWLEDGMENT,
                content, self.model.steps, priority=0.7
            )

            if success:
                eta = content["estimated_arrival"]
                print(f"📤 Robot {self.unique_id}: Sent acknowledgment to Drone {drone_id} "
                      f"(ETA: {eta} steps, Battery: {self.battery})")
            return success
        return False

    def _estimate_arrival_time(self) -> int:
        """Intelligent arrival time estimation"""
        if not self.target_coordinate:
            return 0

        current_x, current_y = self.cell.coordinate
        target_x, target_y = self.target_coordinate

        # Manhattan distance
        distance = abs(target_x - current_x) + abs(target_y - current_y)

        # Factor in terrain difficulty
        terrain_penalty = 0
        if distance > 0:
            # Sample a few points along the path to estimate terrain difficulty
            steps_x = (target_x - current_x) / max(1, distance)
            steps_y = (target_y - current_y) / max(1, distance)

            for i in range(1, min(distance + 1, 5)):  # Sample up to 5 points
                sample_x = int(current_x + steps_x * i)
                sample_y = int(current_y + steps_y * i)
                if 0 <= sample_x < self.model.width and 0 <= sample_y < self.model.height:
                    altitude = self.model.get_altitude((sample_x, sample_y))
                    terrain_penalty += altitude * 0.5  # Each altitude level adds 0.5 steps

        # Factor in current battery level (slower when battery is low)
        battery_factor = self.battery / self.battery_capacity
        battery_penalty = (1 - battery_factor) * distance * 0.2

        estimated_time = distance + terrain_penalty + battery_penalty + 2  # +2 buffer
        return max(1, int(estimated_time))

    def get_highest_priority_mission(self, messages) -> Optional:
        """Intelligent mission selection from multiple options."""
        location_messages = [msg for msg in messages
                             if hasattr(msg, 'message_type') and msg.message_type == MessageType.PERSON_LOCATION]

        if not location_messages:
            return None

        # Filter out missions for people who are already rescued
        valid_messages = []
        for msg in location_messages:
            person_id = msg.content.get("person_id")
            if person_id:
                # Find the person and check if they're still available
                person = None
                for agent in self.model.agents:
                    if isinstance(agent, Person) and agent.unique_id == person_id:
                        person = agent
                        break

                if person and not person.is_rescued:
                    valid_messages.append(msg)
                else:
                    print(f"ℹ️ Robot {self.unique_id}: Filtering out mission for Person {person_id} (already rescued)")

        if not valid_messages:
            print(f"ℹ️ Robot {self.unique_id}: No valid missions available (all persons already rescued)")
            return None

        if len(valid_messages) == 1:
            return valid_messages[0]

        # Multi-criteria mission selection from valid missions only
        print(f"🔍 Robot {self.unique_id}: Evaluating {len(valid_messages)} valid missions...")

        best_mission = None
        best_score = -1

        for msg in valid_messages:
            content = msg.content

            # Scoring logic
            urgency = content.get("urgency", 0.5)
            urgency_score = urgency * 40

            target_coord = tuple(content.get("coordinate", [0, 0]))
            distance = abs(self.cell.coordinate[0] - target_coord[0]) + \
                       abs(self.cell.coordinate[1] - target_coord[1])
            max_distance = self.model.width + self.model.height
            distance_score = (1 - distance / max_distance) * 25

            target_altitude = content.get("terrain_altitude", 0)
            current_altitude = self.model.get_altitude(self.cell.coordinate)
            altitude_compatibility = 1 - abs(target_altitude - current_altitude) / 3.0
            terrain_score = max(0, altitude_compatibility) * 15

            difficulty = content.get("estimated_rescue_difficulty", 0.5)
            difficulty_score = (1 - difficulty) * 10

            message_age = self.model.steps - content.get("timestamp", self.model.steps)
            age_penalty = min(message_age * 0.5, 5)
            age_score = 5 - age_penalty

            person_health = content.get("person_health", 0.5)
            health_score = (1 - person_health) * 5

            # Bonus for exclusive assignments (targeted messages)
            exclusive_bonus = 10 if content.get("exclusive", False) else 0

            total_score = (urgency_score + distance_score + terrain_score +
                           difficulty_score + age_score + health_score + exclusive_bonus)

            print(f"   📋 Mission {content.get('person_id', '?')}: Score={total_score:.1f} "
                  f"(Urg:{urgency:.2f}, Dist:{distance}, Excl:{content.get('exclusive', False)})")

            if total_score > best_score:
                best_score = total_score
                best_mission = msg

        if best_mission:
            person_id = best_mission.content.get("person_id", "Unknown")
            print(f"✅ Robot {self.unique_id}: Selected mission for Person {person_id} (Score: {best_score:.1f})")

        return best_mission

    def send_mission_complete(self):
        """Send mission completion notification"""
        if (self.model.mode in ["extended", "novel"] and
                hasattr(self.model, 'message_system') and
                self.model.message_system and
                self.target_person_id):

            completion_time = self.model.steps - (self.mission_start_step or self.model.steps)

            content = {
                "robot_id": self.unique_id,
                "person_id": self.target_person_id,
                "completion_time": completion_time,
                "final_battery": self.battery,
                "rescue_location": list(self.cell.coordinate),
                "mission_success": True,
                "completion_timestamp": self.model.steps
            }

            # Send to all drones (broadcast) so any monitoring drone knows
            success = self.model.message_system.send_message(
                self.unique_id, None, MessageType.MISSION_COMPLETE,
                content, self.model.steps, priority=0.8
            )

            if success:
                print(f"📤 Robot {self.unique_id}: Mission completion broadcast sent "
                      f"(Person {self.target_person_id}, Time: {completion_time} steps)")
            return success
        return False

    def request_charging(self) -> bool:
        """Request charging from MobileCharger when battery is low."""
        if not hasattr(self.model, 'message_system') or not self.model.message_system:
            return False

        # Don't send duplicate charging requests
        if hasattr(self, '_charging_requested') and self._charging_requested:
            return False

        # Find available mobile chargers
        chargers = [agent for agent in self.model.agents 
                   if isinstance(agent, MobileCharger)]
        
        if not chargers:
            return False

        # Send charging request to the first available charger
        charger = chargers[0]  # Simple selection, could be improved
        
        content = {
            "agent_id": self.unique_id,
            "agent_type": "FirstAidRobot",
            "coordinate": list(self.cell.coordinate),
            "battery_level": self.battery,
            "battery_capacity": self.battery_capacity,
            "urgency": 1.0 - (self.battery / self.battery_capacity),  # Higher urgency = lower battery
            "timestamp": self.model.steps
        }

        success = self.model.message_system.send_message(
            self.unique_id, charger.unique_id, MessageType.CHARGING_REQUEST,
            content, self.model.steps, priority=0.9
        )

        if success:
            self._charging_requested = True
            if self._should_print():
                print(f"🔋 Robot {self.unique_id}: Charging request sent (Battery: {self.battery}%)")
        
        return success

    def request_emergency_charging(self) -> bool:
        """Emergency charging request for completely depleted agents."""
        if not hasattr(self.model, 'message_system') or not self.model.message_system:
            return False

        # Find available mobile chargers
        chargers = [agent for agent in self.model.agents 
                   if isinstance(agent, MobileCharger)]
        
        if not chargers:
            return False

        # Send emergency request to ALL chargers for immediate response
        emergency_content = {
            "agent_id": self.unique_id,
            "agent_type": self.__class__.__name__,
            "coordinate": list(self.cell.coordinate),
            "battery_level": self.battery,
            "battery_capacity": self.battery_capacity,
            "urgency": 1.0,  # Maximum urgency
            "emergency": True,
            "timestamp": self.model.steps
        }

        success = False
        for charger in chargers:
            if self.model.message_system.send_message(
                self.unique_id, charger.unique_id, MessageType.CHARGING_REQUEST,
                emergency_content, self.model.steps, priority=1.0
            ):
                success = True

        if success and self._should_print():
            print(f"🚨 {self.__class__.__name__} {self.unique_id}: EMERGENCY charging request sent!")
        
        return success

    # ================================
    # MODE-SPECIFIC BEHAVIOR
    # ================================

    def step_basic_mode(self) -> None:
        """Basic Mode behavior (original Activity 1 logic)."""
        if self.battery <= 0 and self.state != RobotState.OUT_OF_BATTERY:
            print(f"⚡ Robot {self.unique_id}: BATTERY DEPLETED! Switching to OUT_OF_BATTERY state")
            self.state = RobotState.OUT_OF_BATTERY

        # State machine logic
        if self.state == RobotState.IDLE:
            print(f"🟦 Robot {self.unique_id}: IDLE → EXPLORING (Starting mission)")
            self.state = RobotState.EXPLORING

        elif self.state == RobotState.EXPLORING:
            print(f"🔍 Robot {self.unique_id}: EXPLORING at {self.cell.coordinate} (Battery: {self.battery})")
            
            # Check for higher priority messages occasionally during exploration
            if hasattr(self.model, 'message_system') and self.model.message_system and self.model.steps % 3 == 0:
                messages = self.model.message_system.get_messages(self.unique_id)
                urgent_mission = self.get_highest_priority_mission(messages)
                if urgent_mission and urgent_mission.content.get('urgency', 0) > 0.7:
                    print(f"🚨 Robot {self.unique_id}: EXPLORING → WAITING_FOR_MISSION (Urgent mission received!)")
                    self.state = RobotState.WAITING_FOR_MISSION
                    return
            
            self.move_random()
            if self.search_for_person():
                print(f"🎯 Robot {self.unique_id}: EXPLORING → AT_PERSON (Person found!)")
                self.state = RobotState.AT_PERSON
            elif self.battery <= 20:  # Return when battery low
                print(f"🔋 Robot {self.unique_id}: EXPLORING → RETURNING (Low battery: {self.battery})")
                self.state = RobotState.RETURNING

        elif self.state == RobotState.AT_PERSON:
            print(f"💊 Robot {self.unique_id}: AT_PERSON - Delivering aid...")
            if self.deliver_aid():
                print(f"✅ Robot {self.unique_id}: AT_PERSON → RETURNING (Aid delivered successfully)")
                self.state = RobotState.RETURNING
            else:
                # Person already rescued by someone else
                print(f"❌ Robot {self.unique_id}: AT_PERSON → EXPLORING (Person already rescued)")
                self.target_person = None
                self.state = RobotState.EXPLORING

        elif self.state == RobotState.RETURNING:
            if self.cell.coordinate == self.model.base_position:
                # Reached base - recharge and reset
                print(
                    f"🏠 Robot {self.unique_id}: RETURNING → IDLE (Reached base, recharging {self.battery} → {self.battery_capacity})")
                self.battery = self.battery_capacity
                self.state = RobotState.IDLE
                self.target_person = None
                self.steps_since_last_action = 0
            else:
                distance_to_base = abs(self.cell.coordinate[0] - self.model.base_position[0]) + abs(
                    self.cell.coordinate[1] - self.model.base_position[1])
                print(
                    f"🏠 Robot {self.unique_id}: RETURNING to base (Distance: {distance_to_base}, Battery: {self.battery})")
                self.move_towards_base()

        elif self.state == RobotState.OUT_OF_BATTERY:
            print(f"💀 Robot {self.unique_id}: OUT_OF_BATTERY at {self.cell.coordinate} - Waiting for rescue")
            # If robot is at base, it can recharge automatically
            if self.cell.coordinate == self.model.base_position:
                print(f"🔋 Robot {self.unique_id}: OUT_OF_BATTERY → IDLE (Emergency recharge at base)")
                self.battery = self.battery_capacity
                self.state = RobotState.IDLE
                self.target_person = None
                self.target_coordinate = None
                self.target_person_id = None
                # Reset charging request flag
                if hasattr(self, '_charging_requested'):
                    self._charging_requested = False
            else:
                # Continuously request emergency charging if in novel mode
                if self.model.mode == "novel":
                    # Send emergency request every few steps, but also on first OUT_OF_BATTERY step
                    if (self.model.steps % 5 == 0 or 
                        not hasattr(self, '_emergency_requested')):
                        success = self.request_emergency_charging()
                        if success:
                            self._emergency_requested = True

    def step_extended_mode(self) -> None:
        """Enhanced Extended Mode behavior with better coordination."""
        # Check for low battery and request charging in Novel mode
        # Use higher threshold and check multiple conditions for better charging request timing
        if (self.model.mode == "novel" and 
            self.state != RobotState.OUT_OF_BATTERY):
            
            # Calculate minimum battery needed for basic movement
            current_altitude_cost = self.model.get_altitude_cost(self.cell.coordinate)
            min_battery_needed = current_altitude_cost * 3  # Safety margin for 3 moves
            
            # Request charging if battery is critically low OR below 10% capacity
            # Also request if unable to perform basic actions due to low battery
            should_request_charging = (
                self.battery <= max(min_battery_needed, self.battery_capacity * 0.10) and
                not (hasattr(self, '_charging_requested') and self._charging_requested)
            )
            
            if should_request_charging:
                if self._should_print():
                    print(f"🔋 Robot {self.unique_id}: Low battery ({self.battery}/{self.battery_capacity}) - requesting charging")
                self.request_charging()
        
        # Handle complete battery depletion
        if self.battery <= 0 and self.state != RobotState.OUT_OF_BATTERY:
            print(f"⚡ Robot {self.unique_id}: BATTERY DEPLETED!")
            self.state = RobotState.OUT_OF_BATTERY
            # Emergency charging request for completely depleted battery
            if self.model.mode == "novel":
                self.request_emergency_charging()
            return

        if self.state == RobotState.IDLE:
            print(f"🟦 Robot {self.unique_id}: IDLE → WAITING_FOR_MISSION")
            self.state = RobotState.WAITING_FOR_MISSION

        elif self.state == RobotState.WAITING_FOR_MISSION:
            # Check for very low battery while waiting - this is critical
            if (self.model.mode == "novel" and 
                self.battery <= self.battery_capacity * 0.05 and 
                not (hasattr(self, '_charging_requested') and self._charging_requested)):
                if self._should_print():
                    print(f"🔋 Robot {self.unique_id}: Critical battery while waiting - requesting charging")
                self.request_charging()
            
            # Advanced message processing
            if hasattr(self.model, 'message_system') and self.model.message_system:
                messages = self.model.message_system.get_messages(self.unique_id)
                mission = self.get_highest_priority_mission(messages)

                if mission:
                    # Double-check the person is still available before accepting
                    person_id = mission.content["person_id"]
                    target_person = None
                    for agent in self.model.agents:
                        if isinstance(agent, Person) and agent.unique_id == person_id:
                            target_person = agent
                            break

                    if target_person and not target_person.is_rescued:
                        # Check if another robot is already en route to this person
                        other_robot_assigned = False
                        for robot in self.model.agents:
                            if (isinstance(robot, FirstAidRobot) and 
                                robot.unique_id != self.unique_id and
                                robot.target_person_id == person_id and
                                robot.state == RobotState.MOVING_TO_TARGET):
                                other_robot_assigned = True
                                print(f"ℹ️ Robot {self.unique_id}: Person {person_id} already assigned to Robot {robot.unique_id}")
                                break

                        if not other_robot_assigned:
                            # Accept the mission - handle both coordinate formats
                            if "coordinate" in mission.content:
                                self.target_coordinate = tuple(mission.content["coordinate"])
                            else:
                                # Handle x, y format from tests
                                self.target_coordinate = (mission.content["x"], mission.content["y"])
                            self.target_person_id = mission.content["person_id"]
                            self.mission_start_step = self.model.steps
                            drone_id = mission.content.get("drone_id")

                            # Send acknowledgment
                            if drone_id:
                                self.send_acknowledgment(drone_id)

                            # Clear all location messages (we've picked one)
                            self.model.message_system.clear_messages(self.unique_id, MessageType.PERSON_LOCATION)

                            self.state = RobotState.MOVING_TO_TARGET

                            urgency = mission.content.get("urgency", 0)
                            distance = abs(self.cell.coordinate[0] - self.target_coordinate[0]) + \
                                       abs(self.cell.coordinate[1] - self.target_coordinate[1])
                            print(f"🎯 Robot {self.unique_id}: Mission accepted - Person {self.target_person_id} "
                                  f"at {self.target_coordinate} (Urgency: {urgency:.2f}, Distance: {distance})")
                        else:
                            # Clear messages for this already-assigned person
                            self.model.message_system.clear_messages(self.unique_id, MessageType.PERSON_LOCATION)
                    else:
                        print(
                            f"ℹ️ Robot {self.unique_id}: Selected person {person_id} already rescued - clearing messages")
                        # Clear stale messages
                        self.model.message_system.clear_messages(self.unique_id, MessageType.PERSON_LOCATION)
                else:
                    # No missions available - stay at base and wait for coordinates
                    print(f"🏠 Robot {self.unique_id}: No missions - staying at base waiting for drone coordinates")
                    if self.battery > 0.5:
                        self.battery -= 0.5  # Minimal idle power consumption

        elif self.state == RobotState.MOVING_TO_TARGET:
            if self.cell.coordinate == self.target_coordinate:
                # Arrived at target location
                if self.search_for_person():
                    print(f"✅ Robot {self.unique_id}: MOVING_TO_TARGET → AT_PERSON (Target reached)")
                    self.state = RobotState.AT_PERSON
                else:
                    print(f"ℹ️ Robot {self.unique_id}: Target location reached but person not found")
                    print(f"   Person {self.target_person_id} was likely rescued by another robot")
                    self.target_coordinate = None
                    self.target_person_id = None
                    self.state = RobotState.WAITING_FOR_MISSION
            else:
                # Check if battery is getting low during movement
                if self.battery <= 15:
                    print(f"🔋 Robot {self.unique_id}: Low battery during mission - returning to base")
                    self.state = RobotState.RETURNING
                    # Clear mission to avoid conflicts
                    self.target_coordinate = None
                    self.target_person_id = None
                else:
                    distance_remaining = abs(self.target_coordinate[0] - self.cell.coordinate[0]) + \
                                         abs(self.target_coordinate[1] - self.cell.coordinate[1])
                    self.move_towards_target()

        # Rest of the method stays the same...
        elif self.state == RobotState.AT_PERSON:
            if self.deliver_aid():
                print(f"✅ Robot {self.unique_id}: AT_PERSON → RETURNING (Rescue successful)")
                self.send_mission_complete()
                self.state = RobotState.RETURNING
            else:
                print(f"ℹ️ Robot {self.unique_id}: Cannot deliver aid - returning to standby")
                self.target_person = None
                self.target_coordinate = None
                self.target_person_id = None
                self.state = RobotState.WAITING_FOR_MISSION

        elif self.state == RobotState.RETURNING:
            if self.cell.coordinate == self.model.base_position:
                mission_duration = self.model.steps - (self.mission_start_step or self.model.steps)
                print(f"🏠 Robot {self.unique_id}: RETURNING → IDLE (Mission complete in {mission_duration} steps)")
                self.battery = self.battery_capacity
                self.state = RobotState.IDLE
                self.target_person = None
                self.target_coordinate = None
                self.target_person_id = None
                self.mission_start_step = None
            else:
                distance_to_base = abs(self.cell.coordinate[0] - self.model.base_position[0]) + \
                                   abs(self.cell.coordinate[1] - self.model.base_position[1])
                self.move_towards_base()

        elif self.state == RobotState.OUT_OF_BATTERY:
            print(f"💀 Robot {self.unique_id}: OUT_OF_BATTERY - Awaiting recovery")
            pass

    def step(self) -> None:
        """Execute one step based on current operation mode."""
        self.steps_since_last_action += 1

        if self.model.mode == "basic":
            self.step_basic_mode()
        elif self.model.mode in ["extended", "novel"]:
            self.step_extended_mode()

    @classmethod
    def create_agents(cls, model, count, cell, battery_capacity=None):
        """Create multiple ExplorerDrone agents and add them to the model.
        
        Args:
            model: The model instance
            count: Number of drones to create
            cell: List of cells where drones should be placed
            battery_capacity: List of battery capacities for each drone
        """
        if battery_capacity is None:
            battery_capacity = [150] * count
        
        for i in range(count):
            drone = cls(
                model=model,
                cell=cell[i],
                battery_capacity=battery_capacity[i]
            )
            model.agents.add(drone)

    @classmethod
    def create_agents(cls, model, count, cell, battery_capacity=None):
        """Create multiple FirstAidRobot agents and add them to the model.
        
        Args:
            model: The model instance
            count: Number of robots to create
            cell: List of cells where robots should be placed
            battery_capacity: List of battery capacities for each robot
        """
        if battery_capacity is None:
            battery_capacity = [100] * count
        
        for i in range(count):
            robot = cls(
                model=model,
                cell=cell[i],
                battery_capacity=battery_capacity[i]
            )
            model.agents.add(robot)


class MobileCharger(CellAgent):
    """Mobile charging unit for recharging robots and drones.
    
    The MobileCharger stays at base unless it receives a charging request from
    a robot or drone with battery <= 5%. It moves to the agent's location,
    charges them, and returns to base to recharge itself.
    
    Attributes:
        battery: Current battery level
        battery_capacity: Maximum battery capacity (1000)
        target_agent: Agent currently being charged or moving to charge
        charging_rate: Battery units transferred per step
        self_charging_rate: Rate at which charger recharges itself at base
    """

    def __init__(self, model, cell, battery_capacity=1000):
        super().__init__(model)
        self.cell = cell
        self.state = MobileChargerState.IDLE_AT_BASE
        self.battery = battery_capacity
        self.battery_capacity = battery_capacity
        self.target_agent = None
        self.target_coordinate = None
        self.charging_rate = 20  # Battery units per step
        self.self_charging_rate = 75  # Self-charging rate at base
        self.steps_since_last_action = 0

    def _should_print(self) -> bool:
        """Check if verbose output should be printed."""
        return not getattr(self.model, 'quiet', False)

    # ================================
    # MOVEMENT METHODS
    # ================================

    def move_towards_target(self) -> None:
        """Move one step towards target coordinate."""
        if self.battery <= 0 or not self.target_coordinate:
            return

        current_x, current_y = self.cell.coordinate
        target_x, target_y = self.target_coordinate

        # Simple pathfinding towards target
        dx = 1 if target_x > current_x else (-1 if target_x < current_x else 0)
        dy = 1 if target_y > current_y else (-1 if target_y < current_y else 0)
        target_coord = (current_x + dx, current_y + dy)

        # Check bounds and find target cell
        if (0 <= target_coord[0] < self.model.width and
                0 <= target_coord[1] < self.model.height):

            target_cell = None
            for cell in self.model.grid.all_cells:
                if cell.coordinate == target_coord:
                    target_cell = cell
                    break

            if target_cell:
                # Mobile charger has lower movement cost due to efficient design
                move_cost = max(1, self.model.get_altitude_cost(target_coord) // 2)
                if self.battery >= move_cost:
                    old_coord = self.cell.coordinate
                    self.cell = target_cell
                    self.battery -= move_cost
                    distance = abs(target_x - target_coord[0]) + abs(target_y - target_coord[1])
                    if self._should_print():
                        print(f"🔌 Charger {self.unique_id}: Moving to charge {old_coord} → {target_coord} "
                              f"(Distance remaining: {distance}, Battery: {self.battery})")

    def move_towards_base(self) -> None:
        """Move one step towards the base station."""
        if self.battery <= 0:
            if self._should_print():
                print(f"🔋 Charger {self.unique_id}: Cannot return - battery depleted!")
            return

        base_x, base_y = self.model.base_position
        curr_x, curr_y = self.cell.coordinate

        # Simple pathfinding towards base
        dx = 1 if base_x > curr_x else (-1 if base_x < curr_x else 0)
        dy = 1 if base_y > curr_y else (-1 if base_y < curr_y else 0)
        target_coord = (curr_x + dx, curr_y + dy)

        # Check if target coordinate is valid and within bounds
        if (0 <= target_coord[0] < self.model.width and
                0 <= target_coord[1] < self.model.height):

            target_cell = None
            for cell in self.model.grid.all_cells:
                if cell.coordinate == target_coord:
                    target_cell = cell
                    break

            if target_cell:
                move_cost = max(1, self.model.get_altitude_cost(target_coord) // 2)
                if self.battery >= move_cost:
                    old_coord = self.cell.coordinate
                    self.cell = target_cell
                    self.battery -= move_cost
                    dist_to_base = abs(target_coord[0] - base_x) + abs(target_coord[1] - base_y)
                    if self._should_print():
                        print(f"🏠 Charger {self.unique_id}: Returning {old_coord} → {target_coord} "
                              f"(Distance to base: {dist_to_base}, Battery: {self.battery})")

    # ================================
    # CHARGING OPERATIONS
    # ================================

    def charge_agent(self) -> bool:
        """Charge the target agent at current location."""
        if not self.target_agent:
            return False

        # Find the target agent at this location
        agent_to_charge = None
        for agent in self.cell.agents:
            if (isinstance(agent, (FirstAidRobot, ExplorerDrone)) and 
                agent.unique_id == self.target_agent.unique_id):
                agent_to_charge = agent
                break

        if not agent_to_charge:
            if self._should_print():
                print(f"❌ Charger {self.unique_id}: Target agent not found at location")
            return False

        # Transfer battery
        charge_amount = min(self.charging_rate, 
                           self.battery, 
                           agent_to_charge.battery_capacity - agent_to_charge.battery)
        
        if charge_amount > 0:
            agent_to_charge.battery += charge_amount
            self.battery -= charge_amount
            
            if self._should_print():
                print(f"⚡ Charger {self.unique_id}: Charging {agent_to_charge.__class__.__name__} {agent_to_charge.unique_id} "
                      f"(+{charge_amount} → {agent_to_charge.battery}/{agent_to_charge.battery_capacity})")
            
            # Check if agent is fully charged or significantly improved
            if agent_to_charge.battery >= agent_to_charge.battery_capacity:
                # Reset charging request flag so agent can request again if needed
                if hasattr(agent_to_charge, '_charging_requested'):
                    agent_to_charge._charging_requested = False
                if self._should_print():
                    print(f"✅ Charger {self.unique_id}: Agent {agent_to_charge.unique_id} fully charged!")
                return True
            # Also reset flag if battery is now sufficient for basic operations
            elif (agent_to_charge.battery > agent_to_charge.battery_capacity * 0.2 and
                  hasattr(agent_to_charge, '_charging_requested')):
                agent_to_charge._charging_requested = False
        
        return False

    def recharge_self(self) -> None:
        """Recharge self at base station."""
        if self.cell.coordinate == self.model.base_position:
            charge_amount = min(self.self_charging_rate, 
                               self.battery_capacity - self.battery)
            if charge_amount > 0:
                self.battery += charge_amount
                if self._should_print():
                    print(f"🔋 Charger {self.unique_id}: Self-charging at base "
                          f"(+{charge_amount} → {self.battery}/{self.battery_capacity})")

    @classmethod
    def create_agents(cls, model, count, cell, battery_capacity=None):
        """Create multiple MobileCharger agents and add them to the model.
        
        Args:
            model: The model instance
            count: Number of chargers to create
            cell: List of cells where chargers should be placed
            battery_capacity: List of battery capacities for each charger
        """
        if battery_capacity is None:
            battery_capacity = [1000] * count
        
        for i in range(count):
            charger = cls(
                model=model,
                cell=cell[i],
                battery_capacity=battery_capacity[i]
            )
            model.agents.add(charger)

    # ================================
    # MESSAGE HANDLING
    # ================================

    def check_for_charging_requests(self) -> bool:
        """Check for charging requests from robots and drones."""
        if not hasattr(self.model, 'message_system') or not self.model.message_system:
            if self._should_print():
                print(f"❌ Charger {self.unique_id}: No message system available")
            return False

        messages = self.model.message_system.get_messages(self.unique_id)
        charging_requests = [msg for msg in messages 
                           if hasattr(msg, 'message_type') and msg.message_type == MessageType.CHARGING_REQUEST]

        if not charging_requests:
            return False

        # Find the most urgent charging request
        best_request = None
        highest_priority = -1

        for msg in charging_requests:
            content = msg.content
            battery_level = content.get("battery_level", 100)
            urgency = content.get("urgency", 0.5)
            is_emergency = content.get("emergency", False)
            
            # Priority calculation: emergency requests get maximum priority
            if is_emergency:
                priority = 10.0 + urgency  # Emergency always wins
            else:
                # Normal priority: lower battery = higher priority
                priority = (1.0 - battery_level / 100.0) + urgency
            
            if priority > highest_priority:
                highest_priority = priority
                best_request = msg

        if best_request:
            # Find the requesting agent
            requester_id = best_request.content.get("agent_id")
            requester_coordinate = tuple(best_request.content.get("coordinate", [0, 0]))
            
            # Find the actual agent object
            for agent in self.model.agents:
                if (isinstance(agent, (FirstAidRobot, ExplorerDrone)) and 
                    agent.unique_id == requester_id):
                    self.target_agent = agent
                    self.target_coordinate = requester_coordinate
                    
                    if self._should_print():
                        print(f"🔌 Charger {self.unique_id}: Accepted charging request from "
                              f"{agent.__class__.__name__} {agent.unique_id} at {requester_coordinate}")
                    
                    # Clear charging request messages
                    self.model.message_system.clear_messages(self.unique_id, MessageType.CHARGING_REQUEST)
                    return True

        return False

    # ================================
    # STATE MACHINE BEHAVIOR
    # ================================

    def step(self) -> None:
        """Execute one step based on current state."""
        if self.model.mode != "novel":
            # MobileCharger only operates in novel mode
            if self._should_print():
                print(f"🚫 Charger {self.unique_id}: Not in novel mode ({self.model.mode})")
            return

        self.steps_since_last_action += 1

        if self.state == MobileChargerState.IDLE_AT_BASE:
            # Stay at base and check for charging requests
            if self.cell.coordinate != self.model.base_position:
                # Move to base if not there
                if self._should_print():
                    print(f"🏠 Charger {self.unique_id}: Not at base, moving from {self.cell.coordinate} to {self.model.base_position}")
                self.move_towards_base()
                return

            # Check if we need to recharge ourselves (only if significantly low)
            if self.battery < self.battery_capacity * 0.9:  # Only self-charge if below 90%
                if self._should_print():
                    print(f"🔋 Charger {self.unique_id}: IDLE_AT_BASE → CHARGING_SELF (Battery: {self.battery}/{self.battery_capacity})")
                self.state = MobileChargerState.CHARGING_SELF
                return

            # Check for charging requests
            if self.check_for_charging_requests():
                if self._should_print():
                    print(f"🚨 Charger {self.unique_id}: IDLE_AT_BASE → MOVING_TO_AGENT")
                self.state = MobileChargerState.MOVING_TO_AGENT

        elif self.state == MobileChargerState.MOVING_TO_AGENT:
            if not self.target_agent or not self.target_coordinate:
                if self._should_print():
                    print(f"❌ Charger {self.unique_id}: No target - returning to base")
                self.state = MobileChargerState.RETURNING_TO_BASE
                return

            # Check if target agent still needs charging
            if self.target_agent.battery > self.target_agent.battery_capacity * 0.05:
                if self._should_print():
                    print(f"ℹ️ Charger {self.unique_id}: Target agent no longer needs charging - returning")
                self.target_agent = None
                self.target_coordinate = None
                self.state = MobileChargerState.RETURNING_TO_BASE
                return

            # Move towards target
            if self.cell.coordinate == self.target_coordinate:
                if self._should_print():
                    print(f"✅ Charger {self.unique_id}: MOVING_TO_AGENT → CHARGING_AGENT")
                self.state = MobileChargerState.CHARGING_AGENT
            else:
                self.move_towards_target()

        elif self.state == MobileChargerState.CHARGING_AGENT:
            if not self.target_agent:
                self.state = MobileChargerState.RETURNING_TO_BASE
                return

            # Check if target agent moved away
            agent_at_location = any(agent.unique_id == self.target_agent.unique_id 
                                  for agent in self.cell.agents 
                                  if isinstance(agent, (FirstAidRobot, ExplorerDrone)))
            
            if not agent_at_location:
                if self._should_print():
                    print(f"⚠️ Charger {self.unique_id}: Target agent moved away - stopping charging")
                self.target_agent = None
                self.target_coordinate = None
                self.state = MobileChargerState.RETURNING_TO_BASE
                return

            # Check if agent is fully charged or we're out of battery
            if (self.target_agent.battery >= self.target_agent.battery_capacity or 
                self.battery <= 10):  # Keep some battery for return trip
                
                # Mark agent as fully charged
                if hasattr(self.target_agent, '_being_charged'):
                    self.target_agent._being_charged = False
                    
                if self._should_print():
                    print(f"✅ Charger {self.unique_id}: CHARGING_AGENT → RETURNING_TO_BASE (Agent fully charged: {self.target_agent.battery}/{self.target_agent.battery_capacity})")
                
                self.target_agent = None
                self.target_coordinate = None
                self.state = MobileChargerState.RETURNING_TO_BASE
            else:
                # Mark agent as being charged to prevent movement
                if not hasattr(self.target_agent, '_being_charged'):
                    self.target_agent._being_charged = True
                    if self._should_print():
                        print(f"🔒 Charger {self.unique_id}: Locking agent in place for charging")
                
                # Continue charging
                self.charge_agent()

        elif self.state == MobileChargerState.RETURNING_TO_BASE:
            if self.cell.coordinate == self.model.base_position:
                if self._should_print():
                    print(f"🏠 Charger {self.unique_id}: RETURNING_TO_BASE → IDLE_AT_BASE")
                self.state = MobileChargerState.IDLE_AT_BASE
            else:
                self.move_towards_base()

        elif self.state == MobileChargerState.CHARGING_SELF:
            if self.battery >= self.battery_capacity:
                if self._should_print():
                    print(f"🔋 Charger {self.unique_id}: CHARGING_SELF → IDLE_AT_BASE (Fully charged)")
                self.state = MobileChargerState.IDLE_AT_BASE
            else:
                self.recharge_self()


class ExplorerDrone(CellAgent):
    """Aerial drone for reconnaissance and coordination.
    
    Supports both Basic Mode (autonomous search) and Extended/Novel Mode (coordinated missions).
    
    Attributes:
        battery: Current battery level
        battery_capacity: Maximum battery capacity
        target_person: Currently targeted person
        coordination_timer: Timer for coordination phase
        max_coordination_time: Maximum coordination wait time
    """

    def __init__(self, model, cell, battery_capacity=150):
        super().__init__(model)
        self.cell = cell
        self.state = DroneState.IDLE
        self.battery = battery_capacity
        self.battery_capacity = battery_capacity
        self.target_person = None
        self.steps_since_last_action = 0
        self.flight_directions = ['up', 'down', 'north', 'south', 'east', 'west']
        self.coordination_timer = 0
        self.max_coordination_time = 5

    def _should_print(self) -> bool:
        """Check if verbose output should be printed."""
        return not getattr(self.model, 'quiet', False)

    # ================================
    # BASIC MOVEMENT METHODS
    # ================================

    def move_3d(self) -> None:
        """Move in 6 directions (front, back, left, right, up, down)."""
        # Check if being charged - don't move during charging
        if hasattr(self, '_being_charged') and self._being_charged:
            if self._should_print():
                print(f"🔒 Drone {self.unique_id}: Cannot move - being charged")
            return
            
        if self.battery <= 0:
            print(f"🔋 Drone {self.unique_id} at {self.cell.coordinate}: Cannot fly - battery depleted!")
            return

        old_pos = self.cell.coordinate
        neighbors = list(self.cell.neighborhood.cells)
        if neighbors and self.battery >= 2:
            new_cell = self.random.choice(neighbors)
            self.cell = new_cell
            self.battery -= 2  # Drones consume more battery for flight
            altitude = self.model.get_altitude(new_cell.coordinate)
            if self._should_print():
                print(
                    f"✈️ Drone {self.unique_id}: Flying {old_pos} → {new_cell.coordinate} (Alt: {altitude}K MASL, Battery: {self.battery})")
        elif neighbors and self.battery < 2:
            if self._should_print():
                print(f"🔋 Drone {self.unique_id} at {self.cell.coordinate}: Insufficient battery for flight (battery: {self.battery})")
        else:
            if self._should_print():
                print(f"🚫 Drone {self.unique_id} at {self.cell.coordinate}: No valid flight paths available")

    def move_towards_base(self) -> None:
        """Move one step towards the base station."""
        # Check if being charged - don't move during charging
        if hasattr(self, '_being_charged') and self._being_charged:
            if self._should_print():
                print(f"🔒 Drone {self.unique_id}: Cannot move - being charged")
            return
            
        if self.battery <= 0:
            return

        base_x, base_y = self.model.base_position
        curr_x, curr_y = self.cell.coordinate

        # Simple pathfinding towards base
        dx = 1 if base_x > curr_x else (-1 if base_x < curr_x else 0)
        dy = 1 if base_y > curr_y else (-1 if base_y < curr_y else 0)
        target_coord = (curr_x + dx, curr_y + dy)

        # Check if target coordinate is valid and within bounds
        if (0 <= target_coord[0] < self.model.width and
                0 <= target_coord[1] < self.model.height):

            target_cell = None
            for cell in self.model.grid.all_cells:
                if cell.coordinate == target_coord:
                    target_cell = cell
                    break

            if target_cell and self.battery >= 2:
                self.cell = target_cell
                self.battery -= 2  # Drone flight cost
            elif self.battery < 2:
                print(f"🔋 Drone {self.unique_id}: Insufficient battery for base return (battery: {self.battery})")
                # CRITICAL FIX: Request charging when unable to move due to battery
                if (self.model.mode == "novel" and 
                    not (hasattr(self, '_charging_requested') and self._charging_requested)):
                    self.request_charging()
        elif self.battery < 2:
            print(f"🔋 Drone {self.unique_id}: Insufficient battery for movement (battery: {self.battery})")
            # CRITICAL FIX: Request charging when unable to move due to battery
            if (self.model.mode == "novel" and 
                not (hasattr(self, '_charging_requested') and self._charging_requested)):
                self.request_charging()

    # ================================
    # RECONNAISSANCE OPERATIONS
    # ================================

    def search_for_person(self) -> bool:
        """Use computer vision to locate and assess people."""
        for agent in self.cell.agents:
            if isinstance(agent, Person) and not agent.is_rescued:
                self.target_person = agent
                if self._should_print():
                    print(
                        f"📸 Drone {self.unique_id} at {self.cell.coordinate}: Person detected! Age: {agent.age}, Health: {agent.health:.2f}, Urgency: {agent.urgency:.2f}")
                return True
        return False

    def assess_urgency(self) -> float:
        """Use computer vision to calculate urgency."""
        if self.target_person:
            urgency = self.target_person.urgency
            if self._should_print():
                print(f"🧠 Drone {self.unique_id}: AI assessment complete - Urgency level: {urgency:.2f}")
            return urgency
        return 0

    # ================================
    # EXTENDED MODE COMMUNICATION
    # ================================

    def should_broadcast(self) -> bool:
        """Decide whether to broadcast or send targeted message."""
        if not self.target_person:
            return False

        urgency = self.target_person.urgency
        # Include robots in more states for better availability
        available_robots = len([r for r in self.model.agents
                                if isinstance(r, FirstAidRobot) and
                                r.state in [RobotState.IDLE, RobotState.WAITING_FOR_MISSION, RobotState.EXPLORING] and
                                r.battery > 20])  # Lowered battery threshold

        # Communication Strategy Decision Tree:
        # 1. CRITICAL urgency (>0.8) = Always broadcast for speed
        # 2. Few robots available (≤2) = Broadcast to ensure response
        # 3. Many robots available (≥6) = Broadcast for load balancing
        # 4. Otherwise = Targeted selection for efficiency

        if urgency > 0.8:
            if self._should_print():
                print(f"📢 Drone {self.unique_id}: CRITICAL urgency ({urgency:.2f}) - Broadcasting to all robots")
            return True
        elif available_robots <= 2:
            if self._should_print():
                print(f"📢 Drone {self.unique_id}: Few robots available ({available_robots}) - Broadcasting")
            return True
        elif available_robots >= 6:
            if self._should_print():
                print(
                    f"📢 Drone {self.unique_id}: Many robots available ({available_robots}) - Broadcasting for load balancing")
            return True
        else:
            if self._should_print():
                print(f"🎯 Drone {self.unique_id}: Normal conditions - Using targeted selection")
            return False

    def select_best_robot(self):
        """Intelligent robot selection using multi-criteria decision making"""
        if not self.target_person:
            return None

        # Include robots in more states for better availability
        available_robots = [r for r in self.model.agents
                            if isinstance(r, FirstAidRobot) and
                            r.state in [RobotState.IDLE, RobotState.WAITING_FOR_MISSION, RobotState.EXPLORING] and
                            r.battery > 20]  # Lowered battery threshold

        if not available_robots:
            print(f"❌ Drone {self.unique_id}: No available robots for mission")
            return None

        target_coord = self.target_person.cell.coordinate
        best_robot = None
        best_score = float('inf')

        if self._should_print():
            print(f"🔍 Drone {self.unique_id}: Evaluating {len(available_robots)} available robots...")

        for robot in available_robots:
            # Multi-criteria scoring system

            # 1. Distance Factor (Manhattan distance)
            distance = (abs(robot.cell.coordinate[0] - target_coord[0]) +
                        abs(robot.cell.coordinate[1] - target_coord[1]))
            distance_score = distance * 2  # Weight: High importance

            # 2. Battery Factor (0.0 = empty, 1.0 = full)
            battery_factor = robot.battery / robot.battery_capacity
            battery_score = (1 - battery_factor) * 15  # Weight: High importance

            # 3. Workload Factor (pending messages)
            pending_messages = len(
                self.model.message_system.get_messages(robot.unique_id)) if self.model.message_system else 0
            workload_score = pending_messages * 8  # Weight: Medium importance

            # 4. Terrain Factor (consider altitude at robot's location)
            robot_altitude = self.model.get_altitude(robot.cell.coordinate)
            target_altitude = self.model.get_altitude(target_coord)
            altitude_penalty = abs(target_altitude - robot_altitude) * 3  # Weight: Medium importance

            # 5. State Preference (IDLE robots are preferred over WAITING_FOR_MISSION, EXPLORING robots get small penalty)
            if robot.state == RobotState.IDLE:
                state_bonus = 0
            elif robot.state == RobotState.WAITING_FOR_MISSION:
                state_bonus = 5
            elif robot.state == RobotState.EXPLORING:
                state_bonus = 10  # Small penalty for robots already exploring
            else:
                state_bonus = 20  # High penalty for busy robots

            # Calculate total score (lower is better)
            total_score = distance_score + battery_score + workload_score + altitude_penalty + state_bonus

            print(f"   🤖 Robot {robot.unique_id}: Score={total_score:.1f} "
                  f"(Dist:{distance}, Bat:{battery_factor:.2f}, Load:{pending_messages}, Alt:{robot_altitude}→{target_altitude})")

            if total_score < best_score:
                best_score = total_score
                best_robot = robot

        if best_robot:
            if self._should_print():
                print(f"✅ Drone {self.unique_id}: Selected Robot {best_robot.unique_id} (Score: {best_score:.1f})")

        return best_robot

    def send_location_message(self):
        """Send person location to robots with intelligent routing."""
        if not self.target_person or self.model.mode not in ["extended", "novel"]:
            return False

        if not hasattr(self.model, 'message_system') or not self.model.message_system:
            print(f"❌ Drone {self.unique_id}: Message system not available")
            return False

        # Don't send if person is already rescued
        if self.target_person.is_rescued:
            print(
                f"ℹ️ Drone {self.unique_id}: Person {self.target_person.unique_id} already rescued - skipping message")
            return False

        # Prevent duplicate processing across all drones
        person_id = self.target_person.unique_id
        
        # Check if any other drone has already processed this person recently
        for drone in self.model.agents:
            if (isinstance(drone, ExplorerDrone) and 
                drone.unique_id != self.unique_id and
                hasattr(drone, '_last_message_person_id') and
                drone._last_message_person_id == person_id):
                print(f"ℹ️ Drone {self.unique_id}: Person {person_id} already being handled by Drone {drone.unique_id} - skipping")
                return False

        # Don't send duplicate messages for the same person from this drone
        if hasattr(self, '_last_message_person_id') and self._last_message_person_id == person_id:
            print(
                f"ℹ️ Drone {self.unique_id}: Already sent message for Person {person_id} - skipping duplicate")
            return False

        # Prepare comprehensive location data
        location_data = {
            "coordinate": list(self.target_person.cell.coordinate),
            "urgency": self.target_person.urgency,
            "person_id": self.target_person.unique_id,
            "drone_id": self.unique_id,
            "person_age": self.target_person.age,
            "person_health": self.target_person.health,
            "terrain_altitude": self.model.get_altitude(self.target_person.cell.coordinate),
            "estimated_rescue_difficulty": self._estimate_rescue_difficulty(),
            "timestamp": self.model.steps,
            "mission_type": "rescue_coordination",
            "exclusive": not self.should_broadcast()  # Mark if this should be exclusive assignment
        }

        success = False

        if self.should_broadcast():
            # BROADCAST STRATEGY: Send to all available robots
            success = self.model.message_system.send_message(
                self.unique_id, None, MessageType.PERSON_LOCATION,
                location_data, self.model.steps, priority=self.target_person.urgency
            )

            if success:
                available_count = len([r for r in self.model.agents
                                       if isinstance(r, FirstAidRobot) and
                                       r.state in [RobotState.IDLE, RobotState.WAITING_FOR_MISSION, RobotState.EXPLORING]])
                print(f"📢 Drone {self.unique_id}: BROADCAST sent to {available_count} robots "
                      f"for Person {self.target_person.unique_id} at {location_data['coordinate']} "
                      f"(Urgency: {location_data['urgency']:.2f})")

                # Mark that we sent a broadcast for this person
                self._last_message_person_id = self.target_person.unique_id
                self._message_type = "broadcast"

        else:
            # TARGETED STRATEGY: Send to best selected robot
            best_robot = self.select_best_robot()
            if best_robot:
                success = self.model.message_system.send_message(
                    self.unique_id, best_robot.unique_id, MessageType.PERSON_LOCATION,
                    location_data, self.model.steps, priority=self.target_person.urgency
                )

                if success:
                    distance = abs(best_robot.cell.coordinate[0] - location_data['coordinate'][0]) + \
                               abs(best_robot.cell.coordinate[1] - location_data['coordinate'][1])
                    print(f"🎯 Drone {self.unique_id}: TARGETED message sent to Robot {best_robot.unique_id} "
                          f"for Person {self.target_person.unique_id} at {location_data['coordinate']} "
                          f"(Distance: {distance}, Urgency: {location_data['urgency']:.2f})")

                    # Mark that we sent a targeted message for this person
                    self._last_message_person_id = self.target_person.unique_id
                    self._message_type = "targeted"
                    self._assigned_robot = best_robot.unique_id
            else:
                print(f"❌ Drone {self.unique_id}: No suitable robot found for targeted messaging")
                return False

        return success

    def _estimate_rescue_difficulty(self) -> float:
        """Estimate rescue difficulty based on multiple factors"""
        if not self.target_person:
            return 1.0

        # Factor in location characteristics
        coord = self.target_person.cell.coordinate
        altitude = self.model.get_altitude(coord)
        distance_from_base = (abs(coord[0] - self.model.base_position[0]) +
                              abs(coord[1] - self.model.base_position[1]))

        # Normalize factors (0.0 = easy, 1.0 = very difficult)
        altitude_factor = min(1.0, altitude / 3.0)  # 3K MASL = max difficulty
        distance_factor = min(1.0, distance_from_base / (self.model.width + self.model.height))
        urgency_factor = 1.0 - self.target_person.urgency  # Lower urgency = more time = easier
        health_factor = 1.0 - self.target_person.health  # Lower health = more difficult

        # Weighted combination
        difficulty = (altitude_factor * 0.3 +
                      distance_factor * 0.2 +
                      urgency_factor * 0.2 +
                      health_factor * 0.3)

        return min(1.0, difficulty)

    def received_acknowledgment(self) -> bool:
        """Check if any robot acknowledged the mission"""
        if not hasattr(self.model, 'message_system') or not self.model.message_system:
            return False

        messages = self.model.message_system.get_messages(self.unique_id, MessageType.MISSION_ACKNOWLEDGMENT)

        if messages:
            for msg in messages:
                robot_id = msg.content.get("robot_id", "Unknown")
                status = msg.content.get("status", "Unknown")
                eta = msg.content.get("estimated_arrival", "Unknown")
                print(f"✅ Drone {self.unique_id}: Received acknowledgment from Robot {robot_id} "
                      f"(Status: {status}, ETA: {eta} steps)")
            return True

        return False

    def request_charging(self) -> bool:
        """Request charging from MobileCharger when battery is low."""
        if not hasattr(self.model, 'message_system') or not self.model.message_system:
            return False

        # Don't send duplicate charging requests
        if hasattr(self, '_charging_requested') and self._charging_requested:
            return False

        # Find available mobile chargers
        chargers = [agent for agent in self.model.agents 
                   if isinstance(agent, MobileCharger)]
        
        if not chargers:
            return False

        # Send charging request to the first available charger
        charger = chargers[0]  # Simple selection, could be improved
        
        content = {
            "agent_id": self.unique_id,
            "agent_type": "ExplorerDrone",
            "coordinate": list(self.cell.coordinate),
            "battery_level": self.battery,
            "battery_capacity": self.battery_capacity,
            "urgency": 1.0 - (self.battery / self.battery_capacity),  # Higher urgency = lower battery
            "timestamp": self.model.steps
        }

        success = self.model.message_system.send_message(
            self.unique_id, charger.unique_id, MessageType.CHARGING_REQUEST,
            content, self.model.steps, priority=0.9
        )

        if success:
            self._charging_requested = True
            if self._should_print():
                print(f"🔋 Drone {self.unique_id}: Charging request sent (Battery: {self.battery}%)")
        
        return success

    def request_emergency_charging(self) -> bool:
        """Emergency charging request for completely depleted drones."""
        if not hasattr(self.model, 'message_system') or not self.model.message_system:
            return False

        # Find available mobile chargers
        chargers = [agent for agent in self.model.agents 
                   if isinstance(agent, MobileCharger)]
        
        if not chargers:
            return False

        # Send emergency request to ALL chargers for immediate response
        emergency_content = {
            "agent_id": self.unique_id,
            "agent_type": self.__class__.__name__,
            "coordinate": list(self.cell.coordinate),
            "battery_level": self.battery,
            "battery_capacity": self.battery_capacity,
            "urgency": 1.0,  # Maximum urgency
            "emergency": True,
            "timestamp": self.model.steps
        }

        success = False
        for charger in chargers:
            if self.model.message_system.send_message(
                self.unique_id, charger.unique_id, MessageType.CHARGING_REQUEST,
                emergency_content, self.model.steps, priority=1.0
            ):
                success = True

        if success and self._should_print():
            print(f"🚨 {self.__class__.__name__} {self.unique_id}: EMERGENCY charging request sent!")
        
        return success

    # ================================
    # MODE-SPECIFIC BEHAVIOR
    # ================================

    def step_basic_mode(self) -> None:
        """Basic Mode behavior (original Activity 1 logic)."""
        if self.battery <= 0 and self.state != DroneState.OUT_OF_BATTERY:
            print(f"⚡ Drone {self.unique_id}: BATTERY DEPLETED! Emergency landing - OUT_OF_BATTERY state")
            self.state = DroneState.OUT_OF_BATTERY

        # State machine logic
        if self.state == DroneState.IDLE:
            print(f"🟢 Drone {self.unique_id}: IDLE → FLYING (Launching mission)")
            self.state = DroneState.FLYING

        elif self.state == DroneState.FLYING:
            print(
                f"✈️ Drone {self.unique_id}: FLYING reconnaissance at {self.cell.coordinate} (Battery: {self.battery})")
            self.move_3d()
            if self.search_for_person():
                urgency = self.assess_urgency()
                print(f"📡 Drone {self.unique_id}: FLYING → WAITING (Person located, urgency: {urgency:.2f})")
                self.state = DroneState.WAITING
            elif self.battery <= 100:  # Return when battery low
                print(f"🔋 Drone {self.unique_id}: FLYING → RETURNING (Low battery: {self.battery})")
                self.state = DroneState.RETURNING

        elif self.state == DroneState.WAITING:
            # Wait for ground robots to arrive and rescue the person
            if self.target_person and self.target_person.is_rescued:
                print(f"✅ Drone {self.unique_id}: WAITING → RETURNING (Person rescued by ground team)")
                self.state = DroneState.RETURNING
            else:
                print(
                    f"⏳ Drone {self.unique_id}: WAITING for ground team at {self.cell.coordinate} (Battery: {self.battery})")
            # Stay in place but still consume battery
            if self.battery > 0:
                self.battery -= 1

        elif self.state == DroneState.RETURNING:
            if self.cell.coordinate == self.model.base_position:
                # Reached base - recharge and reset
                print(
                    f"🏠 Drone {self.unique_id}: RETURNING → IDLE (Landed at base, recharging {self.battery} → {self.battery_capacity})")
                self.battery = self.battery_capacity
                self.state = DroneState.IDLE
                self.target_person = None
                self.steps_since_last_action = 0
            else:
                distance_to_base = abs(self.cell.coordinate[0] - self.model.base_position[0]) + abs(
                    self.cell.coordinate[1] - self.model.base_position[1])
                print(
                    f"🏠 Drone {self.unique_id}: RETURNING to base (Distance: {distance_to_base}, Battery: {self.battery})")
                self.move_towards_base()

        elif self.state == DroneState.OUT_OF_BATTERY:
            print(f"💀 Drone {self.unique_id}: OUT_OF_BATTERY - Emergency landed at {self.cell.coordinate}")
            # If drone is at base, it can recharge automatically
            if self.cell.coordinate == self.model.base_position:
                print(f"🔋 Drone {self.unique_id}: OUT_OF_BATTERY → IDLE (Emergency recharge at base)")
                self.battery = self.battery_capacity
                self.state = DroneState.IDLE
                self.target_person = None
                self.coordination_timer = 0
                self._last_message_person_id = None
                self._message_type = None
                self._assigned_robot = None
                # Reset charging request flag
                if hasattr(self, '_charging_requested'):
                    self._charging_requested = False
            else:
                # Continuously request emergency charging if in novel mode
                if self.model.mode == "novel":
                    # Send emergency request every few steps, but also on first OUT_OF_BATTERY step
                    if (self.model.steps % 5 == 0 or 
                        not hasattr(self, '_emergency_requested')):
                        success = self.request_emergency_charging()
                        if success:
                            self._emergency_requested = True

    def step_extended_mode(self) -> None:
        """Enhanced Extended Mode behavior with better coordination."""
        # Check for low battery and request charging in Novel mode
        # Drones need higher battery thresholds due to flight costs
        if (self.model.mode == "novel" and 
            self.state != DroneState.OUT_OF_BATTERY):
            
            # Drones consume 2 battery per move, so need higher threshold
            min_battery_needed = 10  # Minimum for basic flight operations
            
            # Request charging if battery is critically low OR below 15% capacity
            should_request_charging = (
                self.battery <= max(min_battery_needed, self.battery_capacity * 0.15) and
                not (hasattr(self, '_charging_requested') and self._charging_requested)
            )
            
            if should_request_charging:
                if self._should_print():
                    print(f"🔋 Drone {self.unique_id}: Low battery ({self.battery}/{self.battery_capacity}) - requesting charging")
                self.request_charging()
        
        # Handle complete battery depletion
        if self.battery <= 0 and self.state != DroneState.OUT_OF_BATTERY:
            print(f"⚡ Drone {self.unique_id}: BATTERY DEPLETED!")
            self.state = DroneState.OUT_OF_BATTERY
            # Emergency charging request for completely depleted battery
            if self.model.mode == "novel":
                self.request_emergency_charging()
            return

        if self.state == DroneState.IDLE:
            print(f"🟦 Drone {self.unique_id}: IDLE → FLYING (Starting reconnaissance)")
            self.state = DroneState.FLYING
            # Reset tracking variables
            self._last_message_person_id = None
            self._message_type = None
            self._assigned_robot = None

        elif self.state == DroneState.FLYING:
            print(
                f"✈️ Drone {self.unique_id}: FLYING reconnaissance at {self.cell.coordinate} (Battery: {self.battery})")
            self.move_3d()
            if self.search_for_person():
                # Double-check person isn't already rescued and not already processed
                if (not self.target_person.is_rescued and 
                    (not hasattr(self, '_last_message_person_id') or 
                     self._last_message_person_id != self.target_person.unique_id)):
                    print(f"🎯 Drone {self.unique_id}: FLYING → LOCATING (Person detected)")
                    self.state = DroneState.LOCATING
                else:
                    if self.target_person.is_rescued:
                        print(f"ℹ️ Drone {self.unique_id}: Person already rescued - continuing search")
                    else:
                        print(f"ℹ️ Drone {self.unique_id}: Person already processed - continuing search")
                    self.target_person = None
            elif self.battery <= 30:
                print(f"🔋 Drone {self.unique_id}: FLYING → RETURNING (Low battery: {self.battery})")
                self.state = DroneState.RETURNING

        elif self.state == DroneState.LOCATING:
            # Double-check person is still there and not rescued
            if self.target_person and not self.target_person.is_rescued:
                urgency = self.assess_urgency()
                print(f"📋 Drone {self.unique_id}: LOCATING → COORDINATING (Assessment complete: {urgency:.2f})")
                self.state = DroneState.COORDINATING
                self.coordination_timer = 0
            else:
                print(f"ℹ️ Drone {self.unique_id}: Person no longer needs rescue - resuming search")
                self.target_person = None
                self.state = DroneState.FLYING

        elif self.state == DroneState.COORDINATING:
            # Send location message on first coordination step
            if self.coordination_timer == 0:
                # Final check before sending message
                if self.target_person and not self.target_person.is_rescued:
                    success = self.send_location_message()
                    if not success:
                        print(f"❌ Drone {self.unique_id}: Communication failed, returning to reconnaissance")
                        self.state = DroneState.FLYING
                        return
                else:
                    print(f"ℹ️ Drone {self.unique_id}: Person no longer available - resuming search")
                    self.target_person = None
                    self.state = DroneState.FLYING
                    return

            self.coordination_timer += 1

            # Check for robot acknowledgment or timeout
            if self.received_acknowledgment():
                print(f"✅ Drone {self.unique_id}: COORDINATING → WAITING (Robot(s) en route)")
                self.state = DroneState.WAITING
                # Clear acknowledgment messages
                if hasattr(self.model, 'message_system') and self.model.message_system:
                    self.model.message_system.clear_messages(self.unique_id, MessageType.MISSION_ACKNOWLEDGMENT)
            elif self.coordination_timer >= self.max_coordination_time:
                print(f"⏰ Drone {self.unique_id}: COORDINATING → WAITING (Timeout - assuming robots received)")
                self.state = DroneState.WAITING

        elif self.state == DroneState.WAITING:
            # Check if person is already rescued (by any robot)
            if self.target_person and self.target_person.is_rescued:
                print(f"✅ Drone {self.unique_id}: WAITING → FLYING (Person rescued - mission accomplished)")
                self.state = DroneState.FLYING
                self.target_person = None
                # Clear message tracking to prevent infinite loops
                self._last_message_person_id = None
                self._message_type = None
                # IMPORTANT: Move away from rescued person to avoid re-detection
                self.move_3d()
                return

            # Monitor rescue progress and handle mission completion
            if hasattr(self.model, 'message_system') and self.model.message_system:
                completion_messages = self.model.message_system.get_messages(
                    self.unique_id, MessageType.MISSION_COMPLETE)

                if completion_messages:
                    for msg in completion_messages:
                        robot_id = msg.content.get("robot_id", "Unknown")
                        completion_time = msg.content.get("completion_time", "Unknown")
                        person_id = msg.content.get("person_id", "Unknown")

                        # Check if this completion is for our target person
                        if self.target_person and str(person_id) == str(self.target_person.unique_id):
                            print(f"🎉 Drone {self.unique_id}: Our mission completed by Robot {robot_id} "
                                  f"in {completion_time} steps")

                            print(f"✅ Drone {self.unique_id}: WAITING → FLYING (Mission accomplished)")
                            self.state = DroneState.FLYING
                            self.target_person = None
                            # Clear message tracking to prevent infinite loops
                            self._last_message_person_id = None
                            self._message_type = None
                            # Clear completion messages
                            self.model.message_system.clear_messages(self.unique_id, MessageType.MISSION_COMPLETE)
                            # Move away to avoid re-detection
                            self.move_3d()
                            return

            # Emergency checks
            if not self.target_person:
                print(f"ℹ️ Drone {self.unique_id}: No target person - resuming search")
                self.state = DroneState.FLYING
                # Clear message tracking
                self._last_message_person_id = None
                self._message_type = None
            elif self.battery <= 20:
                print(f"🔋 Drone {self.unique_id}: WAITING → RETURNING (Emergency low battery)")
                self.state = DroneState.RETURNING
            else:
                # Stay in place and monitor
                print(f"⏳ Drone {self.unique_id}: WAITING - Monitoring rescue progress (Battery: {self.battery})")
                if self.battery > 0:
                    self.battery -= 1  # Hovering cost

        elif self.state == DroneState.RETURNING:
            if self.cell.coordinate == self.model.base_position:
                self.battery = self.battery_capacity
                self.state = DroneState.IDLE
                self.target_person = None
                self.coordination_timer = 0
                # Reset tracking variables
                self._last_message_person_id = None
                self._message_type = None
                self._assigned_robot = None
                print(f"🏠 Drone {self.unique_id}: Returned and recharged - Ready for next mission")
            else:
                distance_to_base = abs(self.cell.coordinate[0] - self.model.base_position[0]) + \
                                   abs(self.cell.coordinate[1] - self.model.base_position[1])
                print(f"🏠 Drone {self.unique_id}: RETURNING to base (Distance: {distance_to_base})")
                self.move_towards_base()

        elif self.state == DroneState.OUT_OF_BATTERY:
            print(f"💀 Drone {self.unique_id}: OUT_OF_BATTERY - Emergency landed")
            # If drone is at base, it can recharge automatically
            if self.cell.coordinate == self.model.base_position:
                print(f"🔋 Drone {self.unique_id}: OUT_OF_BATTERY → IDLE (Emergency recharge at base)")
                self.battery = self.battery_capacity
                self.state = DroneState.IDLE
                self.target_person = None
                self.coordination_timer = 0
                self._last_message_person_id = None
                self._message_type = None
                self._assigned_robot = None
            pass

    def step(self) -> None:
        """Execute one step based on current operation mode."""
        self.steps_since_last_action += 1

        if self.model.mode == "basic":
            self.step_basic_mode()
        elif self.model.mode in ["extended", "novel"]:
            self.step_extended_mode()

    @classmethod
    def create_agents(cls, model, count, cell, battery_capacity=None):
        """Create multiple ExplorerDrone agents and add them to the model.
        
        Args:
            model: The model instance
            count: Number of drones to create
            cell: List of cells where drones should be placed
            battery_capacity: List of battery capacities for each drone
        """
        if battery_capacity is None:
            battery_capacity = [150] * count
        
        for i in range(count):
            drone = cls(
                model=model,
                cell=cell[i],
                battery_capacity=battery_capacity[i]
            )
            model.agents.add(drone)
