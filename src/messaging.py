"""Communication system for Mountain Rescue Simulation.

This module provides the messaging infrastructure for inter-agent communication
in Extended and Novel modes of the mountain rescue simulation. It includes:
- Message types and structures
- Message routing and delivery
- Communication statistics and analysis
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any


# ================================
# MESSAGE TYPES AND STRUCTURES
# ================================

class MessageType(Enum):
    """Types of messages that can be sent between agents.

    Used for coordinated communication in Extended and Novel modes.
    """
    PERSON_LOCATION = "person_location"  # Drone -> Robot: Person found
    MISSION_ACKNOWLEDGMENT = "mission_acknowledgment"  # Robot -> Drone: Mission accepted
    MISSION_COMPLETE = "mission_complete"  # Robot -> All: Mission completed
    HELP_REQUEST = "help_request"  # Any -> Any: Request assistance
    STATUS_UPDATE = "status_update"  # Any -> Any: General status update
    CHARGING_REQUEST = "charging_request"  # Robot/Drone -> MobileCharger: Battery low, need charging


@dataclass
class Message:
    """Represents a message between agents.

    Attributes:
        sender_id: ID of the sending agent
        receiver_id: ID of the receiving agent (None for broadcast)
        message_type: Type of message being sent
        content: Message payload as key-value pairs
        timestamp: Simulation step when message was created
        priority: Message priority (0.0 = low, 1.0 = critical)
    """
    sender_id: int
    receiver_id: Optional[int]  # None for broadcast messages
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: int
    priority: float = 0.5  # 0.0 = low, 1.0 = critical

    def __post_init__(self) -> None:
        """Validate message and set priority based on content if not explicitly set."""
        # Validate priority range
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"Priority must be between 0.0 and 1.0, got {self.priority}")

        # Auto-set priority for person location messages based on urgency
        if (self.message_type == MessageType.PERSON_LOCATION and
                "urgency" in self.content and
                self.priority == 0.5):  # Only if default priority
            self.priority = min(1.0, max(0.0, self.content["urgency"]))

    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message."""
        return self.receiver_id is None

    def is_critical(self) -> bool:
        """Check if this is a critical priority message."""
        return self.priority >= 0.8


# ================================
# MESSAGE SYSTEM
# ================================

class MessageSystem:
    """Handles all communication between agents in Extended and Novel modes.

    Provides message routing, delivery, and management for coordinated
    multi-agent operations.

    Attributes:
        inbox: Direct messages by agent ID
        broadcast_messages: Messages sent to all agents
        message_history: Complete message log for analysis
        total_messages_sent: Total count of messages sent
    """

    def __init__(self):
        self.inbox: Dict[int, List[Message]] = {}  # {agent_id: [messages]}
        self.broadcast_messages: List[Message] = []
        self.message_history: List[Message] = []  # For analysis
        self.total_messages_sent = 0

    # ================================
    # MESSAGE SENDING
    # ================================

    def send_message(self, sender_id: int, receiver_id: Optional[int],
                     message_type: MessageType, content: Dict[str, Any],
                     timestamp: int, priority: float = 0.5) -> bool:
        """Send a message to a specific agent or broadcast to all.

        Args:
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent (None for broadcast)
            message_type: Type of message to send
            content: Message payload
            timestamp: Current simulation step
            priority: Message priority (0.0-1.0)

        Returns:
            True if message was sent successfully, False otherwise
        """
        try:
            message = Message(sender_id, receiver_id, message_type, content, timestamp, priority)

            if receiver_id is None:  # Broadcast message
                self.broadcast_messages.append(message)
                # Suppress verbose messaging output in quiet mode
                # print(f"📢 Agent {sender_id}: Broadcasting {message_type.value} to all agents")
            else:  # Direct message
                if receiver_id not in self.inbox:
                    self.inbox[receiver_id] = []
                self.inbox[receiver_id].append(message)
                # Suppress verbose messaging output in quiet mode
                # print(f"📩 Agent {sender_id} → Agent {receiver_id}: {message_type.value}")

            self.message_history.append(message)
            self.total_messages_sent += 1
            return True

        except Exception as e:
            print(f"❌ Failed to send message: {e}")
            return False

    # ================================
    # MESSAGE RETRIEVAL
    # ================================

    def get_messages(self, agent_id: int, message_type: Optional[MessageType] = None) -> List[Message]:
        """Get all messages for an agent, optionally filtered by type.

        Args:
            agent_id: ID of the agent requesting messages
            message_type: Optional filter for specific message types

        Returns:
            List of messages sorted by priority (highest first) then timestamp (oldest first)
        """
        # Get direct messages
        direct_messages = self.inbox.get(agent_id, []).copy()

        # Get broadcast messages
        all_messages = direct_messages + self.broadcast_messages.copy()

        # Filter by message type if specified
        if message_type:
            all_messages = [msg for msg in all_messages if msg.message_type == message_type]

        # Sort by priority (highest first) then by timestamp (oldest first)
        all_messages.sort(key=lambda x: (-x.priority, x.timestamp))

        return all_messages

    # ================================
    # MESSAGE MANAGEMENT
    # ================================

    def clear_messages(self, agent_id: int, message_type: Optional[MessageType] = None) -> None:
        """Clear messages for an agent after processing.

        Args:
            agent_id: ID of the agent whose messages to clear
            message_type: Optional filter to clear only specific message types
        """
        if agent_id in self.inbox:
            if message_type:
                self.inbox[agent_id] = [msg for msg in self.inbox[agent_id]
                                        if msg.message_type != message_type]
            else:
                self.inbox[agent_id] = []

    def clear_old_messages(self, current_step: int, max_age: int = 50) -> None:
        """Remove old messages to prevent memory bloat.

        Args:
            current_step: Current simulation step
            max_age: Maximum age in steps before messages are removed
        """
        cutoff = current_step - max_age

        # Clear old direct messages
        for agent_id in self.inbox:
            self.inbox[agent_id] = [msg for msg in self.inbox[agent_id]
                                    if msg.timestamp > cutoff]

        # Clear old broadcast messages
        self.broadcast_messages = [msg for msg in self.broadcast_messages
                                   if msg.timestamp > cutoff]

        # Keep message history for analysis but limit size
        if len(self.message_history) > 1000:
            self.message_history = self.message_history[-500:]  # Keep last 500

    # ================================
    # STATISTICS AND ANALYSIS
    # ================================

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get statistics about communication for analysis.

        Returns:
            Dictionary containing communication statistics including:
            - total_messages: Total number of messages sent
            - broadcast_messages: Number of broadcast messages
            - direct_messages: Number of direct messages
            - message_types: Breakdown by message type
            - average_priority: Average message priority
        """
        return {
            "total_messages": self.total_messages_sent,
            "broadcast_messages": len([msg for msg in self.message_history
                                       if msg.receiver_id is None]),
            "direct_messages": len([msg for msg in self.message_history
                                    if msg.receiver_id is not None]),
            "message_types": {msg_type.value: len([msg for msg in self.message_history
                                                   if msg.message_type == msg_type])
                              for msg_type in MessageType},
            "average_priority": sum(msg.priority for msg in self.message_history) /
                                max(1, len(self.message_history))
        }

    def get_agent_message_count(self, agent_id: int) -> int:
        """Get the number of messages in an agent's inbox.

        Args:
            agent_id: ID of the agent

        Returns:
            Number of messages in the agent's inbox
        """
        return len(self.inbox.get(agent_id, []))

    def get_critical_messages(self, agent_id: int) -> List[Message]:
        """Get only critical priority messages for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            List of critical priority messages (priority >= 0.8)
        """
        messages = self.get_messages(agent_id)
        return [msg for msg in messages if msg.is_critical()]

    # ================================
    # VALIDATION AND UTILITIES
    # ================================

    def validate_agent_id(self, agent_id: int) -> bool:
        """Validate that an agent ID is valid.

        Args:
            agent_id: ID to validate

        Returns:
            True if agent ID is valid, False otherwise
        """
        return isinstance(agent_id, int) and agent_id >= 0

    def has_pending_messages(self, agent_id: int) -> bool:
        """Check if an agent has any pending messages.

        Args:
            agent_id: ID of the agent to check

        Returns:
            True if agent has pending messages, False otherwise
        """
        return self.get_agent_message_count(agent_id) > 0

    def purge_agent_messages(self, agent_id: int) -> int:
        """Remove all messages for a specific agent.

        Args:
            agent_id: ID of the agent whose messages to purge

        Returns:
            Number of messages that were purged
        """
        if agent_id not in self.inbox:
            return 0

        purged_count = len(self.inbox[agent_id])
        self.inbox[agent_id] = []
        return purged_count

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status for debugging.

        Returns:
            Dictionary with system status information
        """
        total_inbox_messages = sum(len(msgs) for msgs in self.inbox.values())

        return {
            "total_agents_with_messages": len(self.inbox),
            "total_inbox_messages": total_inbox_messages,
            "total_broadcast_messages": len(self.broadcast_messages),
            "total_history_messages": len(self.message_history),
            "total_messages_sent": self.total_messages_sent,
            "average_messages_per_agent": total_inbox_messages / max(1, len(self.inbox))
        }