#!/usr/bin/env python3
"""
GNN-MARL Training Script for Mountain Rescue Simulation

This module provides a comprehensive training framework for Graph Neural Network-based 
Multi-Agent Reinforcement Learning (GNN-MARL) in the mountain rescue environment.

The training system implements:
- GPU/CPU device management
- Episode-based training with early stopping
- Performance evaluation and metrics tracking
- Model checkpointing and visualization
- Command-line interface for configuration
"""

import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

from model import MountainRescueModel
from gnn_marl import GNNConfig, MARLConfig, GNN_MARL_System
from agents import FirstAidRobot, ExplorerDrone, Person


# =============================================================================
# DEVICE AND ENVIRONMENT SETUP
# =============================================================================

def setup_device(force_cpu: bool = False) -> torch.device:
    """
    Setup PyTorch device for training.
    
    Args:
        force_cpu: If True, force CPU usage even if GPU is available
        
    Returns:
        PyTorch device (cuda or cpu)
    """
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        if force_cpu:
            print("💻 Using CPU (forced by --cpu-only)")
        else:
            print("💻 Using CPU (GPU not available)")
    
    return device


def create_default_config() -> Dict[str, Any]:
    """
    Create default training configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        # Training parameters
        'num_episodes': 1000,
        'max_steps_per_episode': 500,
        'eval_interval': 50,
        'save_interval': 100,
        'early_stopping_patience': 200,
        'force_cpu': False,
        'results_dir': 'results',
        
        # Environment configuration
        'environment': {
            'width': 15,
            'height': 15,
            'n_robots': 4,
            'n_drones': 3,
            'n_persons': 8,
            'spawn_interval': 30,
            'max_persons': 30
        },
        
        # GNN configuration
        'node_feature_dim': 64,
        'edge_feature_dim': 32,
        'hidden_dim': 128,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'use_attention': True,
        'use_residual': True,
        
        # MARL configuration
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'eps_start': 1.0,
        'eps_end': 0.01,
        'eps_decay': 0.995,
        'memory_size': 20000,
        'batch_size': 128,
        'target_update_freq': 100,
        'use_double_dqn': True,
        'use_dueling': True
    }


# =============================================================================
# MAIN TRAINING CLASS
# =============================================================================

class GNNMARLTrainer:
    """
    Comprehensive trainer for GNN-MARL system in mountain rescue environment.
    
    This class handles:
    - Training loop with episode management
    - Performance evaluation and metrics tracking
    - Model checkpointing and persistence
    - Visualization and logging
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GNN-MARL trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.training_start_time = datetime.now()
        
        # Setup device and configurations
        self._setup_device()
        self._setup_training_config()
        self._setup_model_configs()
        self._setup_directories()
        self._initialize_metrics()
        
        self._print_initialization_summary()

    def _setup_device(self):
        """Setup PyTorch device for training."""
        force_cpu = self.config.get('force_cpu', False)
        self.device = setup_device(force_cpu=force_cpu)

    def _setup_training_config(self):
        """Setup training hyperparameters."""
        self.num_episodes = self.config.get('num_episodes', 1000)
        self.max_steps_per_episode = self.config.get('max_steps_per_episode', 500)
        self.eval_interval = self.config.get('eval_interval', 50)
        self.save_interval = self.config.get('save_interval', 100)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 200)

    def _setup_model_configs(self):
        """Setup GNN and MARL configurations."""
        # Environment configuration
        self.env_config = self.config.get('environment', {
            'width': 15,
            'height': 15,
            'n_robots': 4,
            'n_drones': 3,
            'n_persons': 8,
            'spawn_interval': 30,
            'max_persons': 30
        })
        
        # GNN configuration
        self.gnn_config = GNNConfig(
            node_feature_dim=self.config.get('node_feature_dim', 64),
            edge_feature_dim=self.config.get('edge_feature_dim', 32),
            hidden_dim=self.config.get('hidden_dim', 128),
            num_layers=self.config.get('num_layers', 4),
            num_heads=self.config.get('num_heads', 8),
            dropout=self.config.get('dropout', 0.1),
            use_attention=self.config.get('use_attention', True),
            use_residual=self.config.get('use_residual', True)
        )
        
        # MARL configuration
        self.marl_config = MARLConfig(
            lr=self.config.get('learning_rate', 1e-4),  # Lower learning rate
            gamma=self.config.get('gamma', 0.95),       # Lower discount factor
            eps_start=self.config.get('eps_start', 0.5), # Lower starting exploration
            eps_end=self.config.get('eps_end', 0.05),   # Higher minimum exploration
            eps_decay=self.config.get('eps_decay', 0.995), # Faster decay
            memory_size=self.config.get('memory_size', 20000), # Larger buffer
            batch_size=self.config.get('batch_size', 128),    # Larger batch
            target_update_freq=self.config.get('target_update_freq', 200), # Less frequent updates
            use_double_dqn=self.config.get('use_double_dqn', True),
            use_dueling=self.config.get('use_dueling', True)
        )

    def _setup_directories(self):
        """Setup results and output directories."""
        self.results_dir = self.config.get('results_dir', 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def _initialize_metrics(self):
        """Initialize training metrics tracking."""
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'rescue_rates': [],
            'robot_losses': [],
            'drone_losses': [],
            'communication_efficiency': [],
            'average_response_times': [],
            'evaluation_scores': [],
            'eval_rescue_rates': [],
            'eval_agent_utilizations': [],
            'eval_coordination_scores': [],
            'eval_consistency_scores': [],
            'eval_success_rates': []
        }
        
        # Best model tracking
        self.best_eval_score = float('-inf')
        self.episodes_without_improvement = 0

    def _print_initialization_summary(self):
        """Print trainer initialization summary."""
        print(f"🧠 GNN-MARL Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Episodes: {self.num_episodes}")
        print(f"   Max steps per episode: {self.max_steps_per_episode}")
        print(f"   Environment: {self.env_config['width']}x{self.env_config['height']}")
        print(f"   Agents: {self.env_config['n_robots']} robots, {self.env_config['n_drones']} drones")
        print(f"   Results directory: {self.results_dir}")

    # =============================================================================
    # ENVIRONMENT MANAGEMENT
    # =============================================================================

    def create_environment(self, seed: Optional[int] = None) -> MountainRescueModel:
        """
        Create a new training environment.
        
        Args:
            seed: Random seed for environment initialization
            
        Returns:
            Configured MountainRescueModel instance
        """
        model = MountainRescueModel(
            width=self.env_config['width'],
            height=self.env_config['height'],
            n_robots=self.env_config['n_robots'],
            n_drones=self.env_config['n_drones'],
            n_persons=self.env_config['n_persons'],
            mode="novel",
            seed=seed,
            spawn_interval=self.env_config['spawn_interval'],
            max_persons=self.env_config['max_persons'],
            use_gnn_marl=True,
            device=self.device,
            quiet=self.config.get('quiet', False)
        )
        
        # Move GNN-MARL system to device if it exists
        if hasattr(model, 'gnn_marl_system') and model.gnn_marl_system:
            model.gnn_marl_system.to(self.device)
        
        return model

    def reset_episode(self, model: MountainRescueModel) -> MountainRescueModel:
        """
        Reset environment for new episode.
        
        Args:
            model: Current model instance
            
        Returns:
            New model instance with transferred GNN-MARL system
        """
        # Create new model instance to ensure clean reset
        seed = np.random.randint(0, 10000)
        new_model = self.create_environment(seed)
        
        # Transfer the trained GNN-MARL system
        if hasattr(model, 'gnn_marl_system') and model.gnn_marl_system:
            new_model.gnn_marl_system = model.gnn_marl_system
            new_model.gnn_marl_system.episode_count += 1
            # Ensure system is on correct device
            new_model.gnn_marl_system.to(self.device)
        
        return new_model

    # =============================================================================
    # METRICS AND EVALUATION
    # =============================================================================

    def calculate_episode_reward(self, model: MountainRescueModel) -> float:
        """
        Calculate total episode reward based on multiple criteria.
        
        Args:
            model: Model instance to evaluate
            
        Returns:
            Total episode reward
        """
        # Global mission success metrics
        persons = [a for a in model.agents if isinstance(a, Person)]
        rescue_rate = model.rescued_count / max(1, len(persons))
        rescue_bonus = model.rescued_count * 100  # 100 points per rescue
        
        # Efficiency metrics
        steps_penalty = model.steps * 0.1  # Small penalty for longer episodes
        
        # Communication efficiency (lower is better)
        comm_efficiency = model.calculate_communication_efficiency()
        comm_bonus = max(0, 20 - comm_efficiency) if comm_efficiency > 0 else 0
        
        # Agent activity bonus
        robots_drones = [a for a in model.agents if isinstance(a, (FirstAidRobot, ExplorerDrone))]
        active_agents = len([a for a in robots_drones if a.battery > 0])
        total_agents = len(robots_drones)
        activity_bonus = (active_agents / max(1, total_agents)) * 50
        
        total_reward = rescue_bonus + comm_bonus + activity_bonus - steps_penalty
        
        return total_reward

    def evaluate_model(self, model: MountainRescueModel, num_eval_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate the trained model over multiple episodes with comprehensive metrics.
        
        Args:
            model: Model to evaluate
            num_eval_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"🔍 Evaluating model over {num_eval_episodes} episodes...")
        
        # Disable training for evaluation
        if model.gnn_marl_system:
            model.gnn_marl_system.disable_training()
        
        eval_metrics = {
            'total_rescues': 0,
            'total_steps': 0,
            'total_episodes': num_eval_episodes,
            'rescue_rates': [],
            'episode_lengths': [],
            'communication_efficiencies': [],
            'episode_rewards': [],
            'agent_utilizations': [],
            'completion_times': [],
            'coordination_scores': []
        }
        
        for eval_ep in range(num_eval_episodes):
            eval_model = self.create_environment(seed=eval_ep + 42)
            
            # Transfer trained model
            if model.gnn_marl_system:
                eval_model.gnn_marl_system = model.gnn_marl_system
                eval_model.gnn_marl_system.disable_training()
                eval_model.gnn_marl_system.to(self.device)
            
            step_count = 0
            initial_persons = len([a for a in eval_model.agents if isinstance(a, Person)])
            
            while step_count < self.max_steps_per_episode:
                eval_model.step()
                step_count += 1
                
                # Early termination if all persons rescued
                persons = [a for a in eval_model.agents if isinstance(a, Person)]
                if persons and all(p.is_rescued for p in persons):
                    break
            
            # Collect comprehensive episode metrics
            persons = [a for a in eval_model.agents if isinstance(a, Person)]
            rescue_rate = eval_model.rescued_count / max(1, len(persons))
            comm_eff = eval_model.calculate_communication_efficiency()
            episode_reward = self.calculate_episode_reward(eval_model)
            
            # Calculate agent utilization
            robots_drones = [a for a in eval_model.agents if isinstance(a, (FirstAidRobot, ExplorerDrone))]
            active_agents = len([a for a in robots_drones if a.battery > 0])
            agent_util = active_agents / max(1, len(robots_drones))
            
            # Calculate completion efficiency (faster completion = higher score)
            completion_time = step_count / self.max_steps_per_episode
            
            # Calculate coordination score based on rescue efficiency
            rescue_efficiency = eval_model.rescued_count / max(1, step_count) * 100
            coordination_score = rescue_efficiency * (1 + agent_util) * (2 - completion_time)
            
            eval_metrics['total_rescues'] += eval_model.rescued_count
            eval_metrics['total_steps'] += step_count
            eval_metrics['rescue_rates'].append(rescue_rate)
            eval_metrics['episode_lengths'].append(step_count)
            eval_metrics['communication_efficiencies'].append(comm_eff)
            eval_metrics['episode_rewards'].append(episode_reward)
            eval_metrics['agent_utilizations'].append(agent_util)
            eval_metrics['completion_times'].append(completion_time)
            eval_metrics['coordination_scores'].append(coordination_score)
        
        # Calculate comprehensive averages
        avg_metrics = {
            'avg_rescue_rate': np.mean(eval_metrics['rescue_rates']),
            'avg_episode_length': np.mean(eval_metrics['episode_lengths']),
            'avg_communication_efficiency': np.mean(eval_metrics['communication_efficiencies']),
            'avg_episode_reward': np.mean(eval_metrics['episode_rewards']),
            'avg_agent_utilization': np.mean(eval_metrics['agent_utilizations']),
            'avg_completion_time': np.mean(eval_metrics['completion_times']),
            'avg_coordination_score': np.mean(eval_metrics['coordination_scores']),
            'total_rescues': eval_metrics['total_rescues'],
            'success_rate': len([r for r in eval_metrics['rescue_rates'] if r >= 0.8]) / num_eval_episodes,
            'consistency_score': 1.0 - np.std(eval_metrics['rescue_rates']) if len(eval_metrics['rescue_rates']) > 1 else 1.0
        }
        
        # Re-enable training
        if model.gnn_marl_system:
            model.gnn_marl_system.enable_training()
        
        # Calculate comprehensive evaluation score
        # Weighted combination of multiple factors
        rescue_score = avg_metrics['avg_rescue_rate'] * 40  # 40% weight
        efficiency_score = (1 - avg_metrics['avg_completion_time']) * 20  # 20% weight
        coordination_score = avg_metrics['avg_coordination_score'] * 0.2  # Normalized coordination
        consistency_score = avg_metrics['consistency_score'] * 15  # 15% weight
        success_bonus = avg_metrics['success_rate'] * 25  # 25% weight
        
        eval_score = rescue_score + efficiency_score + coordination_score + consistency_score + success_bonus
        avg_metrics['eval_score'] = eval_score
        
        print(f"   📊 Evaluation Results:")
        print(f"      Average Rescue Rate: {avg_metrics['avg_rescue_rate']:.3f}")
        print(f"      Success Rate (≥80%): {avg_metrics['success_rate']:.3f}")
        print(f"      Average Episode Length: {avg_metrics['avg_episode_length']:.1f}")
        print(f"      Agent Utilization: {avg_metrics['avg_agent_utilization']:.3f}")
        print(f"      Coordination Score: {avg_metrics['avg_coordination_score']:.2f}")
        print(f"      Consistency Score: {avg_metrics['consistency_score']:.3f}")
        print(f"      Evaluation Score: {eval_score:.2f}")
        
        return avg_metrics

    # =============================================================================
    # MODEL PERSISTENCE AND VISUALIZATION
    # =============================================================================

    def save_model(self, model: MountainRescueModel, episode: int, is_best: bool = False):
        """
        Save the trained model to disk.
        
        Args:
            model: Model to save
            episode: Current episode number
            is_best: Whether this is the best model so far
        """
        if not model.gnn_marl_system:
            return
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            filepath = os.path.join(self.results_dir, f"best_gnn_marl_model.pth")
        else:
            filepath = os.path.join(self.results_dir, f"gnn_marl_model_ep{episode}_{timestamp}.pth")
        
        try:
            model.gnn_marl_system.save_models(filepath)
            print(f"💾 Model saved: {filepath}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            print(f"   Continuing training without saving...")

    def save_training_metrics(self):
        """Save training metrics to JSON file."""
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        metrics_file = os.path.join(self.results_dir, "training_metrics.json")
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_metrics = {}
            for key, value in self.training_metrics.items():
                if isinstance(value, list):
                    json_metrics[key] = [float(x) if not isinstance(x, (list, dict)) else x for x in value]
                else:
                    json_metrics[key] = float(value) if isinstance(value, (int, float, np.number)) else value
            
            with open(metrics_file, 'w') as f:
                json.dump(json_metrics, f, indent=2)
            
            print(f"📈 Training metrics saved: {metrics_file}")
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")
            print(f"   Continuing training without saving metrics...")

    def plot_training_progress(self):
        """Generate and save comprehensive training progress plots."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('GNN-MARL Training Progress', fontsize=16)
        
        # Episode rewards
        if self.training_metrics['episode_rewards']:
            axes[0, 0].plot(self.training_metrics['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True)
        
        # Rescue rates
        if self.training_metrics['rescue_rates']:
            axes[0, 1].plot(self.training_metrics['rescue_rates'], label='Training', alpha=0.7)
            if self.training_metrics['eval_rescue_rates']:
                eval_episodes = [i * self.eval_interval for i in range(len(self.training_metrics['eval_rescue_rates']))]
                axes[0, 1].plot(eval_episodes, self.training_metrics['eval_rescue_rates'], 
                              label='Evaluation', color='red', linewidth=2)
            axes[0, 1].set_title('Rescue Rates')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Rescue Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Episode lengths
        if self.training_metrics['episode_lengths']:
            axes[0, 2].plot(self.training_metrics['episode_lengths'])
            axes[0, 2].set_title('Episode Lengths')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Steps')
            axes[0, 2].grid(True)
        
        # Training losses
        if self.training_metrics['robot_losses']:
            axes[1, 0].plot(self.training_metrics['robot_losses'], label='Robot', alpha=0.7)
        if self.training_metrics['drone_losses']:
            axes[1, 0].plot(self.training_metrics['drone_losses'], label='Drone', alpha=0.7)
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Communication efficiency
        if self.training_metrics['communication_efficiency']:
            axes[1, 1].plot(self.training_metrics['communication_efficiency'])
            axes[1, 1].set_title('Communication Efficiency')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Messages/Rescue')
            axes[1, 1].grid(True)
        
        # Evaluation scores
        if self.training_metrics['evaluation_scores']:
            axes[1, 2].plot(self.training_metrics['evaluation_scores'], linewidth=2)
            axes[1, 2].set_title('Evaluation Scores')
            axes[1, 2].set_xlabel('Evaluation')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].grid(True)
        
        # Agent utilization
        if self.training_metrics['eval_agent_utilizations']:
            axes[2, 0].plot(self.training_metrics['eval_agent_utilizations'], color='green')
            axes[2, 0].set_title('Agent Utilization')
            axes[2, 0].set_xlabel('Evaluation')
            axes[2, 0].set_ylabel('Utilization Rate')
            axes[2, 0].grid(True)
        
        # Coordination scores
        if self.training_metrics['eval_coordination_scores']:
            axes[2, 1].plot(self.training_metrics['eval_coordination_scores'], color='orange')
            axes[2, 1].set_title('Coordination Scores')
            axes[2, 1].set_xlabel('Evaluation')
            axes[2, 1].set_ylabel('Coordination Score')
            axes[2, 1].grid(True)
        
        # Success rates
        if self.training_metrics['eval_success_rates']:
            axes[2, 2].plot(self.training_metrics['eval_success_rates'], color='purple')
            axes[2, 2].set_title('Success Rates (≥80%)')
            axes[2, 2].set_xlabel('Evaluation')
            axes[2, 2].set_ylabel('Success Rate')
            axes[2, 2].grid(True)
        
        plt.tight_layout()
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        plot_file = os.path.join(self.results_dir, "training_progress.png")
        try:
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Training plot saved: {plot_file}")
        except Exception as e:
            print(f"❌ Error saving plot: {e}")
        finally:
            plt.close()

    # =============================================================================
    # MAIN TRAINING LOOP
    # =============================================================================

    def train(self):
        """Execute main training loop."""
        print(f"🚀 Starting GNN-MARL training...")
        print(f"   Training configuration: {self.num_episodes} episodes, {self.max_steps_per_episode} max steps")
        
        # Create initial environment
        model = self.create_environment(seed=42)
        
        # Clear GPU cache at start
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        try:
            for episode in range(self.num_episodes):
                episode_start_time = time.time()
                
                # Reset environment for new episode
                model = self.reset_episode(model)
                
                episode_reward = 0
                step_count = 0
                
                print(f"\n📅 Episode {episode + 1}/{self.num_episodes}")
                
                # Run episode
                while step_count < self.max_steps_per_episode:
                    model.step()
                    step_count += 1
                    
                    # Check if all persons rescued (early termination)
                    persons = [a for a in model.agents if isinstance(a, Person)]
                    if persons and all(p.is_rescued for p in persons):
                        print(f"   ✅ All persons rescued at step {step_count}")
                        break
                
                # Calculate episode metrics
                episode_reward = self.calculate_episode_reward(model)
                persons = [a for a in model.agents if isinstance(a, Person)]
                rescue_rate = model.rescued_count / max(1, len(persons))
                comm_eff = model.calculate_communication_efficiency()
                
                # Store metrics
                self.training_metrics['episode_rewards'].append(episode_reward)
                self.training_metrics['episode_lengths'].append(step_count)
                self.training_metrics['rescue_rates'].append(rescue_rate)
                self.training_metrics['communication_efficiency'].append(comm_eff)
                
                # Get GNN-MARL performance metrics
                if model.gnn_marl_system:
                    perf_metrics = model.gnn_marl_system.get_performance_metrics()
                    if 'avg_robot_loss' in perf_metrics:
                        self.training_metrics['robot_losses'].append(perf_metrics['avg_robot_loss'])
                    if 'avg_drone_loss' in perf_metrics:
                        self.training_metrics['drone_losses'].append(perf_metrics['avg_drone_loss'])
                
                episode_time = time.time() - episode_start_time
                
                # Add GPU memory info if using GPU
                gpu_info = ""
                if self.device.type == 'cuda':
                    gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
                    gpu_info = f", GPU: {gpu_memory:.1f}GB"
                
                print(f"   💯 Reward: {episode_reward:.2f}, Rescue Rate: {rescue_rate:.3f}, "
                      f"Steps: {step_count}, Time: {episode_time:.2f}s{gpu_info}")
                
                # Evaluation
                if (episode + 1) % self.eval_interval == 0:
                    eval_metrics = self.evaluate_model(model)
                    
                    # Store comprehensive evaluation metrics
                    self.training_metrics['evaluation_scores'].append(eval_metrics['eval_score'])
                    self.training_metrics['eval_rescue_rates'].append(eval_metrics['avg_rescue_rate'])
                    self.training_metrics['eval_agent_utilizations'].append(eval_metrics['avg_agent_utilization'])
                    self.training_metrics['eval_coordination_scores'].append(eval_metrics['avg_coordination_score'])
                    self.training_metrics['eval_consistency_scores'].append(eval_metrics['consistency_score'])
                    self.training_metrics['eval_success_rates'].append(eval_metrics['success_rate'])
                    
                    # Check for best model
                    if eval_metrics['eval_score'] > self.best_eval_score:
                        self.best_eval_score = eval_metrics['eval_score']
                        self.episodes_without_improvement = 0
                        self.save_model(model, episode + 1, is_best=True)
                        print(f"   🏆 New best model! Score: {self.best_eval_score:.2f}")
                    else:
                        self.episodes_without_improvement += self.eval_interval
                
                # Save model periodically
                if (episode + 1) % self.save_interval == 0:
                    self.save_model(model, episode + 1)
                
                # Early stopping check
                if self.episodes_without_improvement >= self.early_stopping_patience:
                    print(f"🛑 Early stopping: No improvement for {self.early_stopping_patience} episodes")
                    break
                
                # Clear GPU cache periodically
                if self.device.type == 'cuda' and (episode + 1) % 50 == 0:
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            print(f"\n⏹️ Training interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._finalize_training(model)

    def _finalize_training(self, model: MountainRescueModel):
        """Finalize training with evaluation and saving."""
        print(f"\n🏁 Training completed!")
        
        if model.gnn_marl_system:
            final_eval = self.evaluate_model(model, num_eval_episodes=10)
            print(f"\n📊 Final Evaluation:")
            for key, value in final_eval.items():
                print(f"   {key}: {value:.4f}")
        
        # Save results
        self.save_model(model, self.num_episodes)
        self.save_training_metrics()
        self.plot_training_progress()
        
        training_time = datetime.now() - self.training_start_time
        print(f"⏱️ Total training time: {training_time}")


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='Train GNN-MARL for Mountain Rescue')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--max-steps', type=int, default=500, 
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--eval-interval', type=int, default=50, 
                       help='Evaluation interval in episodes (default: 50)')
    parser.add_argument('--save-interval', type=int, default=100, 
                       help='Model save interval in episodes (default: 100)')
    
    # Model parameters
    parser.add_argument('--learning-rate', type=float, default=3e-4, 
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--hidden-dim', type=int, default=128, 
                       help='GNN hidden dimension (default: 128)')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size (default: 32)')
    
    # System parameters
    parser.add_argument('--results-dir', type=str, default='results', 
                       help='Results directory (default: results)')
    parser.add_argument('--cpu-only', action='store_true', 
                       help='Force CPU usage even if GPU available')
    parser.add_argument('--quiet', action='store_true', 
                       help='Suppress verbose agent and model output')
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create training configuration from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Training configuration dictionary
    """
    config = create_default_config()
    
    # Update with command-line arguments
    config.update({
        'num_episodes': args.episodes,
        'max_steps_per_episode': args.max_steps,
        'eval_interval': args.eval_interval,
        'save_interval': args.save_interval,
        'learning_rate': args.learning_rate,
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'results_dir': args.results_dir,
        'force_cpu': args.cpu_only,
        'quiet': args.quiet,
    })
    
    return config


def main():
    """Main function for command-line training execution."""
    # Parse arguments and create configuration
    args = parse_arguments()
    config = create_config_from_args(args)
    
    # Create and run trainer
    trainer = GNNMARLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()