#!/usr/bin/env python3
"""
Curriculum Learning GNN-MARL Training Script for Mountain Rescue Simulation

This module implements curriculum learning for Graph Neural Network-based 
Multi-Agent Reinforcement Learning (GNN-MARL) in the mountain rescue environment.

The curriculum learning system provides:
- Progressive difficulty scaling through predefined stages
- Performance-based and episode-based stage progression
- Dynamic environment parameter adjustment
- Comprehensive curriculum metrics tracking
- Visualization of learning progress across curriculum stages

Based on concepts from research paper 2403.13093v1.pdf with curriculum learning enhancements.
"""

import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from model import MountainRescueModel
from gnn_marl import GNNConfig, MARLConfig, GNN_MARL_System
from agents import FirstAidRobot, ExplorerDrone, Person


# =============================================================================
# CURRICULUM LEARNING CONFIGURATION
# =============================================================================

def create_curriculum_config() -> Dict[str, Any]:
    """
    Create curriculum learning configuration with progressive difficulty stages.
    
    Returns:
        Curriculum configuration dictionary
    """
    return {
        # Curriculum learning parameters
        'curriculum_enabled': True,
        'curriculum_stages': [
            {
                'name': 'Easy',
                'episodes': 100,
                'width': 8,
                'height': 8,
                'n_robots': 2,
                'n_drones': 1,
                'n_persons': 2,
                'spawn_interval': 60,
                'max_persons': 4,
                'success_threshold': 0.7,  # 70% success rate to advance
                'min_episodes': 50  # Minimum episodes before advancement
            },
            {
                'name': 'Medium',
                'episodes': 150,
                'width': 10,
                'height': 10,
                'n_robots': 3,
                'n_drones': 2,
                'n_persons': 4,
                'spawn_interval': 45,
                'max_persons': 8,
                'success_threshold': 0.6,
                'min_episodes': 75
            },
            {
                'name': 'Hard',
                'episodes': 200,
                'width': 12,
                'height': 12,
                'n_robots': 4,
                'n_drones': 3,
                'n_persons': 6,
                'spawn_interval': 35,
                'max_persons': 16,
                'success_threshold': 0.5,
                'min_episodes': 100
            },
            {
                'name': 'Expert',
                'episodes': 300,
                'width': 15,
                'height': 15,
                'n_robots': 5,
                'n_drones': 4,
                'n_persons': 8,
                'spawn_interval': 25,
                'max_persons': 32,
                'success_threshold': 0.4,
                'min_episodes': 150
            }
        ],
        
        # Curriculum progression parameters
        'performance_window': 20,  # Episodes to average for performance evaluation
        'progression_patience': 30,  # Episodes to wait before forcing progression
        'regression_threshold': 0.1,  # Performance drop threshold for regression
        'max_regressions': 2,  # Maximum allowed stage regressions
        
        # Training parameters
        'num_episodes': 1000,
        'max_steps_per_episode': 500,
        'eval_interval': 25,
        'save_interval': 50,
        'early_stopping_patience': 100,
        'force_cpu': False,
        'results_dir': 'results_curriculum',
        
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
# CURRICULUM LEARNING MANAGER
# =============================================================================

class CurriculumManager:
    """
    Manages curriculum learning progression and stage transitions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize curriculum learning manager.
        
        Args:
            config: Training configuration with curriculum parameters
        """
        self.config = config
        self.curriculum_stages = config['curriculum_stages']
        self.current_stage = 0
        self.stage_episode_count = 0
        self.total_episodes = 0
        self.performance_history = []
        self.stage_history = []
        self.regression_count = 0
        self.last_progression_episode = 0
        
        # Performance tracking
        self.performance_window = config.get('performance_window', 20)
        self.progression_patience = config.get('progression_patience', 30)
        self.regression_threshold = config.get('regression_threshold', 0.1)
        self.max_regressions = config.get('max_regressions', 2)
        
        print(f"🎯 Curriculum Learning initialized with {len(self.curriculum_stages)} stages")
        self._print_current_stage()
    
    def get_current_stage_config(self) -> Dict[str, Any]:
        """
        Get current curriculum stage configuration.
        
        Returns:
            Current stage parameters
        """
        return self.curriculum_stages[self.current_stage]
    
    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get environment configuration for current curriculum stage.
        
        Returns:
            Environment parameters for current stage
        """
        stage_config = self.get_current_stage_config()
        return {
            'width': stage_config['width'],
            'height': stage_config['height'],
            'n_robots': stage_config['n_robots'],
            'n_drones': stage_config['n_drones'],
            'n_persons': stage_config['n_persons'],
            'spawn_interval': stage_config['spawn_interval'],
            'max_persons': stage_config['max_persons']
        }
    
    def update_performance(self, episode_metrics: Dict[str, float]) -> None:
        """
        Update performance tracking with episode metrics.
        
        Args:
            episode_metrics: Dictionary containing episode performance metrics
        """
        self.performance_history.append({
            'episode': self.total_episodes,
            'stage': self.current_stage,
            'stage_episode': self.stage_episode_count,
            'rescue_rate': episode_metrics.get('rescue_rate', 0.0),
            'episode_reward': episode_metrics.get('episode_reward', 0.0),
            'communication_efficiency': episode_metrics.get('communication_efficiency', 0.0)
        })
        
        self.stage_episode_count += 1
        self.total_episodes += 1
    
    def should_progress_stage(self) -> bool:
        """
        Check if curriculum should progress to next stage.
        
        Returns:
            True if should progress to next stage
        """
        if self.current_stage >= len(self.curriculum_stages) - 1:
            return False
        
        stage_config = self.get_current_stage_config()
        
        # Check minimum episodes requirement
        if self.stage_episode_count < stage_config['min_episodes']:
            return False
        
        # Check performance-based progression
        if len(self.performance_history) >= self.performance_window:
            recent_performance = self.performance_history[-self.performance_window:]
            avg_rescue_rate = np.mean([p['rescue_rate'] for p in recent_performance])
            
            if avg_rescue_rate >= stage_config['success_threshold']:
                print(f"✅ Performance threshold reached: {avg_rescue_rate:.3f} >= {stage_config['success_threshold']:.3f}")
                return True
        
        # Check patience-based progression (force advancement)
        episodes_since_progression = self.stage_episode_count - self.last_progression_episode
        if episodes_since_progression >= self.progression_patience:
            print(f"⏰ Patience threshold reached: {episodes_since_progression} >= {self.progression_patience}")
            return True
        
        return False
    
    def should_regress_stage(self) -> bool:
        """
        Check if curriculum should regress to previous stage due to poor performance.
        
        Returns:
            True if should regress to previous stage
        """
        if self.current_stage <= 0 or self.regression_count >= self.max_regressions:
            return False
        
        if len(self.performance_history) >= self.performance_window:
            recent_performance = self.performance_history[-self.performance_window:]
            avg_rescue_rate = np.mean([p['rescue_rate'] for p in recent_performance])
            
            # Get expected performance from previous stage
            if self.current_stage > 0:
                prev_stage_config = self.curriculum_stages[self.current_stage - 1]
                expected_threshold = prev_stage_config['success_threshold']
                
                if avg_rescue_rate < expected_threshold - self.regression_threshold:
                    print(f"⚠️ Performance regression detected: {avg_rescue_rate:.3f} < {expected_threshold - self.regression_threshold:.3f}")
                    return True
        
        return False
    
    def progress_to_next_stage(self) -> bool:
        """
        Progress to next curriculum stage.
        
        Returns:
            True if progression successful
        """
        if self.current_stage >= len(self.curriculum_stages) - 1:
            return False
        
        prev_stage = self.current_stage
        self.current_stage += 1
        self.stage_episode_count = 0
        self.last_progression_episode = 0
        
        self.stage_history.append({
            'episode': self.total_episodes,
            'from_stage': prev_stage,
            'to_stage': self.current_stage,
            'type': 'progression'
        })
        
        print(f"📈 Progressed to stage {self.current_stage + 1}/{len(self.curriculum_stages)}")
        self._print_current_stage()
        return True
    
    def regress_to_previous_stage(self) -> bool:
        """
        Regress to previous curriculum stage.
        
        Returns:
            True if regression successful
        """
        if self.current_stage <= 0:
            return False
        
        prev_stage = self.current_stage
        self.current_stage -= 1
        self.stage_episode_count = 0
        self.regression_count += 1
        
        self.stage_history.append({
            'episode': self.total_episodes,
            'from_stage': prev_stage,
            'to_stage': self.current_stage,
            'type': 'regression'
        })
        
        print(f"📉 Regressed to stage {self.current_stage + 1}/{len(self.curriculum_stages)} (regression {self.regression_count}/{self.max_regressions})")
        self._print_current_stage()
        return True
    
    def _print_current_stage(self) -> None:
        """Print current curriculum stage information."""
        stage_config = self.get_current_stage_config()
        print(f"🎯 Current Stage: {stage_config['name']} ({self.current_stage + 1}/{len(self.curriculum_stages)})")
        print(f"   Grid: {stage_config['width']}x{stage_config['height']}")
        print(f"   Agents: {stage_config['n_robots']} robots, {stage_config['n_drones']} drones")
        print(f"   Persons: {stage_config['n_persons']} initial, {stage_config['max_persons']} max")
        print(f"   Success threshold: {stage_config['success_threshold']:.1%}")
        print(f"   Episodes in stage: {self.stage_episode_count}")
    
    def get_curriculum_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive curriculum learning metrics.
        
        Returns:
            Dictionary containing curriculum metrics
        """
        return {
            'current_stage': self.current_stage,
            'stage_episode_count': self.stage_episode_count,
            'total_episodes': self.total_episodes,
            'performance_history': self.performance_history,
            'stage_history': self.stage_history,
            'regression_count': self.regression_count,
            'stages_completed': self.current_stage,
            'current_stage_config': self.get_current_stage_config()
        }


# =============================================================================
# CURRICULUM LEARNING TRAINER
# =============================================================================

class CurriculumGNNMARLTrainer:
    """
    Comprehensive trainer for GNN-MARL system with curriculum learning.
    
    This class handles:
    - Curriculum learning progression and stage management
    - Dynamic environment parameter adjustment
    - Performance tracking across curriculum stages
    - Model checkpointing with curriculum state
    - Visualization of curriculum learning progress
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize curriculum learning GNN-MARL trainer.
        
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
        
        # Initialize curriculum manager
        self.curriculum_manager = CurriculumManager(config)
        
        # Initialize metrics tracking
        self._initialize_metrics()
        
        self._print_initialization_summary()
    
    def _setup_device(self):
        """Setup PyTorch device for training."""
        force_cpu = self.config.get('force_cpu', False)
        
        if not force_cpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device("cpu")
            if force_cpu:
                print("💻 Using CPU (forced by --cpu-only)")
            else:
                print("💻 Using CPU (GPU not available)")
    
    def _setup_training_config(self):
        """Setup training configuration parameters."""
        self.num_episodes = self.config.get('num_episodes', 1000)
        self.max_steps_per_episode = self.config.get('max_steps_per_episode', 500)
        self.eval_interval = self.config.get('eval_interval', 25)
        self.save_interval = self.config.get('save_interval', 50)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 100)
    
    def _setup_model_configs(self):
        """Setup GNN and MARL model configurations."""
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
        
        self.marl_config = MARLConfig(
            lr=self.config.get('learning_rate', 1e-4),
            gamma=self.config.get('gamma', 0.99),
            eps_start=self.config.get('eps_start', 1.0),
            eps_end=self.config.get('eps_end', 0.01),
            eps_decay=self.config.get('eps_decay', 0.995),
            memory_size=self.config.get('memory_size', 20000),
            batch_size=self.config.get('batch_size', 128),
            target_update_freq=self.config.get('target_update_freq', 100),
            use_double_dqn=self.config.get('use_double_dqn', True),
            use_dueling=self.config.get('use_dueling', True)
        )
    
    def _setup_directories(self):
        """Setup results and output directories."""
        self.results_dir = self.config.get('results_dir', 'results_curriculum')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create curriculum-specific subdirectories
        self.curriculum_dir = os.path.join(self.results_dir, 'curriculum')
        self.models_dir = os.path.join(self.results_dir, 'models')
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        
        for dir_path in [self.curriculum_dir, self.models_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
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
            'curriculum_stages': [],
            'stage_episodes': [],
            'stage_progressions': [],
            'evaluation_scores': []
        }
        
        # Best model tracking
        self.best_eval_score = float('-inf')
        self.episodes_without_improvement = 0
    
    def _print_initialization_summary(self):
        """Print trainer initialization summary."""
        print(f"🧠 Curriculum Learning GNN-MARL Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Total episodes: {self.num_episodes}")
        print(f"   Max steps per episode: {self.max_steps_per_episode}")
        print(f"   Results directory: {self.results_dir}")
        print(f"   Curriculum stages: {len(self.curriculum_manager.curriculum_stages)}")
    
    def create_environment(self, seed: Optional[int] = None) -> MountainRescueModel:
        """
        Create a new training environment using current curriculum stage parameters.
        
        Args:
            seed: Random seed for environment initialization
            
        Returns:
            Configured MountainRescueModel instance
        """
        env_config = self.curriculum_manager.get_environment_config()
        
        model = MountainRescueModel(
            width=env_config['width'],
            height=env_config['height'],
            n_robots=env_config['n_robots'],
            n_drones=env_config['n_drones'],
            n_persons=env_config['n_persons'],
            mode="novel",
            seed=seed,
            spawn_interval=env_config['spawn_interval'],
            max_persons=env_config['max_persons'],
            use_gnn_marl=True,
            device=self.device,
            quiet=self.config.get('quiet', False)
        )
        
        # Initialize GNN-MARL system if not present
        if not hasattr(model, 'gnn_marl_system') or model.gnn_marl_system is None:
            model.gnn_marl_system = GNN_MARL_System(
                self.gnn_config, 
                self.marl_config, 
                device=self.device
            )
        
        # Move GNN-MARL system to device
        model.gnn_marl_system.to(self.device)
        
        return model
    
    def calculate_episode_metrics(self, model: MountainRescueModel) -> Dict[str, float]:
        """
        Calculate comprehensive episode metrics.
        
        Args:
            model: The model to calculate metrics for
            
        Returns:
            Dictionary containing episode metrics
        """
        # Calculate rescue rate
        total_persons = len([agent for agent in model.agents if isinstance(agent, Person)])
        rescued_persons = model.rescued_count
        rescue_rate = rescued_persons / max(total_persons, 1)
        
        # Calculate communication efficiency
        total_messages = sum(getattr(agent, 'messages_sent', 0) for agent in model.agents)
        communication_efficiency = total_messages / max(rescued_persons, 1) if rescued_persons > 0 else 0
        
        # Calculate episode reward (simplified)
        episode_reward = rescued_persons * 100 - total_messages * 2
        
        return {
            'rescue_rate': rescue_rate,
            'rescued_persons': rescued_persons,
            'total_persons': total_persons,
            'communication_efficiency': communication_efficiency,
            'episode_reward': episode_reward,
            'total_messages': total_messages
        }
    
    def run_episode(self, model: MountainRescueModel, episode: int) -> Dict[str, float]:
        """
        Run a single training episode.
        
        Args:
            model: The model to run
            episode: Current episode number
            
        Returns:
            Dictionary containing episode metrics
        """
        model.gnn_marl_system.enable_training()
        
        for step in range(self.max_steps_per_episode):
            try:
                model.step()
                
                # Check for early termination conditions
                if hasattr(model, 'all_persons_rescued') and model.all_persons_rescued():
                    break
                    
            except Exception as e:
                print(f"⚠️ Error in episode {episode}, step {step}: {e}")
                break
        
        # Calculate episode metrics
        metrics = self.calculate_episode_metrics(model)
        
        # Update training metrics
        self.training_metrics['episode_rewards'].append(metrics['episode_reward'])
        self.training_metrics['episode_lengths'].append(step + 1)
        self.training_metrics['rescue_rates'].append(metrics['rescue_rate'])
        self.training_metrics['communication_efficiency'].append(metrics['communication_efficiency'])
        self.training_metrics['curriculum_stages'].append(self.curriculum_manager.current_stage)
        self.training_metrics['stage_episodes'].append(self.curriculum_manager.stage_episode_count)
        
        # Extract training losses if available
        if hasattr(model.gnn_marl_system, 'training_history'):
            history = model.gnn_marl_system.training_history
            if history['robot_losses']:
                self.training_metrics['robot_losses'].append(history['robot_losses'][-1])
            if history['drone_losses']:
                self.training_metrics['drone_losses'].append(history['drone_losses'][-1])
        
        return metrics
    
    def evaluate_performance(self, episode: int) -> Dict[str, float]:
        """
        Evaluate model performance on current curriculum stage.
        
        Args:
            episode: Current episode number
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"📊 Evaluating performance at episode {episode}")
        
        # Create evaluation environment
        eval_model = self.create_environment(seed=42)
        eval_model.gnn_marl_system.disable_training()
        
        eval_metrics = []
        num_eval_episodes = 5
        
        for eval_ep in range(num_eval_episodes):
            # Reset environment
            eval_model = self.create_environment(seed=42 + eval_ep)
            eval_model.gnn_marl_system = eval_model.gnn_marl_system
            eval_model.gnn_marl_system.disable_training()
            
            # Run evaluation episode
            for step in range(self.max_steps_per_episode):
                try:
                    eval_model.step()
                    if hasattr(eval_model, 'all_persons_rescued') and eval_model.all_persons_rescued():
                        break
                except Exception as e:
                    print(f"⚠️ Evaluation error: {e}")
                    break
            
            # Calculate evaluation metrics
            metrics = self.calculate_episode_metrics(eval_model)
            eval_metrics.append(metrics)
        
        # Average evaluation metrics
        avg_metrics = {
            'eval_rescue_rate': np.mean([m['rescue_rate'] for m in eval_metrics]),
            'eval_episode_reward': np.mean([m['episode_reward'] for m in eval_metrics]),
            'eval_communication_efficiency': np.mean([m['communication_efficiency'] for m in eval_metrics]),
            'eval_rescued_persons': np.mean([m['rescued_persons'] for m in eval_metrics])
        }
        
        self.training_metrics['evaluation_scores'].append(avg_metrics['eval_rescue_rate'])
        
        print(f"   Evaluation rescue rate: {avg_metrics['eval_rescue_rate']:.3f}")
        print(f"   Evaluation reward: {avg_metrics['eval_episode_reward']:.1f}")
        
        return avg_metrics
    
    def save_model(self, model: MountainRescueModel, episode: int, is_best: bool = False) -> None:
        """
        Save model checkpoint with curriculum state.
        
        Args:
            model: The model to save
            episode: Current episode number
            is_best: Whether this is the best model so far
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model state
        model_filename = f"curriculum_gnn_marl_ep{episode}_{timestamp}.pth"
        if is_best:
            model_filename = f"best_curriculum_gnn_marl_ep{episode}_{timestamp}.pth"
        
        model_path = os.path.join(self.models_dir, model_filename)
        
        checkpoint = {
            'episode': episode,
            'gnn_state_dict': model.gnn_marl_system.gnn.state_dict(),
            'robot_policy_state_dict': model.gnn_marl_system.robot_policy.policy_net.state_dict(),
            'drone_policy_state_dict': model.gnn_marl_system.drone_policy.policy_net.state_dict(),
            'gnn_config': self.gnn_config,
            'marl_config': self.marl_config,
            'curriculum_metrics': self.curriculum_manager.get_curriculum_metrics(),
            'training_metrics': self.training_metrics,
            'timestamp': timestamp
        }
        
        torch.save(checkpoint, model_path)
        print(f"💾 Model saved: {model_path}")
    
    def save_metrics(self) -> None:
        """Save training and curriculum metrics to JSON files."""
        # Save training metrics
        metrics_path = os.path.join(self.results_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        # Save curriculum metrics
        curriculum_metrics_path = os.path.join(self.curriculum_dir, 'curriculum_metrics.json')
        with open(curriculum_metrics_path, 'w') as f:
            json.dump(self.curriculum_manager.get_curriculum_metrics(), f, indent=2)
    
    def create_curriculum_plots(self) -> None:
        """Create comprehensive curriculum learning visualization plots."""
        if not self.training_metrics['episode_rewards']:
            return
        
        # Create curriculum progress plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        episodes = range(len(self.training_metrics['episode_rewards']))
        
        # Plot 1: Episode rewards with curriculum stages
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.training_metrics['episode_rewards'], alpha=0.7, label='Episode Reward')
        
        # Add curriculum stage background colors
        stage_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        current_stage = 0
        stage_start = 0
        
        for i, stage in enumerate(self.training_metrics['curriculum_stages']):
            if stage != current_stage:
                # Draw background for previous stage
                ax1.axvspan(stage_start, i, alpha=0.3, color=stage_colors[current_stage % len(stage_colors)])
                ax1.text(stage_start + (i - stage_start) / 2, ax1.get_ylim()[1] * 0.9, 
                        f'Stage {current_stage + 1}', ha='center', va='center')
                current_stage = stage
                stage_start = i
        
        # Draw final stage
        ax1.axvspan(stage_start, len(episodes), alpha=0.3, color=stage_colors[current_stage % len(stage_colors)])
        ax1.text(stage_start + (len(episodes) - stage_start) / 2, ax1.get_ylim()[1] * 0.9, 
                f'Stage {current_stage + 1}', ha='center', va='center')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Episode Rewards Across Curriculum Stages')
        ax1.legend()
        
        # Plot 2: Rescue rates with curriculum stages
        ax2 = axes[0, 1]
        ax2.plot(episodes, self.training_metrics['rescue_rates'], alpha=0.7, label='Rescue Rate', color='green')
        
        # Add moving average
        window_size = 20
        if len(self.training_metrics['rescue_rates']) >= window_size:
            moving_avg = np.convolve(self.training_metrics['rescue_rates'], 
                                   np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size-1, len(episodes)), moving_avg, 
                    label=f'Moving Average ({window_size})', color='darkgreen', linewidth=2)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Rescue Rate')
        ax2.set_title('Rescue Rate Progress')
        ax2.legend()
        
        # Plot 3: Neural network losses
        ax3 = axes[1, 0]
        if self.training_metrics['robot_losses']:
            ax3.plot(episodes, self.training_metrics['robot_losses'], alpha=0.7, label='Robot Loss', color='red')
        if self.training_metrics['drone_losses']:
            ax3.plot(episodes, self.training_metrics['drone_losses'], alpha=0.7, label='Drone Loss', color='blue')
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Training Loss')
        ax3.set_title('Neural Network Training Losses')
        ax3.legend()
        ax3.set_yscale('log')
        
        # Plot 4: Communication efficiency
        ax4 = axes[1, 1]
        ax4.plot(episodes, self.training_metrics['communication_efficiency'], alpha=0.7, color='purple')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Messages per Rescue')
        ax4.set_title('Communication Efficiency')
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'curriculum_learning_progress.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Curriculum plots saved: {plot_path}")
    
    def train(self) -> None:
        """
        Main training loop with curriculum learning.
        """
        print("🚀 Starting curriculum learning training...")
        
        # Initialize model with first curriculum stage
        model = self.create_environment(seed=42)
        
        for episode in range(self.num_episodes):
            episode_start_time = time.time()
            
            # Create new environment for episode (ensures clean state)
            model = self.create_environment(seed=episode)
            
            # Transfer existing GNN-MARL system
            if hasattr(model, 'gnn_marl_system'):
                model.gnn_marl_system.episode_count = episode
            
            # Run episode
            episode_metrics = self.run_episode(model, episode)
            
            # Update curriculum manager with episode performance
            self.curriculum_manager.update_performance(episode_metrics)
            
            # Check for curriculum progression
            if self.curriculum_manager.should_progress_stage():
                self.curriculum_manager.progress_to_next_stage()
                self.training_metrics['stage_progressions'].append(episode)
                
                # Save model at stage progression
                self.save_model(model, episode, is_best=False)
            
            # Check for curriculum regression
            elif self.curriculum_manager.should_regress_stage():
                self.curriculum_manager.regress_to_previous_stage()
            
            # Print progress
            if episode % 10 == 0:
                episode_time = time.time() - episode_start_time
                stage_config = self.curriculum_manager.get_current_stage_config()
                print(f"Episode {episode:4d} | Stage: {stage_config['name']} | "
                      f"Rescue Rate: {episode_metrics['rescue_rate']:.3f} | "
                      f"Reward: {episode_metrics['episode_reward']:6.1f} | "
                      f"Time: {episode_time:.2f}s")
            
            # Evaluation
            if episode % self.eval_interval == 0 and episode > 0:
                eval_metrics = self.evaluate_performance(episode)
                
                # Check for best model
                if eval_metrics['eval_rescue_rate'] > self.best_eval_score:
                    self.best_eval_score = eval_metrics['eval_rescue_rate']
                    self.episodes_without_improvement = 0
                    self.save_model(model, episode, is_best=True)
                else:
                    self.episodes_without_improvement += self.eval_interval
            
            # Save checkpoint
            if episode % self.save_interval == 0 and episode > 0:
                self.save_model(model, episode)
                self.save_metrics()
                self.create_curriculum_plots()
            
            # Early stopping
            if self.episodes_without_improvement >= self.early_stopping_patience:
                print(f"🛑 Early stopping triggered after {self.episodes_without_improvement} episodes without improvement")
                break
        
        # Final save and evaluation
        self.save_model(model, episode)
        self.save_metrics()
        self.create_curriculum_plots()
        
        # Print final curriculum statistics
        curriculum_metrics = self.curriculum_manager.get_curriculum_metrics()
        print(f"\n🎯 Curriculum Learning Complete!")
        print(f"   Total episodes: {curriculum_metrics['total_episodes']}")
        print(f"   Final stage: {curriculum_metrics['current_stage'] + 1}/{len(self.curriculum_manager.curriculum_stages)}")
        print(f"   Stage progressions: {len(self.training_metrics['stage_progressions'])}")
        print(f"   Regressions: {curriculum_metrics['regression_count']}")
        print(f"   Best rescue rate: {self.best_eval_score:.3f}")
        print(f"   Training time: {datetime.now() - self.training_start_time}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for curriculum learning training."""
    parser = argparse.ArgumentParser(description='Curriculum Learning GNN-MARL Training')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU usage')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--results-dir', type=str, default='results_curriculum', help='Results directory')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose agent and model output')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_curriculum_config()
    
    # Override with command line arguments
    config['num_episodes'] = args.episodes
    config['force_cpu'] = args.cpu_only
    config['learning_rate'] = args.learning_rate
    config['hidden_dim'] = args.hidden_dim
    config['batch_size'] = args.batch_size
    config['results_dir'] = args.results_dir
    config['quiet'] = args.quiet
    
    # Create and run trainer
    trainer = CurriculumGNNMARLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()