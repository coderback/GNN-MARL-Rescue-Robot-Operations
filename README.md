# Mountain Rescue Simulation - GNN-MARL Solution

## Overview
This project implements a Graph Neural Network-based Multi-Agent Reinforcement Learning (GNN-MARL) system for mountain rescue operations. The system coordinates FirstAidRobots, ExplorerDrones, and MobileChargers to efficiently locate and rescue people in mountainous terrain.

## Quick Start

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Solution

#### Basic Training
```bash
python src/train_gnn_marl.py
```

#### Advanced Training with Curriculum Learning
```bash
python src/train_gnn_marl_cl_strategy.py
```

#### Running Tests
```bash
# Basic functionality tests
python tests/tests_basic.py

# Extended mode tests  
python tests/tests_extended.py

# Novel mode tests
python tests/tests_novel.py
```

#### Server Mode (Interactive)
```bash
solara src/server.py
```

## Activity Solutions

### Basic Multi-Agent System
- **Statechart Diagrams**: `Statechart Diagrams/Agent Statechart Diagrams.pdf` - Visual representation of agent state machines and transitions
- **Implementation**: `src/agents.py` - Basic agent classes with state machines
- **Core Features**: Random exploration, battery management, rescue operations
- **Test**: `tests/tests_basic.py`

### Extended Coordination System  
- **Implementation**: `src/messaging.py` + extended agent behaviors
- **Core Features**: Inter-agent communication, mission coordination, intelligent task assignment
- **Slides**: `Slides/Activity 2 - Game Theory Mountain Rescue Operation.pptx`
- **Test**: `tests/tests_extended.py`

### Novel GNN-MARL Implementation
- **Implementation**: `src/gnn_marl.py` - Complete GNN-MARL system
- **Core Features**: 
  - Dynamic graph construction from agent states
  - Graph Neural Networks with attention mechanisms
  - Multi-agent reinforcement learning with actor-critic networks
  - Mobile charging system
  - Curriculum learning
- **Slides**: `Slides/Activity 4 - Novel Mode.pptx`
- **Training**: `src/train_gnn_marl.py`, `src/train_gnn_marl_cl_strategy.py`
- **Test**: `tests/tests_novel.py`
- **Models**: `results/best_gnn_marl_model.pth`, `results_curriculum/best_gnn_marl_cl_model.pth`

### Performance Analysis
- **Analysis Notebooks**: 
  - `KPI Analysis/Activity_3_KPI_Analysis.ipynb`
  - `KPI Analysis/Activity_4_KPI_Analysis.ipynb`
- **Metrics**: Rescue efficiency, communication overhead, battery utilization
- **Results**: `results/training_metrics.json`, `results_curriculum/plots/training_progress.png`



## Key Features

### Agent Types
- **FirstAidRobot**: Ground-based rescue with terrain navigation
- **ExplorerDrone**: Aerial reconnaissance and coordination
- **Person**: Rescue targets with urgency calculations
- **MobileCharger**: Mobile battery charging (Novel mode)

### Operation Modes
- **Basic**: Simple random exploration
- **Extended**: Coordinated missions with communication
- **Novel**: GNN-MARL with mobile charging agents and curriculum learning

### Advanced Features
- Multi-criteria decision making
- Dynamic graph neural networks
- Curriculum learning strategies
- Real-time performance monitoring

## Results
The best trained model achieves:
- Efficient resource utilization
- Robust coordination under varying conditions
- Adaptive learning through curriculum strategies

## File Structure
```
├── src/                    # Core implementation
├── tests/                  # Test suites
├── results/               # Trained models and metrics
├── KPI Analysis/          # Performance analysis notebooks
├── Slides/               # Presentation materials
├── Statechart Diagrams/  # Agent behavior diagrams
└── requirements.txt      # Dependencies
```

## Notes
- The system supports both CPU and GPU training
- Curriculum learning significantly improves convergence and training time