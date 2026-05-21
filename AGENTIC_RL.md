# 🚀 Agentic RL with Cortex CEM

Train your agentic engineering application seamlessly with Reinforcement Learning. Cortex now integrates **AgentScope** and **Trinity-RFT** to support advanced Agentic RL workflows.

## 🛠️ Getting Started

### 1. Installation
The environment is being set up in your `venv` with all necessary dependencies:
```bash
# Verify installation
cortex-tuner --help
```

### 2. Available Training Tasks
You can list all pre-configured Agentic RL tasks inspired by the AgentScope ecosystem:
```bash
cortex-tuner --list
```

| Task | Description | Default Model |
| :--- | :--- | :--- |
| **math_agent** | Tune a math-solving agent with multi-step reasoning. | Qwen3-0.6B |
| **frozen_lake** | Train an agent to navigate the Frozen Lake environment. | Qwen2.5-3B-Instruct |
| **learn_to_ask** | Tune agents using LLM-as-a-judge for automated feedback. | Qwen2.5-7B-Instruct |
| **email_search** | Improve tool-use capabilities without labeled ground truth. | Qwen3-4B-Instruct-2507 |
| **werewolf_game** | Strategic multi-agent game interactions. | Qwen2.5-7B-Instruct |
| **data_augment** | Generate synthetic training data for better tuning. | Qwen3-0.6B |

### 3. Start Training
To start training the math agent on the GSM8K dataset:
```bash
cortex-tuner math_agent --dataset openai/gsm8k
```

## 🏗️ Creating Custom Models
Cortex is designed to be "independent" — you can easily add new models and tasks.

1. **Create a new task file** in `cortex/tuner/tasks/`.
2. **Use the `@TaskRegistry.register` decorator** to expose it.
3. **Define your workflow** (agent logic) and **judge function** (reward logic).

Example:
```python
from tuner.registry import TaskRegistry

@TaskRegistry.register(name="nozzle_optimizer", description="Tune for aerospace engineering")
async def nozzle_workflow(task, model):
    # Your custom agentic RL logic here
    pass
```

## 📈 Monitoring
Training progress is automatically logged. You can view reward curves and rollout metrics in the `checkpoints/` directory.
