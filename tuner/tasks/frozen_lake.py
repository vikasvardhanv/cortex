import re
import asyncio
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

import agentscope.tuner as tuner
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from tuner.registry import TaskRegistry

# Placeholder for FrozenLakeEnv and utilities
# (In a real implementation, these would be in separate utility files)

SYSTEM_PROMPT = (
    "You are an agent navigating a Frozen Lake. "
    "Your goal is to reach the Goal (G) by moving across Frozen (F) tiles, "
    "avoiding Holes (H). You are represented as 'P' in the grid. "
    "Output your next move: 'up', 'down', 'left', 'right' within ``` ```."
)

VALID_ACTIONS = ["up", "down", "left", "right"]

class FrozenLakeTaskAgent(ReActAgent):
    """Refinement of ReActAgent for Frozen Lake Navigation."""
    def __init__(self, model: OpenAIChatModel):
        super().__init__(
            name="navigator",
            model=model,
            sys_prompt=SYSTEM_PROMPT,
            formatter=OpenAIChatFormatter(),
            max_iters=1,
        )

    def get_action(self, msg: Msg) -> str:
        content = msg.get_text_content()
        matches = re.findall(r"```(.*?)```", content, re.DOTALL)
        if matches:
            action = matches[-1].strip().lower()
            if action in VALID_ACTIONS:
                return action
        return "still"

@TaskRegistry.register(
    name="frozen_lake", 
    description="Train an agent to navigate the Frozen Lake environment.",
    model="Qwen2.5-3B-Instruct"
)
async def run_frozen_lake_workflow(
    task: Dict,
    model: OpenAIChatModel,
    auxiliary_models: Dict[str, OpenAIChatModel] | None = None,
) -> tuner.WorkflowOutput:
    """Full Frozen Lake RL navigation workflow with reward logic."""
    
    # Initialize Agent
    agent = FrozenLakeTaskAgent(model=model)
    
    # Environment Setup (Simplified simulation for this walkthrough)
    # In a production setting, this would interact with a gymnasium environment
    grid_observation = task.get("observation", "P _ _\n_ O _\n_ _ G")
    
    response = await agent.reply(
        msg=Msg("user", f"Current grid:\n{grid_observation}\nWhat is your next move?", role="user"),
    )
    
    action = agent.get_action(response)
    
    # Mock Reward for illustration
    reward = 0.0
    if action in ["right", "down"]: # Simplistic "towards goal" reward
        reward = 0.1
    
    return tuner.WorkflowOutput(
        response=response,
        reward=reward,
        metrics={"action": action}
    )
