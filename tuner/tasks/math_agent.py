from typing import Dict, Any
import agentscope.tuner as tuner
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from tuner.registry import TaskRegistry

# Register this task in the Cortex Tuner registry
@TaskRegistry.register(
    name="math_agent", 
    description="Tune a math-solving agent with multi-step reasoning.",
    model="Qwen3-0.6B"
)
async def run_math_workflow(
    task: Dict,
    model: OpenAIChatModel,
    auxiliary_models: Dict[str, OpenAIChatModel] | None = None,
) -> tuner.WorkflowOutput:
    """A math problem-solving workflow using ReActAgent."""
    
    sys_prompt = (
        "You are an agent specialized in solving math problems with tools. "
        "Please solve the math problem given to you. You can write and "
        "execute Python code to perform calculation or verify your answer. "
        "You should return your final answer within \\boxed{{}}."
    )
    
    agent = ReActAgent(
        name="math_expert",
        sys_prompt=sys_prompt,
        model=model,
        enable_meta_tool=True,
        formatter=OpenAIChatFormatter(),
    )
    
    response = await agent.reply(
        msg=Msg("user", task["question"], role="user"),
    )
    
    return tuner.WorkflowOutput(response=response)

async def math_judge(
    task: Dict,
    response: Msg,
    auxiliary_models: Dict[str, OpenAIChatModel] | None = None,
) -> tuner.JudgeOutput:
    """Judge function for math reasoning (GSM8K)."""
    try:
        from trinity.common.rewards.math_reward import MathBoxedRewardFn
    except ImportError:
        # Fallback if trinity-rft internal rewards are missing
        return tuner.JudgeOutput(reward=1.0 if task["answer"] in response.get_text_content() else 0.0)

    reward_fn = MathBoxedRewardFn()
    truth = task["answer"]
    
    # GSM8K standard parsing
    if isinstance(truth, str) and "####" in truth:
        truth = truth.split("####")[1].strip()
    
    result = response.get_text_content()
    reward_dict = reward_fn(response=result, truth=truth)
    
    return tuner.JudgeOutput(
        reward=sum(reward_dict.values()),
        metrics=reward_dict,
    )
