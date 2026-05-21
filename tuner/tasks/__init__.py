# Cortex Tuner Tasks
from . import math_agent
from . import frozen_lake

# Placeholder registrations for other user-requested scenarios
from tuner.registry import TaskRegistry

@TaskRegistry.register("learn_to_ask", "Tune agents with LLM-as-a-judge feedback.", "Qwen2.5-7B-Instruct")
async def stub_l2a(*args, **kwargs): pass

@TaskRegistry.register("email_search", "Improve tool-use without ground truth.", "Qwen3-4B-Instruct-2507")
async def stub_email(*args, **kwargs): pass

@TaskRegistry.register("werewolf", " strategic multi-agent interactions.", "Qwen2.5-7B-Instruct")
async def stub_ww(*args, **kwargs): pass

@TaskRegistry.register("data_augment", "Generate synthetic training data.", "Qwen3-0.6B")
async def stub_da(*args, **kwargs): pass
