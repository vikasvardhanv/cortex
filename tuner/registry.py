from typing import Dict, Callable, Any
import agentscope.tuner as tuner
import mlflow

class TaskRegistry:
    _tasks = {}

    @classmethod
    def register(cls, name: str, description: str, model: str):
        def decorator(func):
            cls._tasks[name] = {
                "func": func,
                "description": description,
                "model": model
            }
            return func
        return decorator

    @classmethod
    def get_task(cls, name: str):
        return cls._tasks.get(name)

    @classmethod
    def list_tasks(cls):
        return cls._tasks

def run_tuning(task_name: str, dataset_path: str):
    task_info = TaskRegistry.get_task(task_name)
    if not task_info:
        print(f"Task {task_name} not found.")
        return

    print(f"Starting {task_name}: {task_info['description']}")
    
    # We define a generic trainer for now that pulls the specific func from info
    # The registration decorator should have captured the workflow func
    workflow_func = task_info["func"]

    # We use some default judge func if not specified in the task, but for now
    # every task should have its own workflow and rewards
    
    dataset = tuner.DatasetConfig(
        path=dataset_path,
        split="train"
    )

    model_config = tuner.TunerModelConfig(
        model_path=task_info.get("model_path", "Qwen/Qwen2.5-7B-Instruct"),
        max_model_len=2048,
        max_tokens=512,
        temperature=1.0,
    )
    
    algorithm_config = tuner.AlgorithmConfig(
        algorithm_type="multi_step_grpo",
        group_size=8,
        learning_rate=1e-6,
        batch_size=32,
    )

    mlflow.set_experiment(f"Cortex-{task_name}")
    with mlflow.start_run():
        mlflow.log_params({
            "task": task_name,
            "base_model": model_config.model_path,
            "algorithm": algorithm_config.algorithm_type,
            "learning_rate": algorithm_config.learning_rate,
            "dataset": dataset_path
        })

        tuner.tune(
            workflow_func=workflow_func,
            train_dataset=dataset,
            model=model_config,
            algorithm=algorithm_config,
        )

# Initialize registry
registry = TaskRegistry()
