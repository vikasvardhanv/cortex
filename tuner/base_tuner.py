import agentscope.tuner as tuner
from typing import Callable, Dict, Any, Optional

class CortexTuner:
    """
    A high-level wrapper for AgentScope's Agentic RL Tuner.
    Designed for seamless integration into the Cortex CEM project.
    """
    def __init__(self, 
                 model_path: str = "Qwen/Qwen3-0.6B", 
                 algorithm_type: str = "multi_step_grpo"):
        self.model_config = tuner.TunerModelConfig(
            model_path=model_path,
            max_model_len=24576,
            max_tokens=16384,
            temperature=1.0,
            inference_engine_num=4,
            tensor_parallel_size=1,
        )
        self.algorithm_config = tuner.AlgorithmConfig(
            algorithm_type=algorithm_type,
            group_size=8,
            learning_rate=1e-6,
            batch_size=32,
        )

    def train(self, 
              workflow_func: Callable, 
              judge_func: Callable, 
              dataset_path: str, 
              dataset_name: str = "main", 
              split: str = "train"):
        """
        Starts the training/tuning process.
        """
        dataset = tuner.DatasetConfig(
            path=dataset_path,
            name=dataset_name,
            split=split,
        )

        print(f"Starting Agentic RL Training for {dataset_path}...")
        tuner.tune(
            workflow_func=workflow_func,
            judge_func=judge_func,
            train_dataset=dataset,
            model=self.model_config,
            algorithm=self.algorithm_config,
        )
        print("Training complete!")

if __name__ == "__main__":
    # Example usage (simplified)
    # tuner = CortexTuner()
    # tuner.train(...)
    pass
