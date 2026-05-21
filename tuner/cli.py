import argparse
import sys
import asyncio
from tuner.registry import TaskRegistry, run_tuning
# Import tasks to ensure they are registered
import tuner.tasks

def main():
    parser = argparse.ArgumentParser(description="Cortex Agentic RL Trainer")
    parser.add_argument("task", nargs="?", help="Task name to train (e.g., math_agent)")
    parser.add_argument("--list", action="store_true", help="List all available tasks")
    parser.add_argument("--dataset", help="Path to the training dataset")

    args = parser.parse_args()

    if args.list:
        tasks = TaskRegistry.list_tasks()
        print("\nAvailable Agentic RL Tasks:")
        print("-" * 60)
        print(f"{'Task Name':<20} | {'Model':<20} | {'Description'}")
        print("-" * 60)
        for name, info in tasks.items():
            print(f"{name:<20} | {info['model']:<20} | {info['description']}")
        return

    if not args.task:
        parser.print_help()
        return

    if args.task:
        if not args.dataset and args.task == "math_agent":
            # Default for math agent if nothing provided
            dataset_path = "openai/gsm8k"
        else:
            dataset_path = args.dataset

        run_tuning(args.task, dataset_path)

if __name__ == "__main__":
    main()
