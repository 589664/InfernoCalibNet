from InquirerPy import inquirer
from rich import print
from src.train_optimize import run_training, optimize_hyperparams, run_testing

if __name__ == "__main__":
    mode = inquirer.select(
        message="Choose mode:",
        choices=["train", "optimize", "test", "exit"],
    ).execute()

    if mode == "optimize":
        optimize_hyperparams()
    elif mode == "train":
        run_training()
    elif mode == "test":
        run_testing()
    elif mode == "exit":
        print("[yellow]Exiting.")
    else:
        print("[red]Invalid mode selected.")
