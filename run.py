from rich import print
from InquirerPy import inquirer
from src.train import runTraining


if __name__ == "__main__":
    mode = inquirer.select(
        message="Choose mode:",
        choices=["train", "exit"],
    ).execute()

    if mode == "train":
        runTraining()

    elif mode == "exit":
        print("[yellow]Exiting.")

    else:
        print("[red]Invalid mode selected.")
