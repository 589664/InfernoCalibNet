from InquirerPy import inquirer
from rich.console import Console

# torch


# Initialize Rich console
console = Console()


def main():
    console.print("[bold green]Preprocessing data...[/bold green]")
    with console.status("Processing data...", spinner="dots"):

    # Define options to call methods on the pipeline manager
    options = {
        "Preprocess Data": pipeline_manager.preprocess_data,
        "Train Model": pipeline_manager.train_model,
        "Inspect Model": pipeline_manager.inspect_model,
        "Quit": None,
    }

    # Interactive menu loop
    while True:
        choice = inquirer.select(
            message="Select an action:",
            choices=list(options.keys()),
            default="Preprocess Data",
        ).execute()

        if choice == "Quit":
            console.print("[bold red]Exiting...[/bold red]")
            break
        else:
            options[choice]()


if __name__ == "__main__":
    main()
