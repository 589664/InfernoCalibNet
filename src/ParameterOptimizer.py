from InquirerPy import inquirer
from rich.console import Console
import config


class ParamOpt:
    def __init__(self, ds_size=None, gpu_mem=None, img_size=None):
        # Assign values from config or use defaults if available, otherwise use a hardcoded default
        self.ds_size = ds_size if ds_size else getattr(config, "DATASET_SIZE", 120000)
        self.gpu_mem = gpu_mem if gpu_mem else getattr(config, "GPU_MEMORY", 16)
        self.img_size = (
            img_size if img_size else getattr(config, "INPUT_IMAGE_SIZE", (300, 300))
        )
        self.bs = None
        self.epochs = None
        self.workers = None
        self.train_size = None
        self.val_size = None
        self.test_size = None
        self.console = Console()
        self.efficientnet_models = {
            "EfficientNet B0": (224, 224),
            "EfficientNet B1": (240, 240),
            "EfficientNet B2": (260, 260),
            "EfficientNet B3": (300, 300),
            "EfficientNet B4": (380, 380),
            "EfficientNet B5": (456, 456),
            "EfficientNet B6": (528, 528),
            "EfficientNet B7": (600, 600),
        }

    def calc_params(self):
        # Estimating appropriate batch size based on GPU memory and image size
        self.console.print(
            "[bold green]Calculating optimal parameters based on provided hardware and dataset information...[/bold green]"
        )
        img_mem = (
            self.img_size[0] * self.img_size[1] * 1 * 4 / (1024**2)
        )  # Image memory in MB (assuming float32)
        bs = int(
            self.gpu_mem / (img_mem * 2.5)
        )  # A factor to leave enough space for activations and overhead
        bs = min(max(bs, 1), 128)  # Limiting batch size between 1 and 128

        # Setting epochs based on dataset size
        epochs = 50 if self.ds_size >= 50000 else 100

        # Setting num_workers based on CPU cores
        workers = (
            8
            if getattr(config, "CPU_CORES", 8) > 8
            else getattr(config, "CPU_CORES", 8) // 2
        )

        # Split dataset into training, validation, and testing
        self.train_size = int(self.ds_size * 0.7)
        self.val_size = int(self.ds_size * 0.15)
        self.test_size = self.ds_size - self.train_size - self.val_size

        self.bs = bs
        self.epochs = epochs
        self.workers = workers

        return {
            "batch_size": self.bs,
            "epochs": self.epochs,
            "workers": self.workers,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
        }

    def user_input(self):
        self.console.print(
            "[bold blue]Interactive Setup for Parameter Optimization[/bold blue]"
        )
        ds_size = inquirer.number(
            message="Enter the dataset size (number of images):", default=self.ds_size
        ).execute()
        gpu_mem = inquirer.number(
            message="Enter the available GPU memory in GB:", default=self.gpu_mem
        ).execute()
        img_size = inquirer.select(
            message="Select input image size:",
            choices=[
                f"{k} ({v[0]}x{v[1]})" for k, v in self.efficientnet_models.items()
            ],
            default=f"EfficientNet B3 ({self.img_size[0]}x{self.img_size[1]})",
        ).execute()

        # Parse selected image size
        selected_model = [k for k in self.efficientnet_models if k in img_size][0]
        self.img_size = self.efficientnet_models[selected_model]

        # Update object attributes with user inputs
        self.ds_size = int(ds_size)
        self.gpu_mem = float(gpu_mem)

    def display_recommendations(self):
        params = self.calc_params()
        self.console.print("[bold yellow]Recommended Parameters:[/bold yellow]")
        self.console.print(f"Batch Size: [green]{params['batch_size']}[/green]")
        self.console.print(f"Epochs: [green]{params['epochs']}[/green]")
        self.console.print(f"Number of Workers: [green]{params['workers']}[/green]")
        self.console.print(
            f"Training Set Size: [green]{params['train_size']} ({params['train_size'] / self.ds_size * 100:.2f}%)[/green]"
        )
        self.console.print(
            f"Validation Set Size: [green]{params['val_size']} ({params['val_size'] / self.ds_size * 100:.2f}%)[/green]"
        )
        self.console.print(
            f"Testing Set Size: [green]{params['test_size']} ({params['test_size'] / self.ds_size * 100:.2f}%)[/green]"
        )


if __name__ == "__main__":
    # Initialize the ParamOpt with config defaults or user inputs
    opt = ParamOpt()
    opt.user_input()  # Optional, to let user adjust the parameters interactively
    opt.display_recommendations()
