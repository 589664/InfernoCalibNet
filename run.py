from InquirerPy import inquirer
from rich import print
from src.train_optimize import (
    run_training,
    optimize_hyperparams,
    run_testing,
    log_tensorboard_graph,
)

from src.testing import predict_on_image

if __name__ == "__main__":
    mode = inquirer.select(
        message="Choose mode:",
        choices=["train", "optimize", "test", "print model", "predict image", "exit"],
    ).execute()

    if mode == "optimize":
        optimize_hyperparams()
    elif mode == "train":
        run_training()
    elif mode == "test":
        run_testing()
    elif mode == "print model":
        log_tensorboard_graph()

    elif mode == "predict image":
        # Predict on a single image
        image_name = "00001946_006.png"
        predictions = predict_on_image(image_name)

        # Print the predictions
        for disease, probability in predictions.items():
            print(f"{disease}: {probability:.4f}")

    elif mode == "exit":
        print("[yellow]Exiting.")

    else:
        print("[red]Invalid mode selected.")
