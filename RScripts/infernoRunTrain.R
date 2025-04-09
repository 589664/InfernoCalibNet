# Load Inferno package
library("inferno")

# Base path (relative to project root)
relative_path <- "data/refined/multilabel/inferno"

# Full file paths
input_file    <- file.path(relative_path, "calibration_sampled.csv")
metadata_file <- file.path(relative_path, "inferno_metadata.csv")
output_dir    <- file.path(relative_path, "inferno_effusion")

# Define parallelism and seed
parallel_cores <- 7
random_seed    <- 42

# Train Inferno model
inferno_model <- learn(
  data = input_file,
  metadata = metadata_file,
  outputdir = output_dir,
  parallel = parallel_cores,
  appendinfo = FALSE,
  appendtimestamp = FALSE,
  # maxhours = 0,        # set 0 for quick test
  # subsampledata = 100, # use subset for faster testing
  seed = random_seed
)
