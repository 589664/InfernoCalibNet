# Load Inferno package
library("inferno")

#=======================================================================================================================
# 📁 Setup Paths and Configuration
#=======================================================================================================================

# Base path (relative to project root)
relative_path <- "data/refined/multilabel/inferno"

# Full file paths
data_file_path <- file.path(relative_path, "calibration_train.csv")
data_file      <- read.csv(data_file_path, na.strings = "", stringsAsFactors = FALSE, tryLogical = FALSE)
metadata_file  <- file.path(relative_path, "metadata_effusion.csv")
output_dir     <- file.path(relative_path, "effusion")

# Parallelism and seed
parallel_cores <- 7
random_seed    <- 42

#=======================================================================================================================
# 🏋️ Train Inferno Model for Effusion
#=======================================================================================================================

inferno_model <- learn(
  data = data_file,
  metadata = metadata_file,
  outputdir = output_dir,
  parallel = parallel_cores,
  appendinfo = FALSE,
  appendtimestamp = FALSE,
  # maxhours = 0,        # uncomment for quick test
  # subsampledata = 100, # uncomment to train on a small subset
  seed = random_seed
)