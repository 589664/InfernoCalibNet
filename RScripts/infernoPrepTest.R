# Load Inferno package
library("inferno")

#=======================================================================================================================
# 📁 Setup Paths and Configuration
#=======================================================================================================================

# Base path (relative to project root)
relative_path <- "data/inferno"

# Parallelism and seed
parallel_cores <- 10
random_seed    <- 42

#=======================================================================================================================
# 🔄 Prepare Prior for Training
#=======================================================================================================================

# File paths
input_file  <- file.path(relative_path, "calibration_train.csv")
metadata    <- file.path(relative_path, "md_calibration_train.csv")

# Generate metadata template
metadatatemplate(
  data = input_file,
  file = metadata,
  includevrt = c("AGE", "GENDER", "VP", "LOGIT_EFFUSION", "LABEL_EFFUSION", "LOGIT_ATELECTASIS", "LABEL_ATELECTASIS")
)
