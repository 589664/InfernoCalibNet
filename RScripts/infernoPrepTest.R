# Load Inferno package
library("inferno")

# Base path (relative to project root)
relative_path <- "data/refined/multilabel/inferno"

# Full file paths
input_file    <- file.path(relative_path, "calibration_input_data.csv")
metadata_file <- file.path(relative_path, "inferno_metadata.csv")
output_dir    <- file.path(relative_path, "inferno_effusion")

# Generate metadata template for selected variables
metadatatemplate(
  data = input_file,
  file = metadata_file,
  includevrt = c("AGE", "GENDER", "VP", "LOGIT_EFFUSION", "LABEL_EFFUSION")
)
