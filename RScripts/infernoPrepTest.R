# Load Inferno package
library("inferno")

#=======================================================================================================================
# 📁 Setup Paths and Configuration
#=======================================================================================================================

# Base path (relative to project root)
relative_path <- "data/inferno"

# Parallelism and seed
parallel_cores <- 4
random_seed    <- 42

#=======================================================================================================================
# 🔄 Prepare Prior for Effusion
#=======================================================================================================================

# File paths for Effusion
input_file_eff  <- file.path(relative_path, "calibration_train.csv")
metadata_eff    <- file.path(relative_path, "metadata_effusion.csv")
output_eff      <- file.path(relative_path, "prior_effusion")

# Generate metadata template for Effusion
metadatatemplate(
  data = input_file_eff,
  file = metadata_eff,
  includevrt = c("AGE", "GENDER", "VP", "LOGIT_EFFUSION", "LABEL_EFFUSION")
)

#=======================================================================================================================
# 🔄 Prepare Prior for Atelectasis
#=======================================================================================================================

# File paths for Atelectasis
input_file_ate    <- file.path(relative_path, "calibration_train.csv")
metadata_ate      <- file.path(relative_path, "metadata_atelectasis.csv")
output_ate        <- file.path(relative_path, "prior_atelectasis")

# Generate metadata template for Atelectasis
metadatatemplate(
  data          = input_file_ate,
  file          = metadata_ate,
  includevrt    = c("AGE", "GENDER", "VP", "LOGIT_ATELECTASIS", "LABEL_ATELECTASIS")
)

#=======================================================================================================================
# 📦 Run Prior Learning for Atelectasis change on paths demand to Effusion
#=======================================================================================================================

prior_ate_model <- learn(
  data = NULL,
  metadata = metadata_ate,
  outputdir = output_ate,
  prior = TRUE,
  output = "learnt",
  appendinfo = FALSE,
  appendtimestamp = FALSE,
  parallel = parallel_cores,
  seed = random_seed
)

# Inspect structure
str(prior_ate_model)
