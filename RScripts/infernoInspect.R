# Load Inferno package
library("inferno")

#=======================================================================================================================
# 📁 Setup Paths and Configuration
#=======================================================================================================================

# Base path (relative to project root)
relative_path <- "data/inferno"

# Full file paths
input_file     <- file.path(relative_path, "calibration_test.csv")
metadata_file  <- file.path(relative_path, "inferno_metadata.csv")
inferno_dir    <- file.path(relative_path, "combinedML50")

# Parallelism and seed
parallel_cores <- 7
random_seed    <- 42

#=======================================================================================================================
# 📦 Load Trained Inferno Model
#=======================================================================================================================

inferno_model <- readRDS(file.path(inferno_dir, "learnt.rds"))

# Check internal structure
str(inferno_model)

#=======================================================================================================================
# 📊 1: Predicting for a Single Random Data Point
#=======================================================================================================================

# Load full dataset
alldata <- read.csv(
  input_file,
  na.strings = "",
  stringsAsFactors = FALSE,
  tryLogical = FALSE
)

# Select one random row
total_rows <- nrow(alldata)
random_row <- alldata[sample(1:total_rows, 1), ]
print(random_row)

# Target prediction values
# target_frame <- data.frame(AGE = (50:70), VP = c("PA", "AP"))

target_frame <- expand.grid(
  LOGIT_EFFUSION = 0:1,
  VP = c("PA", "AP")
)

# target_frame <- expand.grid(
#   AGE = c(51, 72),
#   VP = c("PA", "AP")
# )

# Input features for that row
input_features <- random_row[, c("AGE", "LABEL_EFFUSION", "GENDER")]

# Quantiles for uncertainty interval
uncertainty_bounds <- c(0.055, 0.25, 0.75, 0.945)

# Predict probabilities
result_probs <- Pr(
  Y = target_frame,
  X = input_features,
  learnt = inferno_model,
  parallel = parallel_cores,
  quantile = uncertainty_bounds
)

# Visualize samples
plot(
  result_probs,
  variability = "quantiles",
  col = adjustcolor("#5195b0", alpha.f = 1),
  lwd = 1,
  grid = TRUE,
  legend = TRUE,
  xlab = "Effusion label",
  ylab = "Posterior probability"
)

# Show predicted values and true label line
print(result_probs$values)
plot(result_probs)
abline(v = random_row[, "LABEL_EFFUSION"], lty = 2, lwd = 2, col = 2)

#=======================================================================================================================
# 📊 2: Plot Posterior Distribution for All Variables
#=======================================================================================================================

# Check metadata variables
str(inferno_model$auxmetadata$name)

# Generate population-level distribution plots
plotFsamples(
  file = file.path(relative_path, "plotF_effusion"),
  learnt = inferno_model,
  data = alldata,
  plotprobability = TRUE,
  plotvariability = "samples",
  nFsamples = 50,
  parallel = parallel_cores,
  datahistogram = TRUE,
  datascatter = TRUE
)

#=======================================================================================================================
# 📈 3: Visualizing Logit → Posterior Probability Curve
#=======================================================================================================================

# Generate grid of logit values
logit_values <- vrtgrid("LOGIT_EFFUSION", learnt = inferno_model, length.out = 100)

# Keep all other features fixed
grid_features <- data.frame(
  LOGIT_EFFUSION = logit_values,
  AGE = 50,
  GENDER = "F",
  VP = "PA"
)

# Target: effusion present
target_one <- data.frame(LABEL_EFFUSION = 1)

# Run prediction
curve_probs <- Pr(
  Y = target_one,
  X = grid_features,
  learnt = inferno_model,
  nsamples = 100,
  quantiles = c(0.055, 0.945),
  parallel = parallel_cores
)

# Plot uncertainty bands
plot(
  curve_probs,
  variability = "quantiles",
  lwd = 1,
  col = adjustcolor("#2c7bb6", alpha.f = 0.5),
  xlab = "Effusion Logit",
  ylab = "Posterior probability"
)