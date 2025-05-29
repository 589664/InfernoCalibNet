# ----------------------------------------------------------------------------------------------------------------------
# 🌟 Single Prediction and Clinical Decision for Chest X-rays (Effusion and Atelectasis)
# ----------------------------------------------------------------------------------------------------------------------

library("inferno")
library("jsonlite")

# ----------------------------------------------------------------------------------------------------------------------
# 📂 Read Configuration from JSON
# ----------------------------------------------------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
config_path <- args[1]
config <- fromJSON(config_path)

# Extract configuration values
relative_path   <- dirname(config$model_path)
model_path      <- config$model_path
input_values    <- config$input_values
predictors      <- config$predictors
predictands     <- config$predictands

# ----------------------------------------------------------------------------------------------------------------------
# 📁 Load Model
# ----------------------------------------------------------------------------------------------------------------------
inferno_model <- readRDS(model_path)

# ----------------------------------------------------------------------------------------------------------------------
# 🔍 Setup Prediction
# ----------------------------------------------------------------------------------------------------------------------
y <- setNames(expand.grid(0:1, 0:1), as.list(predictands))
outcomenames <- apply(y, 1, function(x) paste0("E", x[1], "_A", x[2]))
pretty_outcomenames <- gsub("_", " ", outcomenames)

# Clinical actions
actions <- c(
  "Send_to_Hospital",
  "Start_Drainage_Treatment",
  "Start_Bronchodilator_Therapy",
  "Supportive_Care",
  "Observe_Closely"
)
pretty_actions <- gsub("_", " ", actions)

# Utility matrix (actions x outcomes)
utility_matrix <- matrix(
  c(
    0.2, 0.3, 0.9, 0.95,
    0.1, 0.2, 0.8, 0.85,
    0.3, 0.8, 0.4, 0.7,
    0.6, 0.5, 0.3, 0.4,
    0.7, 0.6, 0.2, 0.3
  ),
  nrow = length(actions),
  byrow = TRUE
)
rownames(utility_matrix) <- actions
colnames(utility_matrix) <- outcomenames

# Prepare data for prediction
x_patient <- as.data.frame(input_values[predictors])
true_labels <- as.data.frame(input_values[predictands])

# Run prediction with quantiles
uncertainty_bounds <- c(0.055, 0.25, 0.75, 0.945)
probs <- Pr(
  Y = y,
  X = x_patient,
  learnt = inferno_model,
  parallel = 10,
  quantile = uncertainty_bounds
)

# Expected utilities
expected_utilities <- utility_matrix %*% probs$values

# Select action with highest expected utility
choose_max_action <- function(x) {
  sample(rep(which(x == max(x)), 2), 1)
}

decision_idx <- choose_max_action(expected_utilities)
final_decision <- pretty_actions[decision_idx]

# ----------------------------------------------------------------------------------------------------------------------
# 📊 Report Results
# ----------------------------------------------------------------------------------------------------------------------
cat("\n========================================================\n")
cat("Single Patient Clinical Decision Report\n")
cat("========================================================\n")

cat("Patient Predictor Data (Features Only):\n")
print(x_patient)

cat("\nTrue Labels (Ground Truth):\n")
print(true_labels)

cat("\nPredicted Probabilities for Outcomes (%):\n")
print(data.frame(Outcome = pretty_outcomenames, Probability = round(probs$values * 100, 1)))

cat("\nExpected Utilities for Clinical Actions:\n")
print(data.frame(Action = pretty_actions, Expected_Utility = round(as.numeric(expected_utilities), 1)))

cat("\nRecommended Clinical Action:\n")
cat(paste0(final_decision, "\n"))

# ----------------------------------------------------------------------------------------------------------------------
# 🧠 Recommendation Check
# ----------------------------------------------------------------------------------------------------------------------
cat("\n🧠 Recommendation Check\n")

if (!is.null(probs$quantiles) && length(dim(probs$quantiles)) == 3) {
  for (i in seq_len(dim(probs$quantiles)[1])) {
    lower <- probs$quantiles[i, 1, 1]
    upper <- probs$quantiles[i, 1, 4]
    variability <- upper - lower
    if (!is.na(variability)) {
      pct_var <- round(variability * 100, 1)
      if (variability > 0.2) {
        cat(sprintf("⚠️ Prediction %d shows high uncertainty (±%.1f%%). Consider reviewing.\n", i, pct_var))
      } else {
        cat(sprintf("✅ Prediction %d is reliable (±%.1f%%).\n", i, pct_var))
      }
    }
  }
}

# ----------------------------------------------------------------------------------------------------------------------
# 📅 Save Result to JSON
# ----------------------------------------------------------------------------------------------------------------------
output_path <- file.path(relative_path, "result_probs.json")

export_data <- list(
  values     = probs$values,
  samples    = probs$samples,
  quantiles  = probs$quantiles,
  Y          = probs$Y,
  X          = probs$X,
  lowertail  = probs$lowertail
)

write(
  toJSON(export_data, pretty = TRUE, auto_unbox = TRUE),
  file = output_path
)