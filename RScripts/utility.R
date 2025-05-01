# ================================================================================================================
# 📦 Load Libraries and Configuration
# ================================================================================================================
library("inferno")
library("jsonlite")

num_threads <- 10
inferno_model_dir <- "data/inferno/combinedML50"
conf_output_path <- "data/inferno/inferno_CM.json"

metadata <- read.csv(file.path(inferno_model_dir, "metadata.csv"))
test_data <- read.csv("data/inferno/calibration_test.csv")
test_data <- test_data[, metadata$name]

# ================================================================================================================
# 🔍 Define Predictors and Targets
# ================================================================================================================
target_vars <- c("LABEL_EFFUSION", "LABEL_ATELECTASIS")
predictor_vars <- setdiff(metadata$name, target_vars)

y_grid <- setNames(expand.grid(0:1, 0:1), as.list(target_vars))
y_true <- test_data[, target_vars, drop = FALSE]
x_input <- test_data[, predictor_vars, drop = FALSE]

# ================================================================================================================
# 🔬 Run Inferno Inference
# ================================================================================================================
probabilities <- Pr(
  Y = y_grid,
  X = x_input,
  learnt = inferno_model_dir,
  parallel = num_threads,
  quantiles = c(0.055, 0.945),
  nsamples = NULL
)

# ================================================================================================================
# 🔬 Build Utility Matrix and Label Names
# ================================================================================================================
outcome_labels <- apply(y_grid, 1, function(x) paste0("eff_", x[1], "_ate_", x[2]))

utility_matrix <- matrix(
  c(
    1.00, 0.55, 0.60, 0.40,
    0.90, 1.00, 0.65, 0.75,
    0.90, 0.65, 1.00, 0.75,
    0.80, 0.85, 0.85, 1.00
  ),
  nrow = 4,
  byrow = TRUE
)
colnames(utility_matrix) <- outcome_labels
rownames(utility_matrix) <- outcome_labels

# ================================================================================================================
# 🔄 Decision Making Based on Expected Utility
# ================================================================================================================
expected_utilities <- utility_matrix %*% probabilities$values
select_max <- function(x) sample(rep(which(x == max(x)), 2), 1)
decision_indices <- apply(expected_utilities, 2, select_max)
true_indices <- apply(y_true, 1, function(x) (x[1] + 2 * x[2]) + 1)
true_labels <- apply(y_true, 1, function(x) paste0("eff_", x[1], "_ate_", x[2]))
stopifnot(all(true_labels == outcome_labels[true_indices]))

# ================================================================================================================
# 📊 Evaluate Inferno Accuracy
# ================================================================================================================
avg_yield <- mean(utility_matrix[cbind(decision_indices, true_indices)])
cat("🔍 Average Expected Utility from Inferno Decisions:", round(avg_yield, 6), "\n")

# ================================================================================================================
# 📉 Save Confusion Matrices to JSON for Python Plotting (Inferno-Based)
# ================================================================================================================
conf_matrix_json <- function(true_vals, pred_vals) {
  mat <- table(True = true_vals, Pred = pred_vals)
  as.data.frame.matrix(mat)
}

# Reconstruct Inferno decisions into binary label predictions
y_pred_eff <- (decision_indices - 1) %% 2
y_pred_ate <- (decision_indices - 1) %/% 2

conf_eff <- conf_matrix_json(y_true$LABEL_EFFUSION, y_pred_eff)
conf_ate <- conf_matrix_json(y_true$LABEL_ATELECTASIS, y_pred_ate)

write_json(
  list(
    effusion = conf_eff,
    atelectasis = conf_ate
  ),
  conf_output_path,
  pretty = TRUE
)
