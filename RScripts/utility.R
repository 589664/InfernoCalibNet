# ================================================================================================================
# 📦 Load Libraries and Configuration
# ================================================================================================================
library("inferno")

num_threads <- 8
inferno_model_dir <- "../data/inferno/combined"
metadata <- read.csv(file.path(inferno_model_dir, "metadata.csv"))
test_data <- read.csv("../data/inferno/calibration_test.csv")
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
# 🧮 Build Utility Matrix and Label Names
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
print(avg_yield)

# ================================================================================================================
# 🤖 Baseline Rule Using Raw Logits (Approximate NN Output)
# ================================================================================================================
baseline_responses <- apply(
  test_data[, c("LOGIT_EFFUSION", "LOGIT_ATELECTASIS")],
  1,
  function(x) as.integer(x >= 0)
)
baseline_decisions <- apply(baseline_responses, 2, function(x) (x[1] + 2 * x[2]) + 1)
baseline_labels <- apply(baseline_responses, 2, function(x) paste0("eff_", x[1], "_ate_", x[2]))
stopifnot(all(baseline_labels == outcome_labels[baseline_decisions]))

# ================================================================================================================
# 📊 Evaluate Baseline Accuracy
# ================================================================================================================
avg_yield_baseline <- mean(utility_matrix[cbind(baseline_decisions, true_indices)])
print(avg_yield_baseline)
