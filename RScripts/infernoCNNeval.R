# ======================================================================================================================
# 📁 Setup: Root Directory, Paths, Parallelism, and Data
# ======================================================================================================================

library(inferno)

# Number of cores to use
parallel <- 10

# Root directory for all data
rootdir <- "data/inferno"

# Directory with trained Inferno models
learntdir <- file.path(rootdir, "combined_new_MT")

# Load metadata
metadata <- read.csv(file.path(learntdir, "metadata.csv"))
print(paste("Loaded metadata with", nrow(metadata), "entries"))

# Load test data
testdata <- read.csv(file.path(rootdir, "calibration_test.csv"))[, metadata$name]
print(paste("Loaded test data with", nrow(testdata), "samples and", ncol(testdata), "features"))


# ----------------------------------------------------------------------------------------------------------------------
# 🎯 Single Prediction and Decision
# ----------------------------------------------------------------------------------------------------------------------

# Define predictands (outcome variables) and predictors (input features)
predictands <- c("LABEL_EFFUSION", "LABEL_ATELECTASIS")
predictors <- setdiff(metadata$name, predictands)

# Define possible outcomes
y <- setNames(expand.grid(0:1, 0:1), as.list(predictands))
outcomenames <- apply(y, 1, function(x) paste0("E", x[1], "_A", x[2]))

# Define utility (or cost-benefit) matrix
ematrix <- matrix(
  c(
    1.00, 0.55, 0.60, 0.40,
    0.90, 1.00, 0.65, 0.75,
    0.90, 0.65, 1.00, 0.75,
    0.80, 0.85, 0.85, 1.00
  ),
  nrow = 4,
  byrow = TRUE
)
colnames(ematrix) <- outcomenames
rownames(ematrix) <- outcomenames

# Patient index to evaluate
patient_idx <- 100

# Extract patient predictors
x_patient <- testdata[patient_idx, predictors, drop = FALSE]

# Extract true labels
true_labels <- testdata[patient_idx, predictands, drop = FALSE]
true_outcome_name <- paste0("E", true_labels[[1]], "_A", true_labels[[2]])

# Predict outcome probabilities
probs <- Pr(
  Y = y,
  X = x_patient,
  learnt = learntdir,
  parallel = parallel
)

# Calculate expected utilities
exputilities <- ematrix %*% probs$values

# Decision function: choose option maximizing expected utility
choosemax <- function(x) {
  sample(rep(which(x == max(x)), 2), 1)
}

# Make decision
decision <- choosemax(exputilities)

# ----------------------------------------------------------------------------------------------------------------------
# 📋 Detailed Report for Single Prediction
# ----------------------------------------------------------------------------------------------------------------------

cat("\n================ Single Prediction Report ================\n")
cat("Patient Index:", patient_idx, "\n\n")

cat("Input predictors for this patient:\n")
print(x_patient)

cat("\nPredicted probabilities for outcomes:\n")
predictions_table <- data.frame(
  Outcome = outcomenames,
  Probability = round(as.numeric(probs$values), 4)
)
print(predictions_table)

cat("\nExpected utilities for each decision:\n")
utility_table <- data.frame(
  Outcome = outcomenames,
  ExpectedUtility = round(as.numeric(exputilities), 4)
)
print(utility_table)

cat("\nTrue labels (ground truth):\n")
print(true_labels)
cat("True outcome name:", true_outcome_name, "\n")

cat("\n✅ Best Decision (Maximizing Expected Utility):", outcomenames[decision], "\n")


# ----------------------------------------------------------------------------------------------------------------------
# 📊 Full Dataset Evaluation Maximizing Medical Utility
# ----------------------------------------------------------------------------------------------------------------------

# Full predictors and true labels
X <- testdata[, predictors, drop = FALSE]
trueY <- testdata[, predictands, drop = FALSE]

# Predict outcome probabilities for full dataset
probs_full <- Pr(
  Y = y,
  X = X,
  learnt = learntdir,
  parallel = parallel,
  quantiles = c(0.055, 0.945),
  nsamples = NULL
)

# Calculate expected utilities for full dataset
exputilities_full <- ematrix %*% probs_full$values

# Make decisions for full dataset
decisions_full <- apply(exputilities_full, 2, choosemax)

# Map true labels to indices
truevalues <- apply(trueY, 1, function(x) (x[1] + 2 * x[2]) + 1)

# Map true labels to outcome names
trueoutcomenames <- apply(trueY, 1, function(x) paste0("E", x[1], "_A", x[2]))

# Calculate baseline and model performance
most_common_value <- which.max(table(truevalues))
baseline_accuracy <- sum(truevalues == most_common_value) / length(truevalues)
avgyield <- mean(ematrix[cbind(decisions_full, truevalues)])

# Create bare identity matrix as utility matrix (perfect classification only)
ematrix_diag <- diag(4)

# Calculate expected utilities and decisions with bare diagonal matrix
exputilities_diag <- ematrix_diag %*% probs_full$values
decisions_diag <- apply(exputilities_diag, 2, choosemax)
avgyield_diag <- mean(ematrix_diag[cbind(decisions_diag, truevalues)])

# ----------------------------------------------------------------------------------------------------------------------
# 📋 Printout of Evaluation Results
# ----------------------------------------------------------------------------------------------------------------------

cat("\nTrue outcome distribution (%):\n")
print(round(table(truevalues) / sum(table(truevalues)) * 100, 2))

cat("\nName consistency check:", all(trueoutcomenames == outcomenames[truevalues]), "\n")

cat("\n🚀 Inferno expected utility (accuracy with clinical utility matrix):", round(avgyield * 100, 1), "%\n")

cat("\n🧪 Expected utility (accuracy with bare diagonal utility matrix):", round(avgyield_diag * 100, 1), "%\n")

cat("\n🎯 Baseline accuracy (predicting most common outcome):", round(baseline_accuracy * 100, 1), "%\n")

cat("=============================================================================\n")

# ----------------------------------------------------------------------------------------------------------------------
# 🤖 Comparison with Neural Net Decisions at Thresholds 0.5 and 0.27
# ----------------------------------------------------------------------------------------------------------------------

# --- Neural Net decisions at sigmoid threshold 0.5 ---
responsesNN_05 <- apply(
  testdata[, c("LOGIT_EFFUSION", "LOGIT_ATELECTASIS")],
  1,
  function(x) 1 * (x >= 0)
)

decisionsNN_05 <- apply(responsesNN_05, 2, function(x) {
  (x[1] + 2 * x[2]) + 1
})

responsenames_05 <- apply(responsesNN_05, 2, function(x) paste0("E", x[1], "_A", x[2]))

cat("\nNN decision naming check (threshold 0.5):",
    all(responsenames_05 == outcomenames[decisionsNN_05]), "\n")

avgyieldNN_05 <- mean(ematrix[cbind(decisionsNN_05, truevalues)])
cat("NN expected utility (accuracy, threshold 0.5):",
    round(avgyieldNN_05 * 100, 1), "%\n")

# --- Neural Net decisions at sigmoid threshold 0.27 ---
logit_threshold_027 <- qlogis(0.27)

responsesNN_027 <- apply(
  testdata[, c("LOGIT_EFFUSION", "LOGIT_ATELECTASIS")],
  1,
  function(x) 1 * (x >= logit_threshold_027)
)

decisionsNN_027 <- apply(responsesNN_027, 2, function(x) {
  (x[1] + 2 * x[2]) + 1
})

responsenames_027 <- apply(responsesNN_027, 2, function(x) paste0("E", x[1], "_A", x[2]))

cat("\nNN decision naming check (threshold 0.27):",
    all(responsenames_027 == outcomenames[decisionsNN_027]), "\n")

avgyieldNN_027 <- mean(ematrix[cbind(decisionsNN_027, truevalues)])
cat("NN expected utility (accuracy, threshold 0.27):",
    round(avgyieldNN_027 * 100, 1), "%\n")

cat("==========================================================\n")

# ----------------------------------------------------------------------------------------------------------------------
# 📊 Calibration Curves for Neural Net Outputs vs Inferred Probabilities
# ----------------------------------------------------------------------------------------------------------------------

# Define root directory and saving directory
save_dir <- file.path("data", "plots")

# Helper function to plot calibration curves for one label and save as SVG
plot_calibration <- function(outNN, probNN, label_main, file_prefix = NULL) {
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
  }

  set_plot_params <- function() {
    par(mar = c(6, 7, 5, 3), cex.axis = 1.8, cex.lab = 2.2, cex.main = 2.5, lwd = 3)
  }

  if (!is.null(file_prefix)) {
    svg(filename = file.path(save_dir, paste0(file_prefix, "_sigmoid.svg")), width = 10, height = 10)
    set_plot_params()
  }

  # Sigmoid output vs inferred probability
  flexiplot(
    x = plogis(outNN[, 1]),
    y = c(probNN$values),
    xlab = "NN sigmoid output", ylab = "Inferred probability",
    ylim = 0:1, xlim = 0:1, lwd = 5, main = paste("Calibration:", label_main)
  )
  plotquantiles(x = plogis(outNN[, 1]), y = probNN$quantiles[1, , ], add = TRUE)
  flexiplot(x = 0:1, y = 0:1, lty = 2, lwd = 3, col = 5, add = TRUE)
  polygon(x = c(0.27, 1, 1, 0.27), y = c(0, 0, 1, 1), col = adjustcolor("#0f606b", alpha.f = 0.15), border = NA)
  abline(v = 0.27, col = "#007c6c", lty = 3, lwd = 3)

  if (!is.null(file_prefix)) {
    dev.off()
    svg(filename = file.path(save_dir, paste0(file_prefix, "_logit.svg")), width = 10, height = 10)
    set_plot_params()
  }

  # Logit vs inferred probability
  plot(probNN, xlab = "Logit", ylab = "Inferred probability", ylim = 0:1, legend = FALSE, main = paste(label_main, "Logit View"))
  flexiplot(x = outNN, y = plogis(outNN[, 1]), lty = 2, col = 2, lwd = 5, add = TRUE)
  polygon(x = c(qlogis(0.27), 5, 5, qlogis(0.27)), y = c(0, 0, 1, 1), col = adjustcolor("darkgreen", alpha.f = 0.15), border = NA)
  abline(v = qlogis(0.27), col = "darkgreen", lty = 3, lwd = 3)

  if (!is.null(file_prefix)) {
    dev.off()
  }
}

# Effusion Calibration
outNN_effusion <- data.frame(LOGIT_EFFUSION = seq(-5, 5, length.out = 129))
probNN_effusion <- Pr(
  Y = data.frame(LABEL_EFFUSION = 1),
  X = outNN_effusion,
  learnt = learntdir,
  parallel = parallel,
  quantiles = c(0.055, 0.945),
  nsamples = NULL
)
plot_calibration(outNN_effusion, probNN_effusion, "Effusion", file_prefix = "effusion_calibration")

# Atelectasis Calibration
outNN_atelectasis <- data.frame(LOGIT_ATELECTASIS = seq(-5, 5, length.out = 129))
probNN_atelectasis <- Pr(
  Y = data.frame(LABEL_ATELECTASIS = 1),
  X = outNN_atelectasis,
  learnt = learntdir,
  parallel = parallel,
  quantiles = c(0.055, 0.945),
  nsamples = NULL
)
plot_calibration(outNN_atelectasis, probNN_atelectasis, "Atelectasis", file_prefix = "atelectasis_calibration")
