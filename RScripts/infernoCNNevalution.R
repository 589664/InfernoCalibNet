# ======================================================================================================================
# 📁 Setup: Paths, Parallelism, and Data
# ======================================================================================================================

library(inferno)

# Number of cores to use
parallel <- 10

# Directory with trained Inferno models
learntdir <- "data/inferno/combined_new_MT"

# Load metadata and test data
metadata <- read.csv(file.path(learntdir, "metadata.csv"))
print(paste("Loaded metadata with", nrow(metadata), "entries"))

testdata <- read.csv("data/inferno/calibration_test.csv")[, metadata$name]
print(paste("Loaded test data with", nrow(testdata), "samples and", ncol(testdata), "features"))


# ======================================================================================================================
# 📊 Example Inference for a Single Test Case
# ======================================================================================================================

Ynames <- c("LABEL_EFFUSION", "LABEL_ATELECTASIS")
Xnames <- setdiff(metadata$name, Ynames)

Y <- setNames(expand.grid(0:1, 0:1), as.list(Ynames))

probs <- Pr(
    Y = Y,
    X = testdata[1, Xnames, drop = FALSE],
    learnt = learntdir,
    parallel = parallel
)

print("Probabilities for first test sample:")
print(cbind(Y, probs$values))

print("True labels for first sample:")
print(testdata[1, Ynames])


# ======================================================================================================================
# 🔢 Inference on Entire Test Set
# ======================================================================================================================

trueY <- testdata[, Ynames, drop = FALSE]
X <- testdata[, Xnames, drop = FALSE]

probs <- Pr(
    Y = Y,
    X = X,
    learnt = learntdir,
    parallel = parallel,
    quantiles = c(0.055, 0.945),
    nsamples = NULL
)

outcomenames <- apply(Y, 1, function(x) paste0("E", x[1], "_A", x[2]))
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

exputilities <- ematrix %*% probs$values

choosemax <- function(x) {
    sample(rep(which(x == max(x)), 2), 1)
}

decisions <- apply(exputilities, 2, choosemax)

truevalues <- apply(trueY, 1, function(x) {
    (x[1] + 2 * x[2]) + 1
})

print(table(truevalues) / sum(table(truevalues)) * 100)

trueoutcomenames <- apply(trueY, 1, function(x) paste0("E", x[1], "_A", x[2]))
print(paste("Name consistency check:", all(trueoutcomenames == outcomenames[truevalues])))

avgyield <- mean(ematrix[cbind(decisions, truevalues)])
print(paste("Inferno expected utility (accuracy):", round(avgyield * 100, 1), "%"))


# ======================================================================================================================
# 🤖 Comparison with Neural Net Decisions at Thresholds 0.5 and 0.27
# ======================================================================================================================

# --- Logit >= 0 (sigmoid threshold 0.5) ---
responsesNN <- apply(
  testdata[, c("LOGIT_EFFUSION", "LOGIT_ATELECTASIS")],
  1,
  function(x) 1 * (x >= 0)
)

decisionsNN <- apply(responsesNN, 2, function(x) {
  (x[1] + 2 * x[2]) + 1
})

responsenames <- apply(responsesNN, 2, function(x) paste0("E", x[1], "_A", x[2]))
print(paste("NN decision naming check (threshold 0.5):",
            all(responsenames == outcomenames[decisionsNN])))

avgyieldNN <- mean(ematrix[cbind(decisionsNN, truevalues)])
print(paste("NN expected utility (accuracy, threshold 0.5):",
            round(avgyieldNN * 100, 1), "%"))


# --- Logit >= qlogis(0.27) (sigmoid threshold 0.27) ---
logit_threshold <- qlogis(0.27)

responsesNN <- apply(
  testdata[, c("LOGIT_EFFUSION", "LOGIT_ATELECTASIS")],
  1,
  function(x) 1 * (x >= logit_threshold)
)

decisionsNN <- apply(responsesNN, 2, function(x) {
  (x[1] + 2 * x[2]) + 1
})

responsenames <- apply(responsesNN, 2, function(x) paste0("E", x[1], "_A", x[2]))
print(paste("NN decision naming check (threshold 0.27):",
            all(responsenames == outcomenames[decisionsNN])))

avgyieldNN <- mean(ematrix[cbind(decisionsNN, truevalues)])
print(paste("NN expected utility (accuracy, threshold 0.27):",
            round(avgyieldNN * 100, 1), "%"))


# ======================================================================================================================
# 📈 Calibration Curves
# ======================================================================================================================

#--- Effusion Calibration ---#
outNN <- data.frame(LOGIT_EFFUSION = seq(-5, 5, length.out = 129))
probNN <- Pr(
    Y = data.frame(LABEL_EFFUSION = 1),
    X = outNN,
    learnt = learntdir,
    parallel = parallel,
    quantiles = c(0.055, 0.945),
    nsamples = NULL
)

flexiplot(
    x = plogis(outNN[, 1]),
    y = c(probNN$values),
    xlab = "NN sigmoid output", ylab = "probability",
    ylim = 0:1, xlim = 0:1, lwd = 3, main = "effusion"
)
plotquantiles(x = plogis(outNN[, 1]), y = probNN$quantiles[1, , ], add = TRUE)
flexiplot(x = 0:1, y = 0:1, lty = 2, lwd = 2, col = 5, add = TRUE)
dev.off()

plot(probNN, xlab = "logit", ylab = "probability", ylim = 0:1, legend = FALSE)
flexiplot(x = outNN, y = plogis(outNN[, 1]), lty = 2, col = 2, lwd = 3, add = TRUE)


#--- Atelectasis Calibration ---#
outNN <- data.frame(LOGIT_ATELECTASIS = seq(-5, 5, length.out = 129))
probNN <- Pr(
    Y = data.frame(LABEL_ATELECTASIS = 1),
    X = outNN,
    learnt = learntdir,
    parallel = parallel,
    quantiles = c(0.055, 0.945),
    nsamples = NULL
)

flexiplot(
    x = plogis(outNN[, 1]),
    y = c(probNN$values),
    xlab = "NN sigmoid output", ylab = "probability",
    ylim = 0:1, xlim = 0:1, lwd = 3, main = "atelectasis"
)
plotquantiles(x = plogis(outNN[, 1]), y = probNN$quantiles[1, , ], add = TRUE)
flexiplot(x = 0:1, y = 0:1, lty = 2, lwd = 2, col = 5, add = TRUE)

polygon(x = c(0.27, 1, 1, 0.27), y = c(0, 0, 1, 1), col = adjustcolor("#0f606b", alpha.f = 0.15), border = NA)
abline(v = 0.27, col = "#007c6c", lty = 3, lwd = 2)
dev.off()

logit_thresh <- qlogis(0.27)
plot(probNN, xlab = "logit", ylab = "probability", ylim = 0:1, legend = FALSE)
flexiplot(x = outNN, y = plogis(outNN[, 1]), lty = 2, col = 2, lwd = 3, add = TRUE)
polygon(x = c(logit_thresh, 5, 5, logit_thresh), y = c(0, 0, 1, 1), col = adjustcolor("darkgreen", alpha.f = 0.15), border = NA)
abline(v = logit_thresh, col = "darkgreen", lty = 3, lwd = 2)
