#=======================================================================================================================
# 📦 Load Libraries and Configuration
#=======================================================================================================================
library("inferno")

parallel <- 8
learntdir <- "data/inferno/luca_inferno"
metadata <- read.csv(file.path(learntdir, "metadata.csv"))
testdata <- read.csv("data/inferno/calibration_test.csv")[, metadata$name]

#=======================================================================================================================
# 🔍 Define Predictors and Targets
#=======================================================================================================================
Ynames <- c("LABEL_EFFUSION", "LABEL_ATELECTASIS")
Xnames <- setdiff(metadata$name, Ynames)

Y <- setNames(expand.grid(0:1, 0:1), as.list(Ynames))
trueY <- testdata[, Ynames, drop=FALSE]
X <- testdata[, Xnames, drop=FALSE]

#=======================================================================================================================
# 🔬 Run Inferno Inference
#=======================================================================================================================
probs <- Pr(
  Y = Y,
  X = X,
  learnt = learntdir,
  parallel = parallel,
  quantiles = c(0.055, 0.945),
  nsamples = NULL
)

#=======================================================================================================================
# 🔢 Build Utility Matrix and Outcome Labels
#=======================================================================================================================
outcomenames <- apply(Y, 1, function(x) paste0("E", x[1], "_A", x[2]))
ematrix <- diag(4)
colnames(ematrix) <- outcomenames
rownames(ematrix) <- outcomenames

#=======================================================================================================================
# 🔄 Decision Making Based on Expected Utility
#=======================================================================================================================
exputilities <- ematrix %*% probs$values
choosemax <- function(x) sample(rep(which(x == max(x)), 2), 1)
decisions <- apply(exputilities, 2, choosemax)
truevalues <- apply(trueY, 1, function(x) (x[1] + 2 * x[2]) + 1)

trueoutcomenames <- apply(trueY, 1, function(x) paste0("E", x[1], "_A", x[2]))
stopifnot(all(trueoutcomenames == outcomenames[truevalues]))

#=======================================================================================================================
# 📊 Evaluate Inferno Accuracy
#=======================================================================================================================
avgyield <- mean(ematrix[cbind(decisions, truevalues)])
print(avgyield)  # ~0.655

#=======================================================================================================================
# 🤖 Baseline Rule Using Raw Logits (Approximate NN Output)
#=======================================================================================================================
responsesNN <- apply(
  testdata[, c("LOGIT_EFFUSION", "LOGIT_ATELECTASIS")],
  1,
  function(x) as.integer(x >= 0)
)
decisionsNN <- apply(responsesNN, 2, function(x) (x[1] + 2 * x[2]) + 1)
responsenames <- apply(responsesNN, 2, function(x) paste0("E", x[1], "_A", x[2]))
stopifnot(all(responsenames == outcomenames[decisionsNN]))

#=======================================================================================================================
# 📊 Evaluate Baseline Accuracy
#=======================================================================================================================
avgyieldNN <- mean(ematrix[cbind(decisionsNN, truevalues)])
print(avgyieldNN)  # ~0.646