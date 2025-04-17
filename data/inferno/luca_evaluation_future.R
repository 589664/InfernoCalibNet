#### Evaluation of neuralnet+Inferno inferences
library(inferno)

## How many parallel cores for the calculations
parallel <- 10

## Name of directory where 'learnt' has been saved
learntdir <- "data/inferno/combined_new_MT"

metadata <- read.csv(file.path(learntdir, "metadata.csv"))

## Load test data
testdata <- read.csv("data/inferno/calibration_test.csv")[, metadata$name]


#### Example calculation

## names of predictands
Ynames <- c("LABEL_EFFUSION", "LABEL_ATELECTASIS")
## names of predictors
Xnames <- setdiff(metadata$name, Ynames)

## Create 2x2 grid of possible values for predictands
Y <- setNames(expand.grid(0:1, 0:1), as.list(Ynames))


probs <- Pr(
    Y = Y, X = testdata[1, Xnames, drop = FALSE],
    learnt = learntdir, parallel = parallel
)

cbind(Y, probs$values)
testdata[1, Ynames] # true value
## > LABEL_EFFUSION LABEL_ATELECTASIS probs$values
## 1              0                 0    0.3250759
## 2              1                 0    0.0978164
## 3              0                 1    0.4643846
## 4              1                 1    0.1127231
## > LABEL_EFFUSION LABEL_ATELECTASIS
## 1              0                 0


#### Make inference for all points in the test set

## names of predictands
Ynames <- c("LABEL_EFFUSION", "LABEL_ATELECTASIS")
## names of predictors
Xnames <- setdiff(metadata$name, Ynames)

## Create 2x2 grid of possible values for predictands
Y <- setNames(expand.grid(0:1, 0:1), as.list(Ynames))

trueY <- testdata[, Ynames, drop = FALSE]
X <- testdata[, Xnames, drop = FALSE]

## We omit calculation of samples to save memory
probs <- Pr(
    Y = Y, X = X,
    learnt = learntdir, parallel = parallel,
    quantiles = c(0.055, 0.945), nsamples = NULL
)
## now probs$values contains the probabilities of the four outcomes (rows)
## for each test datapoint (columns)

## Calculate accuracy: correspond to unit-diagonal utility matrix
## we have 2x2=4 possible outcomes
## rows: decisions, columns: true value
outcomenames <- apply(Y, 1, function(x) paste0("E", x[1], "_A", x[2]))
## [1] "E0_A0" "E1_A0" "E0_A1" "E1_A1"
ematrix <- diag(4)
## ## uncomment below to create a random u.matrix with entries between 0 and 1
## ematrix <- matrix(rnorm(4*4), 4, 4)#diag(4)
## ematrix <- ematrix - min(ematrix)
## ematrix <- ematrix/max(ematrix)
colnames(ematrix) <- outcomenames
rownames(ematrix) <- outcomenames


## This is the list of expected utilities:
## each row is the exp. utility of each of the 4 decisions
## each column is a test datapoint
exputilities <- ematrix %*% probs$values

## list of decisions for all test datapoints
## use a special function that choose randomly in case of draw
## (this is important to avoid biases)
choosemax <- function(x) {
    sample(rep(which(x == max(x)), 2), 1)
}

decisions <- apply(exputilities, 2, choosemax)

## translate true values to integer in 1:4
truevalues <- apply(trueY, 1, function(x) {
    (x[1] + 2 * x[2]) + 1
})

## Note that by predicting the most common condition all the time,
## we could at most reach 46.7% accuracy:
table(truevalues) / sum(table(truevalues)) * 100
## truevalues
##        1        2        3        4
## 46.74435 23.50925 22.61823  7.12817


## test consistency
trueoutcomenames <- apply(trueY, 1, function(x) paste0("E", x[1], "_A", x[2]))
all(trueoutcomenames == outcomenames[truevalues])
## [1] TRUE

avgyield <- mean(ematrix[cbind(decisions, truevalues)])
avgyield
## 0.655243
## 65.5% accuracy


## Now check answers from neural net instead
## assume a simple "logit>=0" rule is applied

responsesNN <- apply(
    testdata[, c("LOGIT_EFFUSION", "LOGIT_ATELECTASIS")],
    1, function(x) {
        1 * (x >= 0)
    }
)

decisionsNN <- apply(responsesNN, 2, function(x) {
    (x[1] + 2 * x[2]) + 1
})

## test consistency
responsenames <- apply(responsesNN, 2, function(x) paste0("E", x[1], "_A", x[2]))
all(responsenames == outcomenames[decisionsNN])
## [1] TRUE

avgyieldNN <- mean(ematrix[cbind(decisionsNN, truevalues)])
avgyieldNN
## 0.645648
## 64.6% accuracy


#### "Calibration curves":
## compare value of sigmoid output with corresponding probability
## this is just an average, since the other variates are omitted

## Effusion
outNN <- data.frame(LOGIT_EFFUSION = seq(-5, 5, length.out = 129))
probNN <- Pr(
    Y = data.frame(LABEL_EFFUSION = 1), X = outNN,
    learnt = learntdir, parallel = parallel,
    quantiles = c(0.055, 0.945), nsamples = NULL
)

mypdf("calibration_effusion", asp = 1)
flexiplot(
    x = plogis(outNN[, 1]), y = c(probNN$values),
    xlab = "NN sigmoid output", ylab = "probability", ylim = 0:1, xlim = 0:1,
    lwd = 3, main = "effusion"
)
plotquantiles(x = plogis(outNN[, 1]), y = probNN$quantiles[1, , ], add = TRUE)
flexiplot(x = 0:1, y = 0:1, lty = 2, lwd = 2, col = 5, add = TRUE)
dev.off()

## extra
plot(probNN, xlab = "logit", ylab = "probability", ylim = 0:1, legend = FALSE)
flexiplot(x = outNN, y = plogis(outNN[, 1]), lty = 2, col = 2, lwd = 3, add = TRUE)

## Atelectasis
outNN <- data.frame(LOGIT_ATELECTASIS = seq(-5, 5, length.out = 129))
probNN <- Pr(
    Y = data.frame(LABEL_ATELECTASIS = 1), X = outNN,
    learnt = learntdir, parallel = parallel,
    quantiles = c(0.055, 0.945), nsamples = NULL
)


flexiplot(
    x = plogis(outNN[, 1]), y = c(probNN$values),
    xlab = "NN sigmoid output", ylab = "probability", ylim = 0:1, xlim = 0:1,
    lwd = 3, main = "atelectasis"
)

plotquantiles(x = plogis(outNN[, 1]), y = probNN$quantiles[1, , ], add = TRUE)
flexiplot(x = 0:1, y = 0:1, lty = 2, lwd = 2, col = 5, add = TRUE)

# Add shaded decision region (sigmoid ≥ 0.27)
polygon(
    x = c(0.27, 1, 1, 0.27),
    y = c(0, 0, 1, 1),
    col = adjustcolor("darkgreen", alpha.f = 0.15),
    border = NA
)

# Add vertical threshold line at 0.27
abline(v = 0.27, col = "darkgreen", lty = 3, lwd = 2)

dev.off()

## extra
# Compute logit threshold for sigmoid = 0.27
logit_thresh <- qlogis(0.27)

# Plot calibrated probabilities vs logit
plot(probNN, xlab = "logit", ylab = "probability", ylim = 0:1, legend = FALSE)

# Add sigmoid curve for reference
flexiplot(x = outNN, y = plogis(outNN[, 1]), lty = 2, col = 2, lwd = 3, add = TRUE)

# Highlight decision region (logit ≥ qlogis(0.27))
polygon(
    x = c(logit_thresh, 5, 5, logit_thresh),
    y = c(0, 0, 1, 1),
    col = adjustcolor("darkgreen", alpha.f = 0.15),
    border = NA
)

# Add vertical threshold line at logit corresponding to sigmoid = 0.27
abline(v = logit_thresh, col = "darkgreen", lty = 3, lwd = 2)
