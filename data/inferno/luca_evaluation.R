#### Evaluation of neuralnet+Inferno inferences
library(inferno)

## How many parallel cores for the calculations
parallel <- 8

## Name of directory where 'learnt' has been saved
learntdir <- 'luca_output_5-250412T221231-vrt7_dat1500_smp3600'

metadata <- read.csv(file.path(learntdir, 'metadata.csv'))

## Load test data
testdata <- read.csv('calibration_test.csv')[, metadata$name]


#### Example calculation

## names of predictands
Ynames <- c('LABEL_EFFUSION', 'LABEL_ATELECTASIS')
## names of predictors
Xnames <- setdiff(metadata$name, Ynames)

## Create 2x2 grid of possible values for predictands
Y <- setNames(expand.grid(0:1, 0:1), as.list(Ynames))


probs <- Pr(Y = Y, X =  testdata[1, Xnames, drop=FALSE],
    learnt = learntdir, parallel = parallel)

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
Ynames <- c('LABEL_EFFUSION', 'LABEL_ATELECTASIS')
## names of predictors
Xnames <- setdiff(metadata$name, Ynames)

## Create 2x2 grid of possible values for predictands
Y <- setNames(expand.grid(0:1, 0:1), as.list(Ynames))

trueY <- testdata[, Ynames, drop=FALSE]
X <- testdata[, Xnames, drop=FALSE]

## We omit calculation of samples to save memory
probs <- Pr(Y = Y, X = X,
    learnt = learntdir, parallel = parallel,
    quantiles = c(0.055, 0.945), nsamples = NULL)
## now probs$values contains the probabilities of the four outcomes (rows)
## for each test datapoint (columns)

## Calculate accuracy: correspond to unit-diagonal utility matrix
## we have 2x2=4 possible outcomes
## rows: decisions, columns: true value
outcomenames <- apply(Y, 1, function(x)paste0('E', x[1], '_A', x[2]))
## [1] "E0_A0" "E1_A0" "E0_A1" "E1_A1"
ematrix <- diag(4)
rownames(ematrix) <- colnames(ematrix) <- outcomenames


## This is the list of expected utilities:
## each row is the exp. utility of each of the 4 decisions
## each column is a test datapoint
exputilities <- ematrix %*% probs$values

## list of decisions for all test datapoints
## use a special function that choose randomly in case of draw
## (this is important to avoid biases)
choosemax <- function(x){sample(rep(which(x == max(x)), 2), 1)}

decisions <- apply(exputilities, 2, choosemax)

## translate true values to integer in 1:4
truevalues <- apply(trueY, 1, function(x){(x[1] + 2 * x[2]) + 1})

## test consistency
trueoutcomenames <- apply(trueY, 1, function(x)paste0('E', x[1], '_A', x[2]))
all(trueoutcomenames == outcomenames[truevalues])
## [1] TRUE

avgyield <- mean(ematrix[cbind(decisions, truevalues)])
## 0.655243
## 66% accuracy
