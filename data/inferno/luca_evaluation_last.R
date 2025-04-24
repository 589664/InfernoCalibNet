###########################################################################
#### Neuralnet+Inferno inferences
###########################################################################

library(inferno)

## utility function for pdf
pdf2 <- function(file, ...){
    pdf(file = paste0(sub('.pdf$', '', file), '.pdf'),
        paper = 'special', height=148/25.4, width=148/25.4, #210/25.4, #A5 size
        ...)
}


## How many parallel cores for the calculations
parallel <- 8

## Name of directory where 'learnt' has been saved
learntdir <- 'data/inferno/combinedML50'

metadata <- read.csv(file.path(learntdir, 'metadata.csv'))

## Load test data
testdata <- read.csv('calibration_test.csv')[, metadata$name]



###########################################################################
#### Example visualization of probability of binary variate
#### depending (conditional on) another
###########################################################################

Xage <- data.frame(AGE=1:100)
Yeff <- data.frame(LABEL_EFFUSION=1)
Yale <- data.frame(LABEL_ATELECTASIS=1)

condpreff <- Pr(Y = Yeff, X = Xage, learnt = learntdir,
    parallel = parallel, quantiles = c(0.055, 0.945))

condprale <- Pr(Y = Yale, X = Xage, learnt = learntdir,
    parallel = parallel, quantiles = c(0.055, 0.945))

pdf2('lungcondition_vs_age')
plot(condpreff, ylim = 0:1, col = 1, lty = 1, lwd = 2,
    legend=FALSE, ylab='Prob. of Effusion/Atelectasis given Age (89% variability)')
plot(condprale, ylim = 0:1, col = 2, lty = 2, lwd = 2,
    legend=FALSE, add=TRUE)
legend('top', legend = c('Effusion', 'Atelectasis'),
    col = 1:2, lty = 1:2, lwd = 2, pch=NA, bty='n')
dev.off()


###########################################################################
#### Example "calibration" curves for different age groups
###########################################################################

#### Unfortunately inferno has no built-in function to
#### calculate probabilities conditional on intervals.
#### So such probabilities must be calculated explicitly
#### using the probability rules:
#### P(Y | X1, a<X2<b) = P(Y, a<X2<b | X1)/P(a<X2<b | X1)
#### summing for X2=...
#### Unfortunately this way we lose the variability
#### (it could also be calculated, but by a lengthier procedure)

## Effusion

## Calculate probabilities for all ages
probs1 <- Pr(Y = data.frame(LABEL_EFFUSION = 1, AGE = 0:99),
    X = data.frame(LOGIT_EFFUSION = seq(-5, 5, length.out=129)),
    learnt = learntdir, parallel = parallel,
    quantiles = NULL, nsamples = NULL)
##
probs2 <- Pr(Y = data.frame(AGE = 0:99),
    X = data.frame(LOGIT_EFFUSION = seq(-5, 5, length.out=129)),
    learnt = learntdir, parallel = parallel,
    quantiles = NULL, nsamples = NULL)

## sum according to age groups and calculate conditionals
condprobs <- sapply(seq(0, 80, by = 20),
    function(minage){
        colSums(probs1$values[minage:(minage+19),]) /
            colSums(probs2$values[minage:(minage+19),])
        }
)

pdf2('calibration_vs_age')
flexiplot(x = plogis(seq(-5, 5, length.out=129)), y = condprobs,
    xlab = 'NN sigmoid output', ylab = 'probability', main = 'effusion',
    ylim = 0:1, xlim = 0:1,
    col = palette('Okabe-Ito'), lty = 1:10, lwd = 3)
legend('topleft',
    legend = sapply(seq(0, 80, by = 20),
        function(minage){paste0('age ', minage, ' -- ', minage+19)}),
    lty=1:10, col=palette('Okabe-Ito'), lwd=2, pch=NA, bty='n'
    )
dev.off()


###########################################################################
#### Example probability calculation
###########################################################################

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



###########################################################################
#### Utility-based evaluation
###########################################################################

#### Draw inference for all points in the test set

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
choosemax <- function(x){sample(rep(which(x == max(x)), 2), 1)}

decisions <- apply(exputilities, 2, choosemax)

## translate true values to integer in 1:4
truevalues <- apply(trueY, 1, function(x){(x[1] + 2 * x[2]) + 1})

## Note that by predicting the most common condition all the time,
## we could at most reach 46.7% accuracy:
## > table(truevalues)/sum(table(truevalues))*100
## truevalues
##        1        2        3        4
## 46.74435 23.50925 22.61823  7.12817


## test consistency
trueoutcomenames <- apply(trueY, 1, function(x)paste0('E', x[1], '_A', x[2]))
all(trueoutcomenames == outcomenames[truevalues])
## [1] TRUE

avgyield <- mean(ematrix[cbind(decisions, truevalues)])
avgyield
## > [1] 0.657985


## Now check answers from neural net instead
## assume a simple "logit>=0" rule is applied

responsesNN <- apply(
    testdata[, c('LOGIT_EFFUSION', 'LOGIT_ATELECTASIS')],
    1, function(x){1*(x>=0)})

decisionsNN <- apply(responsesNN, 2, function(x){(x[1] + 2 * x[2]) + 1})

## test consistency
responsenames <- apply(responsesNN, 2, function(x)paste0('E', x[1], '_A', x[2]))
all(responsenames == outcomenames[decisionsNN])
## [1] TRUE

avgyieldNN <- mean(ematrix[cbind(decisionsNN, truevalues)])
avgyieldNN
## > [1] 0.645648


###########################################################################
#### "Calibration" curves
###########################################################################

## compare value of sigmoid output with corresponding probability
## this is just an average, since the other variates are omitted

## Effusion
outNN <- data.frame(LOGIT_EFFUSION = seq(-5, 5, length.out=129))
probNN <- Pr(Y=data.frame(LABEL_EFFUSION = 1), X = outNN,
    learnt = learntdir, parallel = parallel,
    quantiles = c(0.055, 0.945), nsamples = NULL)

pdf2('calibration_effusion')
flexiplot(x = plogis(outNN[,1]), y = c(probNN$values),
    xlab = 'NN sigmoid output', ylab = 'probability', ylim = 0:1, xlim = 0:1,
    lwd = 3, main = 'effusion')
plotquantiles(x = plogis(outNN[,1]), y = probNN$quantiles[1,,], add = TRUE)
flexiplot(x = 0:1, y=0:1, lty = 2, lwd = 2, col = 5, add = TRUE)
dev.off()


plot(probNN, xlab = 'logit', ylab = 'probability', ylim = 0:1, legend = FALSE)
flexiplot(x = outNN, y = plogis(outNN[,1]), lty = 2, col = 2, lwd = 3, add = TRUE)

## Atelectasis
outNN <- data.frame(LOGIT_ATELECTASIS = seq(-5, 5, length.out=129))
probNN <- Pr(Y=data.frame(LABEL_ATELECTASIS = 1), X = outNN,
    learnt = learntdir, parallel = parallel,
    quantiles = c(0.055, 0.945), nsamples = NULL)

pdf2('calibration_atelectasis')
flexiplot(x = plogis(outNN[,1]), y = c(probNN$values),
    xlab = 'NN sigmoid output', ylab = 'probability', ylim = 0:1, xlim = 0:1,
    lwd = 3, main = 'atelectasis')
plotquantiles(x = plogis(outNN[,1]), y = probNN$quantiles[1,,], add = TRUE)
flexiplot(x = 0:1, y=0:1, lty = 2, lwd = 2, col = 5, add = TRUE)
dev.off()


###########################################################################
#### Mutual information (and other entropies) between logits and age
###########################################################################

mi <- mutualinfo(
    Y1names = c('LOGIT_ATELECTASIS', 'LOGIT_EFFUSION'),
    Y2names = c('AGE'),
    X = NULL,
    learnt = learntdir, parallel = parallel)
mi
## $MI
##      value      error 
## 0.10503075 0.00888871 
## 
## $CondEn12
##     value     error 
## 5.2255378 0.0228907 
## 
## $CondEn21
##     value     error 
## 5.9128910 0.0162255 
## 
## $En1
##    value    error 
## 5.330753 0.022555 
## 
## $En2
##     value     error 
## 6.0173237 0.0157206 
## 
## $MImax
##    value    error 
## 5.330753 0.022555 
## 
## $unit
## [1] "Sh"
## 
## $Y1names
## [1] "LOGIT_ATELECTASIS" "LOGIT_EFFUSION"   
## 
## $Y2names
## [1] "AGE"

###########################################################################
#### Mutual information (and other entropies) between logits and age
###########################################################################




###########################################################################
#### Test on different base rates
###########################################################################

## Rates and counts in current test set

cases <- expand.grid(LABEL_ATELECTASIS = 0:1, LABEL_EFFUSION = 0:1)
casenames <- c('none', 'atel', 'effu', 'both')

oldN <- nrow(testdata)

oldcounts <- apply(cases, 1, function(x){
    nrow(testdata[
        testdata[['LABEL_ATELECTASIS']] == x[['LABEL_ATELECTASIS']] &
        testdata[['LABEL_EFFUSION']] == x[['LABEL_EFFUSION']]
      , ])})
names(oldcounts) <- casenames
oldrates <- oldcounts/sum(oldcounts)
oldrates
##      none      atel      effu      both 
## 0.4674435 0.2261823 0.2350925 0.0712817 


## Maximization of the new test set

## new rates
set.seed(800)
newrates <- c(0.65, 0.15, 0.15, 0.05)
names(newrates) <- casenames
## none atel effu both 
## 0.75 0.10 0.10 0.05 

for(acase in seq_len(nrow(cases))){
    testcounts <- floor(newrates * oldcounts[acase] / newrates[acase])
    if(all(testcounts <= oldcounts)){
        message('Keep "', casenames[acase], '"')
        newcounts <- testcounts
        newN <- sum(testcounts)
        message('New N: ', newN)
    }
}
## Keep "none"
## New N: 1048

newtestdata <- testdata[0,] # empty
for(acase in seq_len(nrow(cases))){
    tochoose <- sample( which(
        testdata[['LABEL_ATELECTASIS']] == cases[acase, 'LABEL_ATELECTASIS'] &
            testdata[['LABEL_EFFUSION']] == cases[acase, 'LABEL_EFFUSION']
    ), size = newcounts[acase], replace = FALSE)
    ##
    newtestdata <- rbind(newtestdata, testdata[tochoose,])
}

## double check the new rates, adjust for rounding
newcounts <- apply(cases, 1, function(x){
    nrow(newtestdata[
        newtestdata[['LABEL_ATELECTASIS']] == x[['LABEL_ATELECTASIS']] &
        newtestdata[['LABEL_EFFUSION']] == x[['LABEL_EFFUSION']]
      , ])})
names(newcounts) <- casenames
newrates <- newcounts/sum(newcounts)
newcounts
newrates
## none atel effu both 
##  682  157  157   52 
##      none      atel      effu      both 
## 0.6507634 0.1498092 0.1498092 0.0496183 

write.csv(newtestdata, 'calibration_test_newbaserate.csv', row.names = FALSE, quote = TRUE, na = '')
