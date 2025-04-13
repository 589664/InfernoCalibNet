#### Tentative learning
library('inferno')

## Set random-generator seed to reproduce results if repeated
seed <- 16
## Parallel CPUs for the computation
parallel <- 10

## Name of directory where to save what has been "learned"
## a timestamp may be appended to this string
dat <- 'calibration_train.csv'
metadata <- 'meta_calibration_train.csv'
outputdir <- 'luca_output_1'

## NOTE:
## If one has to reduce the amount of learning data
## because of computational limitations,
## still the learn() can use the full set of training data
## at little computational cost
## to obtain useful learning information.
## Give the full file of data in the 'auxdata' argument.

outputdir <- learn(
    data = dat,
    prior = FALSE,
    metadata = metadata,
    outputdir = outputdir,
    appendtimestamp = TRUE,
    appendinfo = TRUE,
    output = 'directory',
    parallel = parallel,
    ## parameters for short test run:
    ## subsampledata = 100,
    ## auxdata = dat,
    ## maxhours = 0,
    ##
    seed = seed
)

