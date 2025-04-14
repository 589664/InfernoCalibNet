#### Prepare metadata
library('inferno')

datafile <- 'calibration_train.csv'
outfile <- paste0('meta_', datafile)

metadatatemplate(data = datafile, file = outfile,
    includevrt = c(
        'AGE',
        'GENDER',
        'VP',
        'LOGIT_EFFUSION',
        'LABEL_EFFUSION',
        'LOGIT_ATELECTASIS',
        'LABEL_ATELECTASIS'
    ))

## Further changes to created metadata: none


