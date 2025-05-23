# -------------------------------
# General Utility Functions
# -------------------------------

# Set number of parallel cores
parallel <- 15
set.seed(42)

# Utility function to save plots to SVG
svg2 <- function(file, path = '.', ...) {
    fullpath <- file.path(path, paste0(sub(".svg$", "", file), ".svg"))
    svg(filename = fullpath, width = 148/25.4, height = 148/25.4, ...)
}

# Utility function to save plots to PDF
pdf2 <- function(file, path = '.', ...) {
    fullpath <- file.path(path, paste0(sub(".pdf$", "", file), ".pdf"))
    pdf(file = fullpath, width = 148/25.4, height = 148/25.4, ...)
}


# Tie-breaking maximum selector
choosemax <- function(x) {
    sample(rep(which(x == max(x)), 2), 1)
}

# Load metadata and test data
load_metadata_testdata <- function(learntdir, testfile = 'calibration_test.csv') {
    metadata <- read.csv(file.path(learntdir, 'metadata.csv'))
    testdata <- read.csv(file.path(dirname(learntdir), testfile))[, metadata$name]
    list(metadata = metadata, testdata = testdata)
}

# -------------------------------
# Calibration Curve Utilities
# -------------------------------

# Make grouped calibration curves
make_calibration_curve <- function(probs1, probs2, group_size = 20) {
    sapply(seq(0, nrow(probs1$values) - group_size, by = group_size), function(minage) {
        colSums(probs1$values[minage:(minage + group_size - 1),]) /
            colSums(probs2$values[minage:(minage + group_size - 1),])
    })
}

# -------------------------------
# Utility Evaluation Functions
# -------------------------------

# Compute expected utilities
compute_expected_utilities <- function(probs, ematrix) {
    ematrix %*% probs$values
}

# Make decisions based on expected utilities
make_decisions <- function(exputilities) {
    apply(exputilities, 2, choosemax)
}

# Evaluate average yield
evaluate_accuracy <- function(decisions, truevalues, ematrix) {
    mean(ematrix[cbind(decisions, truevalues)])
}

# -------------------------------
# Mutual Information Utilities
# -------------------------------

# Mutual information wrapper
calculate_mi <- function(Y1names, Y2names, learntdir) {
    mutualinfo(Y1names = Y1names, Y2names = Y2names, X = NULL, learnt = learntdir, parallel = parallel)
}

# -------------------------------
# Base Rate Adjustment Utilities
# -------------------------------

# Resample test data to match new base rates
resample_to_baserate <- function(testdata, cases, target_rates) {
    oldcounts <- apply(cases, 1, function(x) {
        nrow(testdata[
            testdata[["LABEL_ATELECTASIS"]] == x[["LABEL_ATELECTASIS"]] &
            testdata[["LABEL_EFFUSION"]] == x[["LABEL_EFFUSION"]]
        , ])
    })

    names(oldcounts) <- rownames(cases)

    for (acase in seq_len(nrow(cases))) {
        testcounts <- floor(target_rates * oldcounts[acase] / target_rates[acase])
        if (all(testcounts <= oldcounts)) {
            newcounts <- testcounts
            break
        }
    }

    newtestdata <- testdata[0,]
    for (acase in seq_len(nrow(cases))) {
        tochoose <- sample(which(
            testdata[["LABEL_ATELECTASIS"]] == cases[acase, 'LABEL_ATELECTASIS'] &
            testdata[["LABEL_EFFUSION"]] == cases[acase, 'LABEL_EFFUSION']
        ), size = newcounts[acase], replace = FALSE)
        newtestdata <- rbind(newtestdata, testdata[tochoose,])
    }
    newtestdata
}
