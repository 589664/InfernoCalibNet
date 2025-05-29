# Load Inferno package
library("inferno")

# Paths and configuration
relative_path <- "data/inferno"
plots_save_path <- "data/plots/PRexample.pdf"
input_file    <- file.path(relative_path, "calibration_test.csv")
metadata_file <- file.path(relative_path, "inferno_metadata.csv")
inferno_dir   <- file.path(relative_path, "combinedML50")
parallel_cores <- 7
random_seed    <- 42

# Load trained model and test data
inferno_model <- readRDS(file.path(inferno_dir, "learnt.rds"))
alldata <- read.csv(input_file, na.strings = "", stringsAsFactors = FALSE, tryLogical = FALSE)

# Select instance by index
index <- 42
selected_row <- alldata[index, ]
input_features <- selected_row[, c("AGE", "LOGIT_EFFUSION", "GENDER", "LOGIT_ATELECTASIS")]
target_frame <- data.frame(LABEL_EFFUSION = 0:1)

# Predict with uncertainty quantiles
uncertainty_bounds <- c(0.055, 0.25, 0.75, 0.945)
result_probs <- Pr(Y = target_frame, X = input_features, learnt = inferno_model,
     parallel = parallel_cores, quantile = uncertainty_bounds)

# Save plot to PDF with bold Palatino text
pdf(file = file.path(plots_save_path), width = 7, height = 7, family = "Palatino")
par(font.lab = 2, font.axis = 2, font.main = 2, font.sub = 2)  # Bold text and legend
plot(result_probs, variability = "quantiles", col = adjustcolor("#5195b0", alpha.f = 1),
     lwd = 1, grid = TRUE, legend = TRUE, xlab = "Effusion label", ylab = "Posterior probability")
abline(v = selected_row$LABEL_EFFUSION, lty = 2, lwd = 2, col = 2)
dev.off()