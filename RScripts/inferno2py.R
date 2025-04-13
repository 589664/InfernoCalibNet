#=======================================================================================================================
# 📦 Load Inferno Package
#=======================================================================================================================
library("inferno")
library("jsonlite")

#=======================================================================================================================
# 📂 Read Configuration from Temporary JSON
#=======================================================================================================================
args <- commandArgs(trailingOnly = TRUE)
config_path <- args[1]
config <- fromJSON(config_path)

# Extract values from config
relative_path <- dirname(config$input_csv)
input_file    <- config$input_csv
model_path    <- config$model_path
row_index     <- config$row_index
quantiles     <- config$quantiles
input_values  <- config$input_values

#=======================================================================================================================
# 📁 Load Model
#=======================================================================================================================
inferno_model <- readRDS(model_path)

#=======================================================================================================================
# 📊 Load Data and Select Row
#=======================================================================================================================
alldata <- read.csv(
  input_file,
  na.strings = "",
  stringsAsFactors = FALSE,
  tryLogical = FALSE
)
row_data <- alldata[row_index, ]
print(row_data)

#=======================================================================================================================
# 🔍 Prepare Input and Predict
#=======================================================================================================================
input_features <- as.data.frame(input_values)
target_frame <- data.frame(LABEL_EFFUSION = 0:1)

result_probs <- Pr(
  Y = target_frame,
  X = input_features,
  learnt = inferno_model,
  parallel = 7,
  quantile = quantiles
)

#=======================================================================================================================
# 🗒️ Save Result to JSON (Including True Label)
#=======================================================================================================================
output_path <- file.path(relative_path, "result_probs.json")
export_data <- list(
  values     = result_probs$values,
  samples    = result_probs$samples,
  quantiles  = result_probs$quantiles,
  Y          = result_probs$Y,
  X          = result_probs$X,
  lowertail  = result_probs$lowertail,
  true_label = row_data$LABEL_EFFUSION
)
write(
  toJSON(export_data, pretty = TRUE, auto_unbox = TRUE),
  file = output_path
)

#=======================================================================================================================
# 📈 Plot Prediction
#=======================================================================================================================
# plot(
#   result_probs,
#   variability = "samples",
#   col = adjustcolor("#5195b0", alpha.f = 1),
#   lwd = 1,
#   grid = TRUE,
#   legend = TRUE,
#   xlab = "Effusion label",
#   ylab = "Posterior probability"
# )
# abline(v = row_data[, "LABEL_EFFUSION"], lty = 2, lwd = 2, col = 2)