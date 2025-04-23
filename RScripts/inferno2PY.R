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
relative_path   <- dirname(config$model_path)
model_path      <- config$model_path
quantiles       <- config$quantiles
input_values    <- config$input_values
predictors      <- config$predictors
predictands     <- config$predictands

#=======================================================================================================================
# 📁 Load Model
#=======================================================================================================================
inferno_model <- readRDS(model_path)

#=======================================================================================================================
# 🔍 Prepare Predictors and Predictands for Prediction
#=======================================================================================================================
predictor_frame <- as.data.frame(input_values[predictors])
predictand_frame <- as.data.frame(expand.grid(input_values[predictands]))

print("🔍 Predictor Features:")
print(predictor_frame)

print("🔍 Predictand Targets:")
print(predictand_frame)

result_probs <- Pr(
  Y = predictand_frame,
  X = predictor_frame,
  learnt = inferno_model,
  parallel = 10,
  quantile = quantiles
)

#=======================================================================================================================
# 📅 Save Result to JSON
#=======================================================================================================================
output_path <- file.path(relative_path, "result_probs.json")

export_data <- list(
  values     = result_probs$values,
  samples    = result_probs$samples,
  quantiles  = result_probs$quantiles,
  Y          = result_probs$Y,
  X          = result_probs$X,
  lowertail  = result_probs$lowertail
)
write(
  toJSON(export_data, pretty = TRUE, auto_unbox = TRUE),
  file = output_path
)