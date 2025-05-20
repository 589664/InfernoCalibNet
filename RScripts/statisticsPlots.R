# Load required libraries
library(circlize)
library(stringr)
library(grid)
library(ComplexHeatmap)

# Define file path
file_path <- "data/raw/xraysAUX.csv"

# Load the dataset
df <- read.csv(file_path, stringsAsFactors = FALSE)

# Rename column if needed to match expected name
if ("Finding.Labels" %in% colnames(df)) {
  colnames(df)[which(colnames(df) == "Finding.Labels")] <- "Finding Labels"
}

# Define disease labels
diseases <- c("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema",
              "Fibrosis", "Hernia", "Infiltration", "Mass", "No Finding", "Nodule",
              "Pleural_Thickening", "Pneumonia", "Pneumothorax")

# One-hot encode the 'Finding Labels' column
for (d in diseases) {
  df[[d]] <- as.integer(str_detect(df$`Finding Labels`, fixed(d)))
}

# Precompute total/unique counts and percentages based on number of dataset entries
n_total <- nrow(df)
total_counts <- sapply(diseases, function(d) sum(df[[d]]))
unique_counts <- sapply(diseases, function(d) sum(df[[d]] == 1 & rowSums(df[diseases]) == 1))
total_pct <- round(100 * total_counts / n_total, 1)
unique_pct <- round(100 * unique_counts / n_total, 1)
bar_matrix <- rbind(total_counts, unique_counts)

# Compute co-occurrence matrix
label_matrix <- df[diseases]
co_mat <- as.matrix(t(label_matrix)) %*% as.matrix(label_matrix)

# Remove self-links (optional)
diag(co_mat) <- 0

# Define heat color function for co-occurrence strength
col_fun <- circlize::colorRamp2(c(0, max(co_mat) * 0.5, max(co_mat)),
                                c("white", "#69c1c5", "#f8766d"))

# Assign updated modern pastel colors to each disease
# disease_colors <- structure(c("#69212D", "#7A522A", "#7A8A33", "#49993D", "#48A87D",
#                               "#579DB2", "#6C72B7", "#A480BC", "#256571", "#2E3B82",
#                               "#673892", "#A1428D", "#AF4E58", "#B49062", "#A9B976"),
#                             names = diseases)

disease_colors <- structure(c("#0D1526", "#173D40", "#21593D", "#32722C", "#6E8A38",
                              "#A18A45", "#B36656", "#BC718F", "#331E12", "#4D1C26",
                              "#662759", "#61327E", "#423E95", "#4B80AC", "#64B8AD"),
                            names = diseases)

# First PDF: Chord diagram with co-occurrence heatmap
pdf("chord_diagram.pdf", width = 10, height = 10, family = "Palatino")
par(mar = c(1, 1, 2, 1))
chordDiagram(
  co_mat,
  grid.col = disease_colors,
  col = col_fun(co_mat),
  transparency = 0.25,
  annotationTrack = "grid",
  preAllocateTracks = list(track.height = 0.1)
)

circos.trackPlotRegion(
  track.index = 1,
  panel.fun = function(x, y) {
    sector_name <- get.cell.meta.data("sector.index")
    circos.text(
      x = mean(get.cell.meta.data("xlim")),
      y = 0,
      labels = sector_name,
      facing = "clockwise",
      niceFacing = TRUE,
      adj = c(0, 0.5),
      cex = 0.9,
      col = "#333333",
      font = 2
    )
  },
  bg.border = NA
)

# Add title
title("Disease Co-occurrence Chord Diagram", line = -1, cex.main = 1.2, col.main = "#333333")

# Add heat color legend
lgd <- Legend(col_fun = col_fun, title = "Co-occurrence", title_gp = gpar(fontfamily = "Palatino"))
draw(lgd, x = unit(0.85, "npc"), y = unit(0.1, "npc"), just = c("left", "bottom"))

circos.clear()
dev.off()

# Second PDF: Disease-label distribution barplot with corrected percentages
pdf("disease_distribution_overview.pdf", width = 10, height = 8, family = "Palatino")
par(mar = c(5, 9, 4, 1))
bar_colors <- c("#69c1c5", "#f8766d")

# Create barplot without axis labels
barplot_result <- barplot(
  bar_matrix,
  beside = TRUE,
  horiz = TRUE,
  col = bar_colors,
  names.arg = rep("", length(diseases)),
  las = 1,
  cex.names = 0.8,
  main = "Disease-label Distribution Overview",
  xlab = "Count",
  legend.text = c("Total", "Unique"),
  xlim = c(0, max(bar_matrix) * 1.15),
  args.legend = list(x = "topright", bty = "n", inset = 0.01, cex = 0.9, text.font = 2)
)

# Add disease names manually to the left of bars
midpoints <- colMeans(barplot_result)
text(x = 0, y = midpoints, labels = diseases, pos = 2, xpd = TRUE, cex = 1.1, font = 2)

# Add percentage labels at the bar ends
text(x = bar_matrix[1, ] + 2, y = barplot_result[1, ],
     labels = paste0(total_pct, "%"), cex = 1, pos = 4, font = 2)

text(x = bar_matrix[2, ] + 2, y = barplot_result[2, ],
     labels = paste0(unique_pct, "%"), cex = 1, pos = 4, font = 2)

dev.off()