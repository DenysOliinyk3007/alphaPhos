# Minimal limma wrapper called via subprocess.
#
# Usage:
#   Rscript limma_wrapper.R <expr.tsv> <meta.tsv> <group_col> <out.tsv>
#
# expr.tsv:  rows = features (sites), columns = sample_id; first column is 'feature'.
# meta.tsv:  columns include sample_id and the group column.
# group_col: the column name in meta.tsv whose values define the two contrast groups
#            (e.g. 'treatment' with levels IgG / PD1).
# out.tsv:   topTable output (one row per feature), columns:
#            feature, logFC, AveExpr, t, P.Value, adj.P.Val, B
# Note: logFC is `level2 - level1` where levels are sorted alphabetically by limma's
#       model.matrix default. For IgG / PD1 this gives PD1 - IgG (P > I).

suppressMessages(library(limma))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript limma_wrapper.R <expr.tsv> <meta.tsv> <group_col> <out.tsv>")
}
expr_file <- args[1]
meta_file <- args[2]
group_col <- args[3]
out_file  <- args[4]

cat("[limma] expr:", expr_file, "\n")
cat("[limma] meta:", meta_file, "\n")

expr <- read.table(expr_file, sep = "\t", header = TRUE, check.names = FALSE,
                   row.names = 1, na.strings = c("NA", ""))
meta <- read.table(meta_file, sep = "\t", header = TRUE, check.names = FALSE,
                   stringsAsFactors = FALSE)

# Align samples: ensure column order of expr matches sample_id rows of meta
sample_ids <- meta$sample_id
stopifnot(all(sample_ids %in% colnames(expr)))
expr <- as.matrix(expr[, sample_ids])

group <- factor(meta[[group_col]])
cat("[limma] groups:", paste(levels(group), collapse=", "), "\n")
cat("[limma] sample sizes:", paste(table(group), collapse=", "), "\n")

design <- model.matrix(~ group)
colnames(design) <- c("Intercept", "group_effect")
cat("[limma] design:\n"); print(design)

fit <- lmFit(expr, design)
fit <- eBayes(fit, trend = TRUE)

tt <- topTable(fit, coef = "group_effect", number = Inf, sort.by = "none")
tt$feature <- rownames(tt)
tt <- tt[, c("feature", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B")]

cat("[limma] writing", nrow(tt), "rows to", out_file, "\n")
write.table(tt, file = out_file, sep = "\t", row.names = FALSE, quote = FALSE,
            na = "NA")
cat("[limma] done\n")
