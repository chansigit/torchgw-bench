#!/usr/bin/env Rscript
# cisTopic LDA fitting, invoked via subprocess from the C2 pipeline.
#
# Reads: a sparse binary peak × cell matrix saved as MatrixMarket (.mtx)
# Writes: a cells × topics float matrix saved as .npz (via Matrix::writeMM
#         on the topic proportion matrix, plus a meta file).
#
# Usage:
#   Rscript cistopic_lda.R <input.mtx> <output.csv> [n_topics=50] [n_iter=500]

suppressPackageStartupMessages({
    library(cisTopic)
    library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
in_mtx <- args[[1]]
peak_ids_file <- args[[2]]
out_csv <- args[[3]]
n_topics <- if (length(args) >= 4) as.integer(args[[4]]) else 50L
n_iter <- if (length(args) >= 5) as.integer(args[[5]]) else 500L
n_cores <- if (length(args) >= 6) as.integer(args[[6]]) else 4L

cat(sprintf("[cisTopic.R] loading %s ...\n", in_mtx))
X <- readMM(in_mtx)                      # rows = peaks, cols = cells
cat(sprintf("[cisTopic.R] matrix dim = %d x %d (peaks x cells)\n",
             nrow(X), ncol(X)))
peak_ids <- readLines(peak_ids_file)     # one chr:start-end per line
stopifnot(length(peak_ids) == nrow(X))
rownames(X) <- peak_ids                  # cisTopic expects genomic coords
colnames(X) <- paste0("cell_", seq_len(ncol(X)))

cat("[cisTopic.R] building cisTopicObject ...\n")
obj <- createcisTopicObject(X, project.name = "c2_pbmc",
                              min.cells = 0, min.regions = 0,
                              keepCountsMatrix = FALSE)

t0 <- Sys.time()
cat(sprintf("[cisTopic.R] runCGSModels(topic=%d, iter=%d, nCores=%d) ...\n",
             n_topics, n_iter, n_cores))
obj <- runCGSModels(obj, topic = n_topics, seed = 0,
                     nCores = n_cores, iterations = n_iter,
                     addModels = FALSE)
t_fit <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("[cisTopic.R] LDA fit done in %.1fs\n", t_fit))

obj <- selectModel(obj, type = "maximum")
cellAssignments <- modelMatSelection(obj, target = "cell",
                                        method = "Probability")
# cellAssignments: topics x cells — transpose for (cells, topics)
V_atac <- t(cellAssignments)
cat(sprintf("[cisTopic.R] V_atac dim = %d x %d (cells x topics)\n",
             nrow(V_atac), ncol(V_atac)))
write.table(V_atac, file = out_csv, sep = ",",
             row.names = FALSE, col.names = FALSE)
cat(sprintf("[cisTopic.R] wrote %s\n", out_csv))
