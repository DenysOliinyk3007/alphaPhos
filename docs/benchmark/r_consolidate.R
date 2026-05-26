# Minimal R wrapper around the canonical Hogrebe `consolidate()` function.
# The function below is copied VERBATIM from the decompiled PeptideCollapse.R
# (lines 1064-1129), extracted from PluginPeptideCollapse.dll v1.4.4.0.
#
# Usage:
#   Rscript r_consolidate.R <input.tsv> <output.tsv>
#
# Input TSV format (one row per precursor, multiple sites stacked):
#   site_key      precursor_id   <sample1>   <sample2>   ...   <sampleN>
#
# Output TSV format (one row per site, after consolidation):
#   site_key      <sample1>   <sample2>   ...   <sampleN>
#
# Pre-filter: drops precursors with <=1 non-NA cell across sample columns
# (matches the caller-side filter in PeptideCollapse.R line 1148).

suppressMessages(library(data.table))

# ---- consolidate (verbatim from PeptideCollapse.R:1064-1129) --------------
consolidate <- function(cons, cond){
  #check if only NA -> return data table with 1 row filled only with NA immediately
  if(cons[, all(is.na(.SD))]){
    return(as.data.table(matrix(rep(NA, length(cond)), nrow = 1, dimnames = list(NULL, cond))))
  }

  #sort rows from lowest median intensity to highest, to later make sure that (if needed), lowest median rows are kicked out first
  cons <- cons[order(cons[, apply(.SD, 1, function(x) median(x, na.rm=T)), .SDcols = cond])]

  #check if consolidation neccessary at all -> check if all conditions have no missing values
  while(cons[, any(!sapply(.SD, function(x) length(which(!is.na(x))))[cons[, sapply(.SD, function(x) length(which(!is.na(x))))]>0]==.N)]){
    #create loop variables for first iteration
    tempc <- NULL

    #start for loop iteration
    for(num in (1:cons[, .N])){
      #check if any NA in this row; if yes, execute normalization, if not simply copy values
      if(any(is.na(unlist(transpose(cons)[, num, with=F])))){
        #create logical vector reading out NA entries
        logv <- is.na(unlist(transpose(cons)[, num, with=F]))

        #this line calculates the missing value substitution values, and then reorders them together with the non-missing values
        if(length(which(logv))==1){
          tempc <- rbind(tempc, c(median(sapply(transpose(cons)[, -num, with=F], function(x) median(unlist(transpose(cons)[, num, with=F])/x, na.rm=T)*x[logv]), na.rm=T),
                                  unlist(transpose(cons)[, num, with=F], use.names = F)[!logv])[order(c(which(logv), which(!logv)))])
        } else {
          tempc <- rbind(tempc, c(apply(sapply(transpose(cons)[, -num, with=F], function(x) median(unlist(transpose(cons)[, num, with=F])/x, na.rm=T)*x[logv]), 1,
                                        function(x) median(x, na.rm=T)),
                                  unlist(transpose(cons)[, num, with=F], use.names = F)[!logv])[order(c(which(logv), which(!logv)))])
        }

      } else {
        tempc <- rbind(tempc, unlist(transpose(cons)[, num, with=F]))
      }
    }

    #when for loop done, check if tempc matrix equals starting matrix
    if(identical(cons, setNames(as.data.table(tempc), cond))){
      cons <- cons[-(grep(min(apply(cons, 1, function(x) length(which(!is.na(x))))), apply(cons, 1, function(x) length(which(!is.na(x)))))[1]), ]
    } else {
      cons <- setNames(as.data.table(tempc), cond)
    }
  }

  #sum all intensities in normal intensity space -> put higher weight on high intensities
  return(setNames(cons[, lapply(.SD, function(x) sum(x))], cond))
}

# ---- driver ---------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if(length(args) != 2){
  stop("Usage: Rscript r_consolidate.R <input.tsv> <output.tsv>")
}
input_file <- args[1]
output_file <- args[2]

cat("[R] reading", input_file, "\n")
dt <- fread(input_file, sep="\t", na.strings=c("NA", ""))
cat("[R] rows:", nrow(dt), ", cols:", ncol(dt), "\n")

sample_cols <- setdiff(names(dt), c("site_key", "precursor_id"))
cat("[R] sample cols:", length(sample_cols), "\n")

# Pre-filter: keep precursors with > 1 non-NA across sample columns
dt[, n_nonna := apply(.SD, 1, function(x) sum(!is.na(x))), .SDcols = sample_cols]
dt <- dt[n_nonna > 1, ]
dt[, n_nonna := NULL]
cat("[R] after pre-filter (>1 non-NA):", nrow(dt), "\n")

# Apply consolidate per site
unique_sites <- unique(dt$site_key)
cat("[R] consolidating", length(unique_sites), "sites...\n")
result <- dt[, consolidate(.SD, sample_cols), by=site_key, .SDcols=sample_cols]

cat("[R] writing", output_file, ", rows:", nrow(result), "\n")
fwrite(result, output_file, sep="\t", na="NA")
cat("[R] done\n")
