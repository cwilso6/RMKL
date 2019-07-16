#' tcga.small
#'
#' Small example from TCGA. THese data are from TCGA ovaian data set. There are 65 genes
#' included. These genes were prioritorized by having the smallest p-values derived 
#' from using a t-test on the comparing the mean gene expression between survivors and 
#' non-survivors. We used a 3 year cutoff, i.e. patients who lived longer than 3 years 
#' after diagonsis were considered survivors, while patients who died within 3 years
#' of their diagnosis were considered non-survivors  
#'
#' @format This data object is matrix with 283 samples (rows) and 66 columns, where the last 
#' column is the survival status, where -1 corrsponds to patients being non-suvivors and 1
#' are patients that did survive. 
"tcga.small"
