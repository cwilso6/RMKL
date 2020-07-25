% RMKL Readme
# RMKL
Integrating multiple heterogeneous high throughput data sources is an emerging topic of interest in cancer research. Making decisions based upon metabolomic, genomic, etc. data sources can lead to better prognosis or diagnosis than simply using clinical data alone. Support vector machines (SVM) are not suitable for analyzing multiple data sources in a single analysis. SVM employs the kernel trick, thus it is able to construct nonlinear classification boundaries. However, obtaining an optimal kernel type, hyperparameter, and regularization parameter is a challenging task. Cross validation can be employed to make these selections. Ultimately, there may not be one single optimal kernel, but rather a convex combination of several kernel representations of the data. Methods that produce a classification rule based on a convex combination of candidate kernels are referred to as multiple kernel learning (MKL) methods

You can install RMKL from GitHub. If you already have a previous version of RMKL installed, you can use that to install the new version:

```{r}
install.packages("devtools")
library(devtools)
devtools::install_github("cwilso6/RMKL")
library(RMKL)
```

# Requirements
In order for RMKL to work properly, the following packages are required:

* caret
* kernlab

See MKL_Classification.pdf and MKL_Survival.pdf for illustrative example showing the properties and implementation of MKL for both classification and survival applications.
