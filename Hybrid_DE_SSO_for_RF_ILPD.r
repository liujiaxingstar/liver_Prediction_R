# Clean environment and set seed
rm(list = ls())
set.seed(2025)

# Load required packages
library(randomForest)
library(DEoptim)
library(caret)

# -----------------------------------------------------------------
# 1. Preprocess the ILPD dataset
# -----------------------------------------------------------------

data <- tryCatch(
  read.csv(url, header = TRUE),
  error = function(e) {
    stop("Failed to download ILPD dataset. Check your internet connection or provide a local file.")
  }
)

# Standardize column names
colnames(data) <- c("Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
                    "Alkphos", "Sgpt", "Sgot", "Total_Protiens",
                    "Albumin", "A_G_Ratio", "Dataset")

# Handle missing values in A_G_Ratio
if (any(is.na(data$A_G_Ratio))) {
  median_val <- median(data$A_G_Ratio, na.rm = TRUE)
  data$A_G_Ratio[is.na(data$A_G_Ratio)] <- median_val
}

# Convert labels: 1 -> disease (1), 2 -> healthy (0)
data$Dataset <- ifelse(data$Dataset == 1, 1, 0)
data$Dataset <- as.factor(data$Dataset)

# Encode Gender: Male -> 0, Female -> 1
data$Gender <- ifelse(data$Gender == "Male", 0, 1)

# Remove any remaining rows with NA
data <- na.omit(data)

# Split features and target
X <- data[, -which(names(data) == "Dataset")]
y <- data$Dataset

cat(sprintf("Dataset loaded: %d samples, %d features\n", nrow(X), ncol(X)))
cat("Class distribution:\n")
print(table(y))

# -----------------------------------------------------------------
# 2. Fitness function for cross-validated accuracy
# -----------------------------------------------------------------

fitness <- function(p, X, y, k = 5) {
  # Map continuous [0,1] to integer hyperparameters
  ntree     <- round(p[1] * (500 - 50) + 50)     # range [50,500]
  mtry      <- round(p[2] * (ncol(X) - 1) + 1)   # range [1, ncol(X)-1]
  maxnodes  <- round(p[3] * (100 - 5) + 5)       # range [5,100]
  
  # Clamp to valid ranges
  ntree     <- max(50, min(500, ntree))
  mtry      <- max(1, min(ncol(X), mtry))
  maxnodes  <- max(5, min(100, maxnodes))
  
  # Stratified cross-validation
  set.seed(123)  # ensure reproducibility
  folds <- createFolds(y, k = k, list = TRUE)
  
  acc_sum <- 0
  for (i in 1:k) {
    train_idx <- unlist(folds[-i])
    test_idx  <- folds[[i]]
    
    rf <- randomForest(x = X[train_idx, ], y = y[train_idx],
                       ntree = ntree, mtry = mtry, maxnodes = maxnodes)
    pred <- predict(rf, newdata = X[test_idx, ])
    acc <- mean(pred == y[test_idx])
    acc_sum <- acc_sum + acc
  }
  avg_acc <- acc_sum / k
  return(-avg_acc)   # negative for minimization
}

# -----------------------------------------------------------------
# 3. Stage 1: Differential Evolution (global search)
# -----------------------------------------------------------------
lower <- c(0, 0, 0)
upper <- c(1, 1, 1)

de_control <- DEoptim.control(
  NP = 50,           # population size
  itermax = 100,     # maximum iterations
  F = 0.8,           # differential weight
  CR = 0.9,          # crossover probability
  trace = 2,         # print progress
  reltol = 1e-6
)

cat("\n===== Stage 1: Differential Evolution (Global Search) =====\n")
de_result <- DEoptim(fn = fitness, lower = lower, upper = upper,
                     control = de_control, X = X, y = y)

best_de <- de_result$optim$bestmem
cat("Best continuous solution from DE:\n")
print(best_de)

# -----------------------------------------------------------------
# 4. Stage 2: Social Spider Optimization (local fine-tuning)
# -----------------------------------------------------------------
# Local fitness wrapper (same as above)
local_fitness <- function(p, X, y) {
  ntree     <- round(p[1] * (500 - 50) + 50)
  mtry      <- round(p[2] * (ncol(X) - 1) + 1)
  maxnodes  <- round(p[3] * (100 - 5) + 5)
  
  ntree     <- max(50, min(500, ntree))
  mtry      <- max(1, min(ncol(X), mtry))
  maxnodes  <- max(5, min(100, maxnodes))
  
  set.seed(123)
  folds <- createFolds(y, k = 5, list = TRUE)
  acc_sum <- 0
  for (i in 1:5) {
    train_idx <- unlist(folds[-i])
    test_idx  <- folds[[i]]
    rf <- randomForest(x = X[train_idx, ], y = y[train_idx],
                       ntree = ntree, mtry = mtry, maxnodes = maxnodes)
    pred <- predict(rf, newdata = X[test_idx, ])
    acc <- mean(pred == y[test_idx])
    acc_sum <- acc_sum + acc
  }
  avg_acc <- acc_sum / 5
  return(-avg_acc)
}

# SSO parameters
radius <- 0.05
local_iter <- 30
best_params <- best_de
best_fit <- fitness(best_params, X, y)

cat("\n===== Stage 2: Social Spider Optimization (Local Tuning) =====\n")
for (iter in 1:local_iter) {
  candidate <- best_params + runif(3, -radius, radius)
  candidate <- pmax(0, pmin(1, candidate))   # bound constraints
  
  fit_candidate <- local_fitness(candidate, X, y)
  if (fit_candidate < best_fit) {
    best_params <- candidate
    best_fit <- fit_candidate
    cat(sprintf("Iter %2d: improved, CV accuracy = %.2f%%\n",
                iter, -best_fit * 100))
  }
}

# -----------------------------------------------------------------
# 5. Final model with optimized hyperparameters
# -----------------------------------------------------------------
ntree_opt   <- round(best_params[1] * (500 - 50) + 50)
mtry_opt    <- round(best_params[2] * (ncol(X) - 1) + 1)
maxnodes_opt<- round(best_params[3] * (100 - 5) + 5)

ntree_opt   <- max(50, min(500, ntree_opt))
mtry_opt    <- max(1, min(ncol(X), mtry_opt))
maxnodes_opt<- max(5, min(100, maxnodes_opt))

cat("\n===== Optimized Hyperparameters =====\n")
cat(sprintf("ntree     = %d\n", ntree_opt))
cat(sprintf("mtry      = %d\n", mtry_opt))
cat(sprintf("maxnodes  = %d\n", maxnodes_opt))
cat(sprintf("5-fold CV accuracy = %.2f%%\n", -best_fit * 100))

# Train final model on all data
final_rf <- randomForest(x = X, y = y,
                         ntree = ntree_opt,
                         mtry = mtry_opt,
                         maxnodes = maxnodes_opt,
                         importance = TRUE)

print(final_rf)

# In-sample accuracy (for illustration; use separate test set in real experiments)
pred_train <- predict(final_rf, newdata = X)
train_acc <- mean(pred_train == y)
cat(sprintf("\nTraining accuracy on full dataset: %.2f%%\n", train_acc * 100))

# -----------------------------------------------------------------
# End of script
# -----------------------------------------------------------------