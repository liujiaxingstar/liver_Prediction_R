# Load dataset
liver <- read.csv(file = "C:\\Users\\Administrator\\Desktop\\liverkeep.csv", stringsAsFactors = FALSE)

# Backup original data (contains complete Albumin column)
original <- liver

# Introduce missing values: randomly delete 116 values in Albumin column
set.seed(200)
liver[sample(1:nrow(liver), 116), "Albumin"] <- NA

# Display missing information
sub <- which(is.na(liver$Albumin))
cat("Rows with missing Albumin:", sub, "\n")
cat("Any missing value in Albumin:", any(is.na(liver$Albumin)), "\n")
cat("Number of missing values in Albumin:", sum(is.na(liver$Albumin)), "\n")

# ========== Convert all non-numeric columns to numeric ==========
# Gender: Male -> 1, Female -> 0
liver$Gender <- ifelse(liver$Gender == "Male", 1, 0)

# Convert other character/factor columns to numeric codes
for (col in names(liver)) {
  if (is.character(liver[[col]])) {
    liver[[col]] <- as.numeric(as.factor(liver[[col]]))
  } else if (is.factor(liver[[col]])) {
    liver[[col]] <- as.numeric(liver[[col]])
  }
}

# Check that all columns are numeric
if (!all(sapply(liver, is.numeric))) {
  stop("Non-numeric column detected. Please check data preprocessing.")
}

# Load required packages
library(randomForest)

# Helper function: compute 1D Wasserstein distance
wasserstein_1d <- function(x, y, n_grid = 100) {
  probs <- seq(0, 1, length.out = n_grid + 2)[-c(1, n_grid + 2)]
  qx <- quantile(x, probs, na.rm = TRUE, type = 1)
  qy <- quantile(y, probs, na.rm = TRUE, type = 1)
  mean(abs(qx - qy))
}

# Adaptive MissForest function
adaptive_missforest <- function(X_miss, c = 0.02, K = 3, max_iter = 20) {
  X_imp <- X_miss
  n <- nrow(X_imp)
  p <- ncol(X_imp)
  
  # 1. Median initialization (all columns should be numeric)
  for (j in 1:p) {
    col <- X_imp[, j]
    if (is.numeric(col)) {
      X_imp[is.na(col), j] <- median(col, na.rm = TRUE)
    } else {
      stop(sprintf("Column %d is not numeric. Please check preprocessing.", j))
    }
  }
  
  # 2. Compute column-wise standard deviations for threshold
  sigma <- apply(X_imp, 2, sd, na.rm = TRUE)
  epsilon <- c / n * sum(sigma)
  
  # 3. Record missing pattern
  missing_mask <- is.na(X_miss)
  
  # 4. Iteration
  stable <- 0
  for (t in 1:max_iter) {
    X_old <- X_imp
    # Sort variables by increasing missing rate
    miss_rate <- colMeans(missing_mask)
    var_order <- order(miss_rate)
    
    for (j in var_order) {
      obs_idx <- which(!missing_mask[, j])  # observed indices
      mis_idx <- which(missing_mask[, j])   # missing indices
      
      if (length(mis_idx) == 0) next
      
      # Train random forest using other variables to predict current variable
      rf <- randomForest(
        x = X_imp[obs_idx, -j, drop = FALSE],
        y = X_imp[obs_idx, j],
        ntree = 100,
        mtry = max(1, floor((p-1)/3))
      )
      
      # Predict missing values
      if (length(mis_idx) > 0) {
        pred <- predict(rf, X_imp[mis_idx, -j, drop = FALSE])
        X_imp[mis_idx, j] <- pred
      }
    }
    
    # Compute global distribution change via Wasserstein distance
    delta <- 0
    for (j in 1:p) {
      delta_j <- wasserstein_1d(X_old[, j], X_imp[, j])
      delta <- delta + delta_j
    }
    delta <- delta / p
    
    # Update stability counter
    if (delta < epsilon) {
      stable <- stable + 1
    } else {
      stable <- 0
    }
    
    cat(sprintf("Iter %d: delta = %.6f, stable = %d\n", t, delta, stable))
    
    if (stable >= K) {
      cat("Converged!\n")
      break
    }
  }
  
  return(X_imp)
}

# Run adaptive MissForest imputation
set.seed(123)
liver_filled <- adaptive_missforest(liver)

# Extract true and imputed values at missing positions
missing_idx <- which(is.na(liver$Albumin))
true_vals <- original$Albumin[missing_idx]
pred_vals <- liver_filled[missing_idx, "Albumin"]

# Compute error metrics
MAE <- mean(abs(true_vals - pred_vals))
MSE <- mean((true_vals - pred_vals)^2)
RMSE <- sqrt(MSE)
MAPE <- mean(abs((true_vals - pred_vals) / true_vals)) * 100

# Output results
cat("\n===== Imputation Error Results =====\n")
cat(sprintf("MAE : %.4f\n", MAE))
cat(sprintf("MSE : %.4f\n", MSE))
cat(sprintf("RMSE: %.4f\n", RMSE))
cat(sprintf("MAPE: %.2f%%\n", MAPE))