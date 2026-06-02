library(randomForest)
library(rpart)
library(mice)
library(missForest)
library(ipred)
library(FNN)
library(caret)

# Load data 
liver <- read.csv(file = "liverkeep.csv", stringsAsFactors = FALSE)

# Convert all non-numeric columns to numeric
for (col in names(liver)) {
  if (is.factor(liver[[col]])) {
    liver[[col]] <- as.numeric(as.character(liver[[col]]))
  } else if (is.character(liver[[col]])) {
    liver[[col]] <- as.numeric(factor(liver[[col]]))
  }
}
if (!all(sapply(liver, is.numeric))) {
  stop("Non-numeric columns detected after conversion. Please check data.")
}

# Temporarily fill missing values in other columns (except Albumin) with median
albumin_col <- which(names(liver) == "Albumin")
for (j in 1:ncol(liver)) {
  if (j == albumin_col) next
  if (any(is.na(liver[, j]))) {
    med <- median(liver[, j], na.rm = TRUE)
    liver[is.na(liver[, j]), j] <- med
  }
}

# Backup original Albumin values
original_albumin <- liver$Albumin

# Artificially create 6 missing values in Albumin column
set.seed(123)
missing_idx <- sample(1:nrow(liver), 6)
liver$Albumin[missing_idx] <- NA

cat("Indices with missing Albumin:", missing_idx, "\n")
cat("Number of missing values in Albumin:", sum(is.na(liver$Albumin)), "\n")

# Split into training (complete) and testing (missing)
train_idx <- which(!is.na(liver$Albumin))
test_idx  <- which(is.na(liver$Albumin))

X_train <- liver[train_idx, setdiff(names(liver), "Albumin")]
y_train <- liver[train_idx, "Albumin"]
X_test  <- liver[test_idx, setdiff(names(liver), "Albumin")]

y_true <- original_albumin[test_idx]

# Results container
results <- data.frame(
  Method = c("Mean", "Median", "KNN", "CART", "MICE", "MissForest", "Bagging"),
  MAE = NA, MSE = NA, RMSE = NA, MAPE = NA
)

# Error calculation function
calc_errors <- function(actual, predicted) {
  mae  <- mean(abs(actual - predicted))
  mse  <- mean((actual - predicted)^2)
  rmse <- sqrt(mse)
  mape <- mean(abs((actual - predicted) / actual)) * 100
  return(c(MAE = mae, MSE = mse, RMSE = rmse, MAPE = mape))
}

# 1. Mean imputation
y_pred_mean <- rep(mean(y_train), length(test_idx))
results[1, 2:5] <- calc_errors(y_true, y_pred_mean)

# 2. Median imputation
y_pred_median <- rep(median(y_train), length(test_idx))
results[2, 2:5] <- calc_errors(y_true, y_pred_median)

# 3. KNN imputation (k=5)
knn_fit <- knn.reg(train = X_train, test = X_test, y = y_train, k = 5)
y_pred_knn <- knn_fit$pred
results[3, 2:5] <- calc_errors(y_true, y_pred_knn)

# 4. CART regression tree
train_data_cart <- data.frame(X_train, Albumin = y_train)
cart_model <- rpart(Albumin ~ ., data = train_data_cart, method = "anova")
y_pred_cart <- predict(cart_model, newdata = X_test)
results[4, 2:5] <- calc_errors(y_true, y_pred_cart)

# 5. MICE (Multiple Imputation by Chained Equations) using predictive mean matching
temp_data <- liver
mice_imp <- mice(temp_data, method = "rf", m = 1, maxit = 5, seed = 123, printFlag = FALSE)
completed <- complete(mice_imp, 1)
y_pred_mice <- completed[test_idx, "Albumin"]
results[5, 2:5] <- calc_errors(y_true, y_pred_mice)

# 6. MissForest
set.seed(123)
missForest_imp <- missForest(temp_data, verbose = FALSE)
completed_mf <- missForest_imp$ximp
y_pred_mf <- completed_mf[test_idx, "Albumin"]
results[6, 2:5] <- calc_errors(y_true, y_pred_mf)

# 7. Bagging with ipred
train_data_bag <- data.frame(X_train, Albumin = y_train)
bag_model <- bagging(Albumin ~ ., data = train_data_bag, nbagg = 50, coob = TRUE)
y_pred_bag <- predict(bag_model, newdata = X_test)
results[7, 2:5] <- calc_errors(y_true, y_pred_bag)

# Print results with formatted MAPE
results$MAPE <- round(results$MAPE, 2)
print(results)

# Optional: save results to CSV
# write.csv(results, "imputation_results.csv", row.names = FALSE)

# Session info for reproducibility
cat("\n--- Session Info ---\n")
sessionInfo()