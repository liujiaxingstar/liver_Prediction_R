# 1. Load required libraries
library(class)
library(caret)
library(C50)
library(gmodels)
library(cluster)
library(fpc)
library(plotly)
library(e1071)
library(randomForest)
library(caTools)
library(neuralnet)

# 2. Load dataset (modify path if necessary)
data_raw <- read.csv(file = "liverkeep.csv", stringsAsFactors = FALSE)

# 3. Data preprocessing ------------------------------------------------
data <- data_raw

# Check for missing values in Albumin_and_Globulin_Ratio
cat("Missing values in Albumin_and_Globulin_Ratio:\n")
print(table(is.na(data$Albumin_and_Globulin_Ratio)))

# Mean imputation for Albumin_and_Globulin_Ratio
avg_ratio <- mean(data$Albumin_and_Globulin_Ratio, na.rm = TRUE)
data$Albumin_and_Globulin_Ratio[is.na(data$Albumin_and_Globulin_Ratio)] <- avg_ratio

# Convert Label: 1 -> disease (1), 2 -> healthy (0)
data$Label <- ifelse(data$Label == 2, 1, 0)   # 1 = disease, 0 = healthy

# Encode Gender: Male -> 0, Female -> 1
data$Gender <- ifelse(data$Gender == "Male", 0, 1)

# Max-min normalization for numeric features
numeric_cols <- c("Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
                  "Alamine_Aminotransferase", "Aspartate_Aminotransferase",
                  "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio")
for (col in numeric_cols) {
  min_val <- min(data[[col]], na.rm = TRUE)
  max_val <- max(data[[col]], na.rm = TRUE)
  data[[col]] <- (data[[col]] - min_val) / (max_val - min_val)
}

# 4. Split data into training (80%) and testing (20%)
set.seed(2026)
train_idx <- createDataPartition(data$Label, p = 0.8, list = FALSE)
train <- data[train_idx, ]
test  <- data[-train_idx, ]

# Ensure Label is factor for classification
train$Label <- as.factor(train$Label)
test$Label  <- as.factor(test$Label)

cat("\nTraining set size:", nrow(train))
cat("\nTesting set size:", nrow(test))
cat("\nClass distribution in training:\n")
print(prop.table(table(train$Label)))
cat("\nClass distribution in testing:\n")
print(prop.table(table(test$Label)))

# =====================================================================
# 5. Model building and evaluation
# =====================================================================

# Helper function to print confusion matrix and metrics
evaluate_model <- function(pred, true, model_name) {
  cm <- confusionMatrix(pred, true)
  cat("\n", model_name, ":\n")
  cat("Accuracy: ", round(cm$overall["Accuracy"], 4),
      " (Kappa: ", round(cm$overall["Kappa"], 4), ")\n")
  return(cm)
}

# --------------------------- KNN --------------------------------------
train_knn <- train[, c("Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
                       "Alkaline_Phosphotase", "Alamine_Aminotransferase",
                       "Aspartate_Aminotransferase", "Total_Protiens",
                       "Albumin", "Albumin_and_Globulin_Ratio")]
test_knn  <- test[, names(train_knn)]
knn_pred  <- knn(train = train_knn, test = test_knn,
                 cl = train$Label, k = floor(sqrt(nrow(train))))
knn_cm    <- evaluate_model(knn_pred, test$Label, "KNN (k = sqrt(n))")

# --------------------------- C5.0 Decision Tree ------------------------
c50_model <- C5.0(x = train[, !names(train) %in% "Label"],
                  y = train$Label)
c50_pred  <- predict(c50_model, newdata = test)
c50_cm    <- evaluate_model(c50_pred, test$Label, "C5.0 Decision Tree")

# --------------------------- Boosted C5.0 ------------------------------
c50_boost <- C5.0(x = train[, !names(train) %in% "Label"],
                  y = train$Label, trials = 10)
boost_pred <- predict(c50_boost, newdata = test)
boost_cm   <- evaluate_model(boost_pred, test$Label, "C5.0 with Boosting (trials=10)")

# --------------------------- K-Means (unsupervised) --------------------
# Note: K-means is not a classifier; we use it to assign clusters and then match clusters to labels.
set.seed(123)
kmeans_model <- kmeans(x = train[, !names(train) %in% "Label"], centers = 2)
kmeans_cluster <- kmeans_model$cluster
# Map clusters to labels (majority voting)
cluster_to_label <- function(clusters, true_labels) {
  tab <- table(clusters, true_labels)
  mapping <- apply(tab, 1, which.max)
  names(mapping) <- rownames(tab)
  return(mapping[as.character(clusters)])
}
train$kmeans_label <- cluster_to_label(kmeans_cluster, train$Label)
# Predict on test set
test_kmeans_clust <- kmeans(test[, !names(test) %in% "Label"],
                            centers = kmeans_model$centers)$cluster
test_kmeans_pred <- cluster_to_label(test_kmeans_clust, train$Label)   # use training mapping
kmeans_cm <- evaluate_model(as.factor(test_kmeans_pred), test$Label, "K-means (unsupervised)")

# --------------------------- Naive Bayes ------------------------------
nb_model <- naiveBayes(x = train[, !names(train) %in% "Label"],
                       y = train$Label)
nb_pred  <- predict(nb_model, newdata = test)
nb_cm    <- evaluate_model(nb_pred, test$Label, "Naive Bayes")

# --------------------------- Random Forest ----------------------------
rf_model <- randomForest(x = train[, !names(train) %in% "Label"],
                         y = train$Label, ntree = 500, importance = TRUE)
rf_pred  <- predict(rf_model, newdata = test)
rf_cm    <- evaluate_model(rf_pred, test$Label, "Random Forest (ntree=500)")

# --------------------------- Support Vector Machine -------------------
svm_model <- svm(Label ~ ., data = train, kernel = "radial")
svm_pred  <- predict(svm_model, newdata = test)
svm_cm    <- evaluate_model(svm_pred, test$Label, "SVM (radial kernel)")

# --------------------------- Neural Network ---------------------------
# Prepare data for neuralnet (requires all numeric and no factor)
# Note: neuralnet works with numeric response (0/1)
train_nn <- train
test_nn <- test
train_nn$Label <- as.numeric(train_nn$Label) - 1   # convert to 0/1
test_nn$Label  <- as.numeric(test_nn$Label) - 1

# Formula
formula_nn <- as.formula(paste("Label ~", paste(names(train_nn)[!names(train_nn) %in% "Label"], collapse = "+")))

set.seed(300)
nn_model <- neuralnet(formula_nn, data = train_nn,
                      hidden = c(13, 10, 2),
                      act.fct = "logistic",
                      err.fct = "ce",
                      linear.output = FALSE,
                      lifesign = "minimal",
                      rep = 5,
                      algorithm = "rprop+")
# Predict
nn_pred_prob <- compute(nn_model, test_nn[, !names(test_nn) %in% "Label"])$net.result
nn_pred_class <- ifelse(nn_pred_prob > 0.5, 1, 0)
nn_cm <- evaluate_model(as.factor(nn_pred_class), as.factor(test_nn$Label), "Neural Network (13-10-2)")

# =====================================================================
# 6. Summary of results
# =====================================================================
cat("\n=========================== FINAL SUMMARY ===========================\n")
models <- c("KNN", "C5.0", "Boosted C5.0", "K-means", "Naive Bayes",
            "Random Forest", "SVM", "Neural Network")
accuracies <- c(knn_cm$overall["Accuracy"],
                c50_cm$overall["Accuracy"],
                boost_cm$overall["Accuracy"],
                kmeans_cm$overall["Accuracy"],
                nb_cm$overall["Accuracy"],
                rf_cm$overall["Accuracy"],
                svm_cm$overall["Accuracy"],
                nn_cm$overall["Accuracy"])
kappa <- c(knn_cm$overall["Kappa"],
           c50_cm$overall["Kappa"],
           boost_cm$overall["Kappa"],
           kmeans_cm$overall["Kappa"],
           nb_cm$overall["Kappa"],
           rf_cm$overall["Kappa"],
           svm_cm$overall["Kappa"],
           nn_cm$overall["Kappa"])
results <- data.frame(Model = models, Accuracy = round(accuracies, 4),
                      Kappa = round(kappa, 4))
print(results)

cat("\nAnalysis completed.\n")