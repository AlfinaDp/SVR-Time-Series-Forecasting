library(e1071)
library(readxl)
library(foreach)
library(doParallel)
library(ggplot2)
library(writexl)

# ------------------- Fungsi ------------------- #
calculate_performance <- function(actual, predicted) {
  mse <- mean((actual - predicted)^2, na.rm = TRUE)
  rmse <- sqrt(mse)
  mae <- mean(abs(actual - predicted), na.rm = TRUE)
  return(list(mse = mse, rmse = rmse, mae = mae))
}

create_lag_features <- function(data_vector, L) {
  df_lag <- as.data.frame(embed(data_vector, L + 1))
  colnames(df_lag) <- c("target", paste0("lag", 1:L))
  return(na.omit(df_lag))
}

determine_epsilon_range <- function(data_vector) {
  base_epsilon <- 0.1 * sd(diff(data_vector), na.rm = TRUE)
  epsilon_range <- c(
    base_epsilon * 0.1,
    base_epsilon * 0.5,
    base_epsilon,
    base_epsilon * 2,
    base_epsilon * 5
  )
  epsilon_range <- pmax(epsilon_range, 0.001)
  epsilon_range <- pmin(epsilon_range, 0.5)
  return(sort(unique(round(epsilon_range, 4))))
}

split_ts <- function(data, test_size = 0.3) {
  n <- length(data)
  test_start <- floor(n * (1 - test_size)) + 1
  list(
    train = data[1:(test_start - 1)],
    test = data[test_start:n]
  )
}

optimize_svr <- function(data_vector, L_values, cost_range, gamma_range, epsilon_range, n_folds = 5) {
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  clusterExport(cl, c("create_lag_features", "calculate_performance"))
  
  param_grid <- expand.grid(L = L_values, cost = cost_range, gamma = gamma_range, epsilon = epsilon_range)
  
  results <- foreach(i = 1:nrow(param_grid), .combine = rbind, .packages = "e1071") %dopar% {
    L <- param_grid$L[i]
    cost <- param_grid$cost[i]
    gamma <- param_grid$gamma[i]
    epsilon <- param_grid$epsilon[i]
    
    df_lag <- create_lag_features(data_vector, L)
    fold_size <- floor(nrow(df_lag) / n_folds)
    cv_rmse <- numeric(n_folds)
    
    for (fold in 1:n_folds) {
      val_index <- ((fold - 1) * fold_size + 1):(fold * fold_size)
      train_data <- df_lag[-val_index, ]
      val_data <- df_lag[val_index, ]
      
      model <- svm(
        target ~ .,
        data = train_data,
        type = "eps-regression",
        kernel = "radial",
        cost = cost,
        gamma = gamma,
        epsilon = epsilon
      )
      
      pred <- predict(model, val_data)
      rmse_val <- calculate_performance(val_data$target, pred)$rmse
      cv_rmse[fold] <- rmse_val
    }
    if (i == 1) {
      fold_result_df <- data.frame(
        fold = 1:n_folds,
        rmse = cv_rmse
      )
      saveRDS(fold_result_df, "fold_result_1.rds")
    }
      
      data.frame(
        L = L,
        cost = cost,
        gamma = gamma,
        epsilon = epsilon,
        avg_rmse = mean(cv_rmse),
        sd_rmse = sd(cv_rmse)
      )
    }
  
  stopCluster(cl)
  return(results)
}

# ------------------- Eksekusi ------------------- #

# 1. Load data dan normalisasi
file_path <- "C:\\Users\\Lenovo\\Downloads\\Data Dummy Per Sheet - Musiman Aditif 1200.xlsx"
df <- read_excel(file_path, sheet = 4)
data_vector <- df[[2]]
data_normalized <- (data_vector - min(data_vector)) / (max(data_vector) - min(data_vector))

# 2. Split data menjadi training & testing (tanpa lag dulu)
split_data <- split_ts(data_normalized, test_size = 0.3)
train_raw <- split_data$train
test_raw <- split_data$test

# 3. Optimasi parameter dengan train_raw
L_values <- 1:15
cost_range <- c(0.1, 1, 10, 100)
gamma_range <- c(0.001, 0.01, 0.1)
epsilon_range <- determine_epsilon_range(train_raw)
n_folds <- 5
cat("Rentang epsilon yang diuji:", epsilon_range, "\n")
param_grid <- expand.grid(L = L_values, cost = cost_range, gamma = gamma_range, epsilon = epsilon_range)

cat("Total kombinasi parameter:", nrow(param_grid), "\n")
#Tunning Svr
set.seed(123)
optim_results <- optimize_svr(train_raw, L_values, cost_range, gamma_range, epsilon_range, n_folds = 5)

#Iterasi Terbaik
best_results <- head(optim_results[order(optim_results$avg_rmse), ], 5)
print(best_results)

# Baca hasil dari iterasi pertama
fold_iterasi_pertama <- readRDS("fold_result_1.rds")
print(fold_iterasi_pertama)
param_grid[1, ]


# 4. Ambil parameter terbaik
best_params <- optim_results[which.min(optim_results$avg_rmse), ]
best_L <- best_params$L
cat("\nPARAMETER OPTIMAL NORMALISASI:",
    "\nL =", best_L,
    "\nCost =", best_params$cost,
    "\nGamma =", best_params$gamma,
    "\nEpsilon =", best_params$epsilon,
    "\nRMSE =", best_params$avg_rmse, "\n")

# 5. Terapkan lag dengan L terbaik
train_data <- create_lag_features(train_raw, best_L)
test_data <- create_lag_features(test_raw, best_L)

# 6. Final training dengan data yang telah dilag
final_model <- svm(
  target ~ .,
  data = train_data,
  type = "eps-regression",
  kernel = "radial",
  cost = best_params$cost,
  gamma = best_params$gamma,
  epsilon = best_params$epsilon
)

# 7. Evaluasi performa
train_pred <- predict(final_model, train_data[-1])
test_pred <- predict(final_model, test_data[-1])
train_perf <- calculate_performance(train_data$target, train_pred)
test_perf <- calculate_performance(test_data$target, test_pred)

cat("\nPERFORMANCE FINAL NORMALISASI:",
    "\n[Training] MSE =", train_perf$mse, " RMSE =", train_perf$rmse, " MAE =", train_perf$mae,
    "\n[Testing]  MSE =", test_perf$mse, " RMSE =", test_perf$rmse, " MAE =", test_perf$mae, "\n")

# 8. Prediksi 1 langkah ke depan
last_values <- tail(data_normalized, best_L)
next_input <- as.data.frame(t(last_values))
colnames(next_input) <- paste0("lag", 1:best_L)
pred_normalized <- predict(final_model, next_input)
pred_actual <- pred_normalized * (max(data_vector) - min(data_vector)) + min(data_vector)
cat("\nPREDIKSI 1 MINGGU DEPAN:", pred_actual, "\n")
# Jumlah langkah ke depan yang ingin diprediksi
n_steps <- 7  # misalnya prediksi 7 hari ke depan
L <- best_L   # panjang lag input (jumlah fitur)
input_values <- tail(data_normalized, L)  # ambil L data terakhir yang tersedia

pred_normalized_all <- numeric(n_steps)

for (j in 1:n_steps) {
  # Ambil L nilai terakhir sebagai input (gabungkan prediksi sebelumnya jika j > 1)
  if (j == 1) {
    current_input <- input_values
  } else {
    current_input <- c(pred_normalized_all[(j-1):1], tail(input_values, L - (j - 1)))
  }
  
  # Bentuk data frame sesuai format input model
  input_df <- as.data.frame(t(current_input))
  colnames(input_df) <- paste0("lag", 1:L)
  
  # Lakukan prediksi
  pred_j <- predict(final_model, input_df)
  
  # Simpan prediksi normalized
  pred_normalized_all[j] <- pred_j
}

# Denormalisasi seluruh hasil prediksi
pred_actual_all <- pred_normalized_all * (max(data_vector) - min(data_vector)) + min(data_vector)

# Buat dataframe hasil prediksi
pred_df <- data.frame(
  Step = paste0("t + ", 1:n_steps),
  Prediksi = round(pred_actual_all, 6)
)

# Cetak dalam format tabel
cat("\nHasil Prediksi", n_steps, "Langkah ke Depan:\n")
print(pred_df, row.names = FALSE)


# 9. Visualisasi hasil
results_df <- data.frame(
  Index = seq_along(c(train_data$target, test_data$target)),
  Actual = c(train_data$target, test_data$target),
  Predicted = c(train_pred, test_pred),
  Type = c(rep("Training", length(train_pred)), rep("Testing", length(test_pred)))
)

ggplot(results_df, aes(x = Index)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  geom_vline(xintercept = length(train_pred), linetype = "dashed") +
  annotate("text", x = length(train_pred)/2, y = max(results_df$Actual), label = "Training") +
  annotate("text", x = length(train_pred) + length(test_pred)/2, y = max(results_df$Actual), label = "Testing") +
  labs(title = "Perbandingan Nilai Aktual dan Prediksi",
       y = "Nilai Normalisasi",
       color = "") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()
ggsave("Sheet 4 1200 normal.jpg", width = 10, height = 6, dpi = 300)
# 10. Denormalisasi dan perbandingan dengan data asli
denorm <- function(norm_value) {
  norm_value * (max(data_vector) - min(data_vector)) + min(data_vector)
}

results_df$Actual_Denorm <- denorm(results_df$Actual)
results_df$Predicted_Denorm <- denorm(results_df$Predicted)

offset_train <- best_L + 1
actual_train_true <- data_vector[offset_train:(offset_train + nrow(train_data) - 1)]

offset_test <- length(train_raw) + best_L + 1
actual_test_true <- data_vector[offset_test:(offset_test + nrow(test_data) - 1)]

results_df$Actual_True <- c(actual_train_true, actual_test_true)

# Validasi kecocokan denormalisasi
cek_kecocokan <- all.equal(round(results_df$Actual_Denorm, 5), round(results_df$Actual_True, 5))
cat("\nCEK KESESUAIAN DENORMALISASI VS DATA ASLI:", cek_kecocokan, "\n")

# Pastikan calculate_performance sudah ada
calculate_performance <- function(actual, predicted) {
  list(
    mse = mean((actual - predicted)^2),
    rmse = sqrt(mean((actual - predicted)^2)),
    mae = mean(abs(actual - predicted))
  )
}
# 1. Bagi hasil prediksi dan aktual ke data training dan testing
n_train <- nrow(train_data)
n_test  <- nrow(test_data)

# Ambil dari kolom denormalisasi
train_actual  <- results_df$Actual_True[1:n_train]
test_actual   <- results_df$Actual_True[(n_train + 1):(n_train + n_test)]

train_pred    <- results_df$Predicted_Denorm[1:n_train]
test_pred     <- results_df$Predicted_Denorm[(n_train + 1):(n_train + n_test)]

# 2. Hitung performa
train_perf_denorm <- calculate_performance(train_actual, train_pred)
test_perf_denorm  <- calculate_performance(test_actual, test_pred)

# 3. Tampilkan hasil
cat("\nPERFORMANCE FINAL DENORMALISASI:",
    "\n[Training] MSE =", round(train_perf_denorm$mse, 6),
    " RMSE =", round(train_perf_denorm$rmse, 6),
    " MAE =", round(train_perf_denorm$mae, 6),
    "\n[Testing]  MSE =", round(test_perf_denorm$mse, 6),
    " RMSE =", round(test_perf_denorm$rmse, 6),
    " MAE =", round(test_perf_denorm$mae, 6), "\n")



ggplot(results_df, aes(x = Index)) +
  geom_line(aes(y = Actual_Denorm, color = "Actual (Asli)")) +
  geom_line(aes(y = Predicted_Denorm, color = "Prediksi (Asli)")) +
  geom_vline(xintercept = length(train_pred), linetype = "dashed") +
  annotate("text", x = length(train_pred)/2, y = max(results_df$Actual_Denorm), label = "Training") +
  annotate("text", x = length(train_pred) + length(test_pred)/2, y = max(results_df$Actual_Denorm), label = "Testing") +
  labs(title = "Perbandingan Nilai Aktual dan Prediksi (Skala Asli)",
       y = "Nilai Asli",
       color = "") +
  scale_color_manual(values = c("Actual (Asli)" = "blue", "Prediksi (Asli)" = "red")) +
  theme_minimal()

# Simpan hasil
write_xlsx(results_df, "Sheet 4 1200.xlsx")
ggsave("Sheet 4 1200.jpg", width = 10, height = 6, dpi = 300)
