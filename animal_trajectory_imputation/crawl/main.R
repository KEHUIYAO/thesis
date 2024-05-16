library(crawl)
library(ggplot2)
library(raster)
library(abind)

crps_loss <- function(Y_samples, Y_true) {

  B <- dim(Y_samples)[1]
  L <- dim(Y_samples)[2]
  C <- dim(Y_samples)[3]
  
  numerator <- 0
  denominator <- 0
  
  for (l in 1:L) {
    
    for (c in 1:C) {
      samples <- Y_samples[, l, c]
      z <- Y_true[l, c]
      crps = 0
      
      for (i in 1:19) {
        q <- quantile(samples, i * 0.05, names = FALSE)
        indicator <- ifelse(z < q, 1, 0)
        loss <- 2 * (i * 0.05 - indicator) * (z - q) / 19
        crps <- crps + loss
      }
      
      numerator <- numerator + crps
      denominator <- denominator + abs(z)
    }
  }
  
  
  return(numerator / denominator)
}



tag_num_list = c(6618, 6981, 7948, 6788, 8165, 7952, 7029, 6870, 6582, 6015, 5171, 6650, 6544, 7179, 6831, 5671, 5221, 5004, 5183, 5540, 6596, 5143, 6847, 5606, 7733, 5209, 7779, 6855, 6758, 5072, 7202, 5583, 6903, 6033, 7258, 8177, 5880, 7098, 8143, 7216, 6762, 5225, 7934, 5189, 6073, 6540, 8185, 6528, 6046, 5239, 5181, 7777, 6726, 6794, 5096, 7195, 6010, 6911, 6546, 6965, 7768, 6819, 7175, 5975, 5867, 5979, 7914, 5639, 7977, 5577, 7266, 6631, 5740, 8197, 5094, 6536, 5281, 5675, 6874, 7826, 5659)
mae_list = c()
crps_list = c()
n_obs_list = c()

p = 0.8

for (i in 1:length(tag_num_list)){
  
  tag_num = tag_num_list[i]
  print(tag_num)
  df = read.csv(paste('../Female/TagData/LowTag', tag_num, '.csv', sep=''))
  
  
  # Rename columns 'X' to 'x' and 'Y' to 'y' using base R
  names(df)[names(df) == "X"] <- "x"
  names(df)[names(df) == "Y"] <- "y"
  
  base_date <- as.POSIXct("2017-01-01", tz = "UTC")
  
  # Generate 'time' column
  df$time <- base_date + (df$jul * 86400) # 86400 seconds in a day
  
  # Ensure the 'time' column is in UTC
  attributes(df$time)$tzone <- "UTC"
  
  # extract month, day, and hour
  # Extract month, day, and hour
  df$month <- as.numeric(format(df$time, "%m"))
  df$day <- as.numeric(format(df$time, "%d"))
  df$hour <- as.numeric(format(df$time, "%H"))
  
  
  # Load the covariate .tif file
  covariate_file_path <- paste0('../Female/NLCDClip/LowTag', tag_num, 'NLCDclip.tif')
  covariate_file <- raster(covariate_file_path)
  
  # Extracting the cell numbers for X and Y coordinates
  cells <- cellFromXY(covariate_file, df[, c('x', 'y')])
  df$covariate = as.character(extract(covariate_file, cells))
  
  
  # set up a hold-out validation set
  set.seed(42) # For reproducibility
  
  # Calculate p% of the number of rows
  num_rows <- nrow(df)
 
  rows_to_set_na <- floor(p * num_rows)
  
  # Randomly select rows for 'x' and 'y'
  rows_for_x_y <- sample(2:num_rows, rows_to_set_na)
  
  df_train = df
  df_train$x[rows_for_x_y] <- NA
  df_train$y[rows_for_x_y] <- NA
  
  
  
  
  # Fit model
  fit <- crwMLE(data = df_train, mov.model=~1, attempts = 5)
  
  if (length(fit) == 2){
    next
  }
 
  # fit <- crwMLE(data = df_train, mov.model=~1+month, attempts = 5)
  # fit <- crwMLE(data = df_train, mov.model=~1, err.model = list(x=~1, y=~1), attempts = 10)
  # The general order for fixPar is location error variance parameters (e.g. ellipse, location quality), followed by sigma and beta and, then, the activity parameter.
  # fit <- crwMLE(data = df_train,
  #               mov.model=~1,
  #               err.model = list(x=~1, y=~1),
  #               fixPar = c(5, 5, NA, NA),
  #               attempts = 10)
  
  # fit <- crwMLE(data = df_train,
  #               mov.model=~1,
  #               err.model = list(x=~1, y=~1),
  #               attempts = 10)
  
  
  
  
  
  
  
  # Use crwPredict to predict locations at these halfway points
  predictions <- crwPredict(object.crwFit = fit, predTime = df_train$time, return.type = "flat")
  rownames(predictions) <- NULL
  # Examine the predictions
  # head(predictions)
  
  
  # evaluate
  x_true = df$x[rows_for_x_y]
  y_true = df$y[rows_for_x_y]
  
  x_pred = predictions$mu.x[rows_for_x_y]
  y_pred = predictions$mu.y[rows_for_x_y]
  
  
  x_pred_sd = predictions$se.mu.x[rows_for_x_y]
  y_pred_sd = predictions$se.mu.y[rows_for_x_y]
  
  if (any(is.na(x_pred_sd))){
    next
  }
  
  x_pred_samples = matrix(nrow=100, ncol=length(x_pred))
  y_pred_samples = matrix(nrow=100, ncol=length(y_pred))
  for (i in 1:length(x_pred)){
    x_pred_samples[, i] = rnorm(100, x_pred[i], x_pred_sd[i])
    y_pred_samples[, i] = rnorm(100, y_pred[i], y_pred_sd[i])
  }
  
  
  
  
  
  
  n_obs_list = c(n_obs_list, length(y_true))
  
  # mae
  mae_list = c(mae_list, (mean(abs(x_true - x_pred)) + mean(abs(y_true - y_pred))) / 2)
  
  # crps
  crps = crps_loss(abind(x_pred_samples, y_pred_samples, along=3), cbind(x_true, y_true))
  crps_list = c(crps_list, crps)
  
  
  # print((mean(abs(x_true - x_pred)) + mean(abs(y_true - y_pred))) / 2)
  
  # resid_x = x_true - x_pred
  # resid_y = y_true - y_pred
  # 
  # plot(resid_x)
  # plot(resid_y)
  
  
  # ggplot(df_train, aes(x = x, y = y)) + 
  #   geom_path(color = "blue") + # Draw the path
  #   geom_point(color = "red") + # Mark the locations
  #   theme_minimal() + 
  #   labs(title = "Animal Movement Path",
  #        x = "Longitude",
  #        y = "Latitude") +
  #   coord_fixed() # Keep aspect ratio of 1 to ensure accurate representation
  
  
  # fit$nms
  # fit$par
}

print(sum(mae_list * n_obs_list) / sum(n_obs_list))
print(mean(crps_list))
