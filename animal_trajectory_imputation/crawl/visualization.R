library(crawl)
library(ggplot2)
library(raster)

setwd('~/Desktop/thesis/animal_trajectory_imputation/crawl')

df = read.csv('y.csv', header = FALSE)
names(df) = c('x', 'y')
jul = read.csv('jul.csv', header = FALSE)$V1
base_date <- as.POSIXct("2017-01-01", tz = "UTC")

# Generate 'time' column
df$time <- base_date + (jul * 86400) # 86400 seconds in a day


crps_loss <- function(Y_samples, Y_true) {
  
  B <- dim(Y_samples)[1]
  L <- dim(Y_samples)[2]
  
  numerator <- 0
  denominator <- 0
  
  for (l in 1:L) {
    
    samples <- Y_samples[, l]
    z <- Y_true[l]
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
  
  
  return(numerator / denominator)
}


# Calculate p% of the number of rows
num_rows <- nrow(df)

eval_mask <- read.csv('eval_mask.csv', header = FALSE)
eval_mask[1,] = 0



df_train = df
df_train$x[eval_mask[,1]==1] <- NA
df_train$y[eval_mask[,1]==1] <- NA




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

# Examine the predictions
head(predictions)


# evaluate



x_pred = predictions$mu.x
y_pred = predictions$mu.y


x_pred_sd = predictions$se.mu.x
y_pred_sd = predictions$se.mu.y

x_pred_samples = matrix(nrow=50, ncol=length(x_pred))
y_pred_samples = matrix(nrow=50, ncol=length(y_pred))
for (i in 1:length(x_pred)){
  x_pred_samples[, i] = rnorm(50, x_pred[i], x_pred_sd[i])
  y_pred_samples[, i] = rnorm(50, y_pred[i], y_pred_sd[i])
}


# predictions
write.csv(predictions, 'y_hat.csv')
# samples
write.csv(x_pred_samples, 'x_pred_samples.csv', row.names=FALSE)
write.csv(y_pred_samples, 'y_pred_samples.csv', row.names=FALSE)

