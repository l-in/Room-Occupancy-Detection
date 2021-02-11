library(tseries)
library(forecast)
library(changepoint)
library(changepoint.np)
library(rpart)
library(randomForest)
library(e1071)
library(gbm)
library(neuralnet)


### Import data ###
data.occ.train = read.table(file = "datatraining.txt", header = TRUE, sep = ",")
data.occ.test = read.table(file = "datatest.txt", header = TRUE, sep = ",") # dates before training
data.occ.test.2 = read.table(file = "datatest2.txt", header = TRUE, sep = ",") # dates after training


### Data wrangling ###
# extract month and year from each date 
data.occ.train$year.month.date = substr(data.occ.train$date, start = 1, stop = 10)

# add id to each observation to facilitate plotting
data.occ.train$id.rel = 1:nrow(data.occ.train)


### Check normality of  predictors ###
qqnorm(data.occ.train$Temperature, xlab = "Temperature Quantiles")
qqnorm(data.occ.train$Humidity, xlab = "Humidity Quantiles")
qqnorm(data.occ.train$Light, xlab = "Light Quantiles")
qqnorm(data.occ.train$CO2, xlab = "CO2 Quantiles")
qqnorm(data.occ.train$HumidityRatio, xlab = "HumidityRatio Quantiles")


### Check non-stationarity of time series ###
# Apply Dickey-Fuller test to each predictor 
adf.test(data.occ.train$Temperature)
adf.test(data.occ.train$Humidity)
adf.test(data.occ.train$Light)
adf.test(data.occ.train$CO2)
adf.test(data.occ.train$HumidityRatio)


### Check for missing values ###
anyNA(data.occ.train)
anyNA(data.occ.test)
anyNA(data.occ.test.2)


### Exploratory Data Analysis ###
# Plot time series of each predictor
plot(data.occ.train$id, data.occ.train$Temperature, type = "l", 
     xlab = "Year-Month-Day", ylab = "Temperature (Celsius)",
     main = "Temperature in an Office Room", xaxt = "n")
axis(side = 1, at = 1:length(data.occ.train$Temperature), labels = data.occ.train$year.month.date)

plot(data.occ.train$id, data.occ.train$Humidity, type = "l", xlab = "Year-Month-Day", 
     ylab = "Relative Humidity (%)",
     main = "Relative Humidity in an Office Room", xaxt = "n")
axis(side = 1, at = 1:length(data.occ.train$Humidity), labels = data.occ.train$year.month.date)

plot(data.occ.train$id, data.occ.train$Light, pch = 20, xlab = "Year-Month-Day", ylab = "Light (Lux)", 
     main = "Light in an Office Room", xaxt = "n", cex = 0.5)
axis(side = 1, at = 1:length(data.occ.train$Light), labels = data.occ.train$year.month.date)

plot(data.occ.train$id, data.occ.train$CO2, pch = 20, xlab = "Year-Month-Day", ylab = "CO2 (ppm)", 
     main = "CO2 in an Office Room", cex = 0.5, xaxt = "n")
axis(side = 1, at = 1:length(data.occ.train$CO2), labels = data.occ.train$year.month.date)

plot(data.occ.train$id, data.occ.train$HumidityRatio, type = "l", xlab = "Year-Month-Day", 
     ylab = "Humidity Ratio (kgwater-vapor/kg-air)", main = "Humidity Ratio in an Office Room", xaxt = "n")
axis(side = 1, at = 1:length(data.occ.train$HumidityRatio), labels = data.occ.train$year.month.date)


### Temperature: plot seasons (days) beside each other ###
days = unique(data.occ.train$year.month.date)
colours = c("azure3", "deepskyblue4", "aquamarine", "blueviolet", "deeppink", "darkorange", 
            "darkgoldenrod")

for (i in 1:length(days)) {

  if (i == 1) {
    plot(data.occ.train$id.rel[data.occ.train$year.month.date == days[i]], 
         data.occ.train$Temperature[data.occ.train$year.month.date == days[i]], 
         xlim = c(0, data.occ.train$id.rel[nrow(data.occ.train)]), 
         ylim = c(min(data.occ.train$Temperature), max(data.occ.train$Temperature)),
         pch = 20, cex = 0.7, col = colours[i], xlab = "Day", ylab = "Temperature (Celsius)", xaxt = "n")
  } else { 
    points(data.occ.train$id.rel[data.occ.train$year.month.date == days[i]], 
           data.occ.train$Temperature[data.occ.train$year.month.date == days[i]], 
           pch = 20, cex = 0.7, col = colours[i], xaxt = "n")
  }
}


# add mean for each day to plots
# find mean and sd for each group
agg.mean.temp = aggregate(x = data.occ.train$Temperature, by = list(data.occ.train$year.month.date), 
                          FUN = mean)
agg.sd.temp = aggregate(x = data.occ.train$Temperature, by = list(data.occ.train$year.month.date), 
                        FUN = sd)
agg.var.temp = aggregate(x = data.occ.train$Temperature, by = list(data.occ.train$year.month.date), 
                         FUN = var)


# assign daily mean temp to each observation of each day 
temp.mean = rep.int(0, nrow(data.occ.train))

for (i in 1:length(days)) {
  temp.mean[data.occ.train$year.month.date == days[i]] = agg.mean.temp$x[i]
}


# plot daily mean 
points(data.occ.train$id.rel, temp.mean, pch = 20, cex = 0.3)


# find median of the ids (ie median index) of every day (x values)
a = aggregate(x = data.occ.train$id.rel, by = list(data.occ.train$year.month.date), 
              FUN = median)
midpoints = floor(a$x)
midpoints = rep(midpoints, 2)
b = c(agg.mean.temp$x + agg.sd.temp$x, agg.mean.temp$x - agg.sd.temp$x)
points(midpoints, b, pch = 175)


# add lines between mean + sd and mean - sd for each day
for (i in 1:(length(midpoints)/2)) {
  lines(midpoints[which(midpoints == midpoints[i])], b[which(midpoints == midpoints[i])])
}


# add axis values
axis(side = 1, at = 1:length(data.occ.train$id.rel), labels = data.occ.train$date)


### Humidity: plot seasons (days) beside each other ###
for (i in 1:length(days)) {
  
  if (i == 1) {
    plot(data.occ.train$id.rel[data.occ.train$year.month.date == days[i]], 
         data.occ.train$Humidity[data.occ.train$year.month.date == days[i]], 
         xlim = c(0, data.occ.train$id.rel[nrow(data.occ.train)]), 
         ylim = c(min(data.occ.train$Humidity), max(data.occ.train$Humidity)),
         pch = 20, cex = 0.7, col = colours[i], xlab = "Day", ylab = "Humidity (%)", xaxt = "n")
  } else { 
    points(data.occ.train$id.rel[data.occ.train$year.month.date == days[i]], 
           data.occ.train$Humidity[data.occ.train$year.month.date == days[i]], 
           pch = 20, cex = 0.7, col = colours[i], xaxt = "n")
  }
}


# add mean for each day to plots
# find mean and sd for each group
agg.mean.hum = aggregate(x = data.occ.train$Humidity, by = list(data.occ.train$year.month.date), 
                         FUN = mean)
agg.sd.hum = aggregate(x = data.occ.train$Humidity, by = list(data.occ.train$year.month.date), 
                       FUN = sd)
agg.var.hum = aggregate(x = data.occ.train$Humidity, by = list(data.occ.train$year.month.date), 
                       FUN = var)

# assign daily mean temp to each observation of each day 
hum.mean = rep.int(0, nrow(data.occ.train))

for (i in 1:length(days)) {
  hum.mean[data.occ.train$year.month.date == days[i]] = agg.mean.hum$x[i]
}


# plot daily mean 
points(data.occ.train$id.rel, hum.mean, pch = 20, cex = 0.1)


# find median of the ids (ie median index) of every day (x values)
b = c(agg.mean.hum$x + agg.sd.hum$x, agg.mean.hum$x - agg.sd.hum$x)
points(midpoints, b, pch = 175)


# add lines between mean + sd and mean - sd for each day
for (i in 1:(length(midpoints)/2)) {
  lines(midpoints[which(midpoints == midpoints[i])], b[which(midpoints == midpoints[i])])
}


# add axis values
axis(side = 1, at = 1:length(data.occ.train$id.rel), labels = data.occ.train$date)


### Light: plot seasons (days) side by side ###
for (i in 1:length(days)) {

  if (i == 1) {
    plot(data.occ.train$id.rel[data.occ.train$year.month.date == days[i]], 
         data.occ.train$Light[data.occ.train$year.month.date == days[i]], 
         xlim = c(0, data.occ.train$id.rel[nrow(data.occ.train)]), 
         ylim = c(min(data.occ.train$Light), max(data.occ.train$Light)),
         pch = 20, cex = 0.7, col = colours[i], xlab = "Day", ylab = "Light (Lux)", xaxt = "n")
  } else { 
    points(data.occ.train$id.rel[data.occ.train$year.month.date == days[i]], 
           data.occ.train$Light[data.occ.train$year.month.date == days[i]], 
           pch = 20, cex = 0.7, col = colours[i], xaxt = "n")
  }
}


# add mean for each day to plots
# find mean and sd for each group
agg.mean.light = aggregate(x = data.occ.train$Light, by = list(data.occ.train$year.month.date), 
                           FUN = mean)
agg.sd.light = aggregate(x = data.occ.train$Light, by = list(data.occ.train$year.month.date), 
                         FUN = sd)
agg.var.light = aggregate(x = data.occ.train$Light, by = list(data.occ.train$year.month.date), 
                         FUN = var)


# assign daily mean temp to each observation of each day 
light.mean = rep.int(0, nrow(data.occ.train))

for (i in 1:length(days)) {
  light.mean[data.occ.train$year.month.date == days[i]] = agg.mean.light$x[i]
}


# plot daily mean 
points(data.occ.train$id.rel, light.mean, pch = 20, cex = 0.3)


# find median of the ids (ie median index) of every day (x values)
b = c(agg.mean.light$x + agg.sd.light$x, agg.mean.light$x - agg.sd.light$x)
points(midpoints, b, pch = 175)


# add lines between mean + sd and mean - sd for each day
for (i in 1:(length(midpoints)/2)) {
  lines(midpoints[which(midpoints == midpoints[i])], b[which(midpoints == midpoints[i])])
}


# add axis values
axis(side = 1, at = 1:length(data.occ.train$id.rel), labels = data.occ.train$date)


### C02: plot seasons (days) side by side ###
for (i in 1:length(days)) {

  if (i == 1) {
    plot(data.occ.train$id.rel[data.occ.train$year.month.date == days[i]], 
         data.occ.train$CO2[data.occ.train$year.month.date == days[i]], 
         xlim = c(0, data.occ.train$id.rel[nrow(data.occ.train)]), 
         ylim = c(min(data.occ.train$CO2), max(data.occ.train$CO2)),
         pch = 20, cex = 0.7, col = colours[i], xlab = "Day", ylab = "CO2 (ppm)", xaxt = "n")
  } else { 
    points(data.occ.train$id.rel[data.occ.train$year.month.date == days[i]], 
           data.occ.train$CO2[data.occ.train$year.month.date == days[i]], 
           pch = 20, cex = 0.7, col = colours[i], xaxt = "n")
  }
}


# add mean for each day to plots
# find mean and sd for each group
agg.mean.co2 = aggregate(x = data.occ.train$CO2, by = list(data.occ.train$year.month.date), 
                         FUN = mean)
agg.sd.co2 = aggregate(x = data.occ.train$CO2, by = list(data.occ.train$year.month.date), 
                       FUN = sd)
agg.var.co2 = aggregate(x = data.occ.train$CO2, by = list(data.occ.train$year.month.date), 
                       FUN = var)

# assign daily mean temp to each observation of each day 
co2.mean = rep.int(0, nrow(data.occ.train))

for (i in 1:length(days)) {
  co2.mean[data.occ.train$year.month.date == days[i]] = agg.mean.co2$x[i]
}


# plot daily mean 
points(data.occ.train$id.rel, co2.mean, pch = 20, cex = 0.3)


# find median of the ids (ie median index) of every day (x values)
b = c(agg.mean.co2$x + agg.sd.co2$x, agg.mean.co2$x - agg.sd.co2$x)
points(midpoints, b, pch = 175)


# add lines between mean + sd and mean - sd for each day
for (i in 1:(length(midpoints)/2)) {
  lines(midpoints[which(midpoints == midpoints[i])], b[which(midpoints == midpoints[i])])
}


# add axis values
axis(side = 1, at = 1:length(data.occ.train$id.rel), labels = data.occ.train$date)


### Humidity ratio: plot seasons (days) side by side ###
for (i in 1:length(days)) {

  if (i == 1) {
    plot(data.occ.train$id.rel[data.occ.train$year.month.date == days[i]], 
         data.occ.train$HumidityRatio[data.occ.train$year.month.date == days[i]], 
         xlim = c(0, data.occ.train$id.rel[nrow(data.occ.train)]), 
         ylim = c(min(data.occ.train$HumidityRatio), max(data.occ.train$HumidityRatio)),
         pch = 20, cex = 0.7, col = colours[i], xlab = "Day", 
         ylab = "Humidity Ratio (kgwater-vapor/kg-air)", xaxt = "n")
  } else { 
    points(data.occ.train$id.rel[data.occ.train$year.month.date == days[i]], 
           data.occ.train$HumidityRatio[data.occ.train$year.month.date == days[i]], 
           pch = 20, cex = 0.7, col = colours[i], xaxt = "n")
  }
}


# add mean for each day to plots
# find mean and sd for each group
agg.mean.hr = aggregate(x = data.occ.train$HumidityRatio, by = list(data.occ.train$year.month.date), 
                        FUN = mean)
agg.sd.hr = aggregate(x = data.occ.train$HumidityRatio, by = list(data.occ.train$year.month.date), 
                      FUN = sd)
agg.var.hr = aggregate(x = data.occ.train$HumidityRatio, by = list(data.occ.train$year.month.date), 
                      FUN = var)


# assign daily mean temp to each observation of each day 
hr.mean = rep.int(0, nrow(data.occ.train))

for (i in 1:length(days)) {
  hr.mean[data.occ.train$year.month.date == days[i]] = agg.mean.hr$x[i]
}


# plot daily mean 
points(data.occ.train$id.rel, hr.mean, pch = 20, cex = 0.3)


# find median of the ids (ie median index) of every day (x values)
b = c(agg.mean.hr$x + agg.sd.hr$x, agg.mean.hr$x - agg.sd.hr$x)
points(midpoints, b, pch = 175)


# add lines between mean + sd and mean - sd for each day
for (i in 1:(length(midpoints)/2)) {
  lines(midpoints[which(midpoints == midpoints[i])], b[which(midpoints == midpoints[i])])
}


# add axis values
axis(side = 1, at = 1:length(data.occ.train$id.rel), labels = data.occ.train$date)


# add legend to separate plot (same for all plots)
plot(1:100, 1:100, xaxt="n", col = "white")
legend(x = 2, y = 100, legend = days, pch = 19, col = colours, 
       cex = 0.8, pt.cex = 1.3, bty = "n", ncol = 1)



### Seasonal plot of each predictor (superimposed) ###
# view starting times each day
data.occ.train$date[match(unique(data.occ.train$year.month.date), data.occ.train$year.month.date)]


# assign id to every minute in the day, starting with 1 at 00:00 and ending at 23:59
id.by.min = rep.int(0, nrow(data.occ.train))

for (i in 1:length(days)) {

    id.by.min[data.occ.train$year.month.date == days[i]] = 
      1:sum(data.occ.train$year.month.date == days[i])
}


# first day starts observations at 17:51 which is 1071 minutes into the day
# add 1070 because the first id.by.min is already set to 1
id.by.min[data.occ.train$year.month.date == days[1]] = 
  id.by.min[data.occ.train$year.month.date == days[1]] + 1070

data.occ.train$id.by.min = id.by.min


### Temperature: plot seasons
for (i in 1:length(days)) {
  
  if (i == 1) {
    plot(data.occ.train$id.by.min[data.occ.train$year.month.date == days[i]],
         data.occ.train$Temperature[data.occ.train$year.month.date == days[i]],
         xlim = c(0, max(data.occ.train$id.by.min)), 
         ylim = c(min(data.occ.train$Temperature), max(data.occ.train$Temperature)),
         type = "l", cex = 0.7, col = colours[i], xlab = "Minute", 
         ylab = "Temperature (Celsius)")
  } else {
    points(data.occ.train$id.by.min[data.occ.train$year.month.date == days[i]],
           data.occ.train$Temperature[data.occ.train$year.month.date == days[i]],
           type = "l", cex = 0.7, col = colours[i], xaxt = "n")
  }
}


### Humidity: plot seasons
for (i in 1:length(days)) {
  
  if (i == 1) {
    plot(data.occ.train$id.by.min[data.occ.train$year.month.date == days[i]],
         data.occ.train$Humidity[data.occ.train$year.month.date == days[i]],
         xlim = c(0, max(data.occ.train$id.by.min)), 
         ylim = c(min(data.occ.train$Humidity), max(data.occ.train$Humidity)),
         type = "l", cex = 0.7, col = colours[i], xlab = "Minute", 
         ylab = "Humidity (%)")
  } else {
    points(data.occ.train$id.by.min[data.occ.train$year.month.date == days[i]],
           data.occ.train$Humidity[data.occ.train$year.month.date == days[i]],
           type = "l", cex = 0.7, col = colours[i], xaxt = "n")
  }
}


### Light: plot seasons
for (i in 1:length(days)) {
  
  if (i == 1) {
    plot(data.occ.train$id.by.min[data.occ.train$year.month.date == days[i]],
         data.occ.train$Light[data.occ.train$year.month.date == days[i]],
         xlim = c(0, max(data.occ.train$id.by.min)), 
         ylim = c(min(data.occ.train$Light), max(data.occ.train$Light)),
         pch = 20, cex = 0.7, col = colours[i], xlab = "Minute", 
         ylab = "Light (Celsius)")
  } else {
    points(data.occ.train$id.by.min[data.occ.train$year.month.date == days[i]],
           data.occ.train$Light[data.occ.train$year.month.date == days[i]],
           pch = 20, cex = 0.7, col = colours[i], xaxt = "n")
  }
}


### CO2: plot seasons
for (i in 1:length(days)) {
  
  if (i == 1) {
    plot(data.occ.train$id.by.min[data.occ.train$year.month.date == days[i]],
         data.occ.train$CO2[data.occ.train$year.month.date == days[i]],
         xlim = c(0, max(data.occ.train$id.by.min)), 
         ylim = c(min(data.occ.train$CO2), max(data.occ.train$CO2)),
         type = "l", cex = 0.7, col = colours[i], xlab = "Minute", 
         ylab = "CO2 (ppm)")
  } else {
    points(data.occ.train$id.by.min[data.occ.train$year.month.date == days[i]],
           data.occ.train$CO2[data.occ.train$year.month.date == days[i]],
           type = "l", cex = 0.7, col = colours[i], xaxt = "n")
  }
}


### Humidity ratio: plot seasons
for (i in 1:length(days)) {
  
  if (i == 1) {
    plot(data.occ.train$id.by.min[data.occ.train$year.month.date == days[i]],
         data.occ.train$HumidityRatio[data.occ.train$year.month.date == days[i]],
         xlim = c(0, max(data.occ.train$id.by.min)), 
         ylim = c(min(data.occ.train$HumidityRatio), max(data.occ.train$HumidityRatio)),
         type = "l", cex = 0.7, col = colours[i], xlab = "Minute", 
         ylab = "Humidity Ratio (kgwater-vapor/kg-air)")
  } else {
    points(data.occ.train$id.by.min[data.occ.train$year.month.date == days[i]],
           data.occ.train$HumidityRatio[data.occ.train$year.month.date == days[i]],
           type = "l", cex = 0.7, col = colours[i], xaxt = "n")
  }
}


### Plot seasonal means against variance
plot(agg.mean.temp$x, agg.var.temp$x, xlab = "Seasonal (Daily) Temperature Mean",
     ylab = "Seasonal (Daily) Temperature Variance")
plot(agg.mean.light$x, agg.var.light$x, xlab = "Seasonal (Daily) Light Mean",
     ylab = "Seasonal (Daily) Light Variance")
plot(agg.mean.hum$x, agg.var.hum$x, xlab = "Seasonal (Daily) Humidity Mean",
     ylab = "Seasonal (Daily) Humidity Variance")
plot(agg.mean.co2$x, agg.var.co2$x, xlab = "Seasonal (Daily) CO2 Mean",
     ylab = "Seasonal (Daily) CO2 Variance")
plot(agg.mean.hr$x, agg.var.hr$x, xlab = "Seasonal (Daily) Humidity Ratio Mean",
     ylab = "Seasonal (Daily) Humidity Ratio Variance")


### Plot seasonal means against time
plot(1:length(agg.mean.temp$x), agg.mean.temp$x, xlab = "Time", ylab = "Mean Temperature", type = "l",
     xaxt = "n", ylim = c(19, 22.5), main = "Temp: Mean vs. Time")
c = c(agg.mean.temp$x + agg.var.temp$x, agg.mean.temp$x - agg.var.temp$x)
axis.vals = rep(1:length(agg.mean.temp$x), 2)
points(axis.vals, c, pch = 175)

for (i in 1:(length(agg.mean.temp$x))) {
  lines(axis.vals[which(axis.vals == axis.vals[i])], c[which(axis.vals == axis.vals[i])])
}

axis(side = 1, at = 1:length(agg.mean.temp$x), labels = days)

d = c(agg.mean.light$x + agg.var.light$x, agg.mean.light$x - agg.var.light$x)
plot(1:length(agg.mean.light$Group.1), agg.mean.light$x, xlab = "Time", ylab = "Mean Light", type = "l",
     xaxt = "n", main = "Light: Mean vs. Time", ylim = c(min(d), max(d)))
points(axis.vals, d, pch = 175) # variance too large to plot for this y axis range

for (i in 1:(length(agg.mean.light$x))) {
  lines(axis.vals[which(axis.vals == axis.vals[i])], d[which(axis.vals == axis.vals[i])])
}

axis(side = 1, at = 1:length(agg.mean.light$x), labels = days)


e = c(agg.mean.hum$x + agg.var.hum$x, agg.mean.hum$x - agg.var.hum$x)
plot(1:length(agg.mean.hum$Group.1), agg.mean.hum$x, xlab = "Time", ylab = "Mean Humidity", type = "l",
     xaxt = "n", ylim = c(min(e), max(e)), main = "Humidity: Mean vs. Time")

points(axis.vals, e, pch = 175)

for (i in 1:(length(agg.mean.hum$x))) {
  lines(axis.vals[which(axis.vals == axis.vals[i])], e[which(axis.vals == axis.vals[i])])
}

axis(side = 1, at = 1:length(agg.mean.hum$x), labels = days)


f = c(agg.mean.co2$x + agg.var.co2$x, agg.mean.co2$x - agg.var.co2$x)
plot(1:length(agg.mean.co2$Group.1), agg.mean.co2$x, xlab = "Time", ylab = "Mean CO2", type = "l",
     xaxt = "n", main = "CO2: Mean vs. Time", ylim = c(min(f), max(f)))
points(axis.vals, f, pch = 175)

for (i in 1:(length(agg.mean.co2$x))) {
  lines(axis.vals[which(axis.vals == axis.vals[i])], f[which(axis.vals == axis.vals[i])])
}

axis(side = 1, at = 1:length(agg.mean.co2$x), labels = days)


g = c(agg.mean.hr$x + agg.var.hr$x, agg.mean.hr$x - agg.var.hr$x)
plot(1:length(agg.mean.hr$Group.1), agg.mean.hr$x, xlab = "Time", ylab = "Mean Humidity Ratio", type = "l",
     xaxt = "n", main = "HR: Mean vs. Time")
points(axis.vals, g, pch = 175) # variance is very small

for (i in 1:(length(agg.mean.hr$x))) {
  lines(axis.vals[which(axis.vals == axis.vals[i])], g[which(axis.vals == axis.vals[i])])
}

axis(side = 1, at = 1:length(agg.mean.hr$x), labels = days)


### Plot scatterplots of predictor pairs
scatterplot.colours = c("#00AFBB", "#E7B800")
pairs(data.occ.train[,2:7], pch = 20, col = scatterplot.colours[as.factor(data.occ.train$Occupancy)])


### Create correlation table for predictors
cor(data.occ.train[,2:6], method = "spearman")


### Plot autocorrelation
ggacf(data.occ.train$Temperature, lag.max = 2000, main = "Temp")
acf(data.occ.train$Humidity, lag.max = 2000, main= "Humidity")
acf(data.occ.train$Light, lag.max = 2000, main = "Light")
acf(data.occ.train$CO2, lag.max = 2000, main = "CO2")
acf(data.occ.train$HumidityRatio, lag.max = 2000, main = "HR")



### Transformations and decompositions
# Create time series
ts.temp = ts(data.occ.train$Temperature, start = c(2015, 1071), frequency = 1440)
ts.hum = ts(data.occ.train$Humidity, start = c(2015, 1071), frequency = 1440)
ts.light = ts(data.occ.train$Light, start = c(2015, 1071), frequency = 1440)
ts.co2 = ts(data.occ.train$CO2, start = c(2015, 1071), frequency = 1440)


### Decompose time series of each predictor using STL method
stl.temp = stl(ts.temp, t.window = 9, s.window = "periodic", robust = TRUE)
plot(stl.temp, main="Temperature Decomposition")

stl.hum = stl(ts.hum, t.window = 9, s.window = "periodic", robust = TRUE)
plot(stl.hum, main="Humidity Decomposition")

stl.light = stl(ts.light, t.window = 151, s.window = "periodic", robust = TRUE)
plot(stl.light, main="Light Decomposition")

stl.co2 = stl(ts.co2, t.window = 91, s.window = "periodic", robust = TRUE)
plot(stl.co2, main="CO2 Decomposition")


### Time series modelling
# Scale predictors using min-max method
ts.temp = (ts.temp-min(ts.temp))/(max(ts.temp)-min(ts.temp))
ts.hum = (ts.hum-min(ts.hum))/(max(ts.hum)-min(ts.hum))
ts.light = (ts.light-min(ts.light))/(max(ts.light)-min(ts.light))
ts.co2 = (ts.co2-min(ts.co2))/(max(ts.co2)-min(ts.co2))
# Note: normality not achieved.

data.occ.train$Temperature = (data.occ.train$Temperature-min(data.occ.train$Temperature))/(max(data.occ.train$Temperature)-min(data.occ.train$Temperature))
data.occ.train$Humidity = (data.occ.train$Humidity-min(data.occ.train$Humidity))/(max(data.occ.train$Humidity)-min(data.occ.train$Humidity))
data.occ.train$Light = (data.occ.train$Light-min(data.occ.train$Light))/(max(data.occ.train$Light)-min(data.occ.train$Light))
data.occ.train$CO2 = (data.occ.train$CO2-min(data.occ.train$CO2))/(max(data.occ.train$CO2)-min(data.occ.train$CO2))


# Check for normality after transformations
hist(ts.temp, probability = TRUE, xlim = c(-0.2, 1), xlab = "Temperature", 
     main = "Temperature")
curve(dnorm(x, mean(ts.temp), sd(ts.temp)), add = TRUE)

hist(ts.hum, probability = TRUE, xlim = c(-0.2, 1), xlab = "Humidity", 
     main = "Humidity")
curve(dnorm(x, mean(ts.hum), sd(ts.hum)), add = TRUE)

hist(ts.light, probability = TRUE, xlim = c(-0.2, 1), xlab = "Light", 
     main = "Light")
curve(dnorm(x, mean(ts.light), sd(ts.light)), add = TRUE)

hist(ts.co2, probability = TRUE, xlim = c(-0.2, 1), xlab = "CO2", 
     main = "CO2")
curve(dnorm(x, mean(ts.co2), sd(ts.co2)), add = TRUE)


### 1. Model time series with kernel density smoothing + Fourier series and 
###    kernel density smoothing + ARMA for residuals 

# For temperature:
# Kernel density estimation
kds.temp = ksmooth(data.occ.train$id.rel, data.occ.train$Temperature, "normal", bandwidth=20)
plot(data.occ.train$Temperature, type="l", col="antiquewhite4", ylab = "Temperature",
     main = "Kernel w/o ARMA errors")
lines(kds.temp, col="cadetblue2")

# Fourier series to model seasonality
fourier.temp = fourier(ts.temp, 3) 
plot(data.occ.train$Temperature, type="l", col="antiquewhite4", ylab = "Temperature", 
     main="Kernel with Fourier series")
lines(rowSums(fourier.temp)+kds.temp$y, type="l", col = 4)
# Result: KDE + Fourier not usable. Use only KDS.

# Get residuals and plot them
kds.res.temp = data.occ.train$Temperature-kds.temp$y
plot(kds.res.temp, type="l")

# Model residuals using ARIMA
kds.res.temp.arima = auto.arima(kds.res.temp, seasonal = FALSE)
kds.res.temp.arima.fit = kds.res.temp.arima$fitted

# Model temperature using KDE and ARIMA errors
plot(data.occ.train$Temperature, type="l", col="antiquewhite4", ylab = "Temperature", 
     main = "Kernel with ARMA errors")
lines(kds.temp$y + kds.res.temp.arima.fit, col=4)

# Use RSS to compare KDS with KDS + ARIMA errors
rss.kds.temp = sum((data.occ.train$Temperature-kds.temp$y)^2)
rss.kds.arima.temp = sum((data.occ.train$Temperature-(kds.temp$y + kds.res.temp.arima.fit))^2)
sort(c("w/o arma" = rss.kds.temp, "w arma" = rss.kds.arima.temp))


# For humidity:
# Kernel density estimation
kds.hum = ksmooth(data.occ.train$id.rel, data.occ.train$Humidity, "normal", bandwidth=20)
plot(data.occ.train$Humidity, type="l", col="antiquewhite4", ylab = "Humidity",
     main = "Kernel w/o ARMA errors")
lines(kds.hum, col="cadetblue2")

# Fourier series to model seasonality
fourier.hum = fourier(ts.hum, 3)
plot(data.occ.train$Humidity, type="l", col="antiquewhite4", ylab="Humidity", 
     main="Kernel with Fourier Series")
lines(rowSums(fourier.hum)+kds.hum$y, type="l", col = 4)
# Result: KDE + Fourier not usable. Use only KDS.

# Get residuals and plot them
kds.res.hum = data.occ.train$Humidity-kds.hum$y
plot(kds.res.hum, type="l")

# Model residuals using ARIMA
kds.res.hum.arima = auto.arima(kds.res.hum, seasonal = FALSE)
kds.res.hum.arima.fit = kds.res.hum.arima$fitted

# Model temperature using KDE and ARIMA errors
plot(data.occ.train$Humidity, type="l", col="antiquewhite4", ylab= "Humidity", 
     main = "Kernel with ARMA errors")
lines(kds.hum$y + kds.res.hum.arima.fit, col=4)

# Use RSS to compare KDS with KDS + ARIMA errors
rss.kds.hum = sum((data.occ.train$Humidity-kds.hum$y)^2)
rss.kds.arima.hum = sum((data.occ.train$Humidity-(kds.hum$y + kds.res.hum.arima.fit))^2)
sort(c("w/o arma" = rss.kds.hum, "w arma" = rss.kds.arima.hum))



# For light:
# Kernel density estimation
kds.light = ksmooth(data.occ.train$id.rel, data.occ.train$Light, "normal", bandwidth=5)
plot(data.occ.train$Light, type="l", col="antiquewhite4", ylab = "Light",
     main = "Kernel w/o ARMA errors")
lines(kds.light, col="cadetblue2") 
#may need to change bandwidth depending on structure/assumpt. of model for residuals
#there are outliers here
# do the residuals need to be sig diff from zero to be modelled? check this.

# Fourier series to model seasonality
fourier.light = fourier(ts.light, 5)
plot(data.occ.train$Light, type="l", col="antiquewhite4", ylab="Light",
     main="Kernel with Fourier series")
lines(rowSums(fourier.light) + kds.light$y, type="l", col=4)

# Use RSS to compare KDE and KDE + Fourier
sum((kds.light$y-data.occ.train$Light)^2)
sum((rowSums(fourier.light)+kds.light$y-data.occ.train$Light)^2)
# Result: KDE alone has smaller RSS. Use it to model light.

# Get residuals and plot them
kds.res.light = data.occ.train$Light-kds.light$y
plot(kds.res.light, type="l")

# Model residuals using ARIMA
kds.res.light.arima = auto.arima(kds.res.light, seasonal = FALSE)
kds.res.light.arima.fit = kds.res.light.arima$fitted

# Model temperature using KDE and ARIMA errors
plot(data.occ.train$Light, type="l", col="antiquewhite4", ylab = "Light",
     main = "Kernel with ARMA errors")
lines(kds.light$y + kds.res.light.arima.fit, col=4)

# Use RSS to compare KDS with KDS + ARIMA errors
rss.kds.light = sum((data.occ.train$Light-kds.light$y)^2)
rss.kds.arima.light = sum((data.occ.train$Light-(kds.light$y + kds.res.light.arima.fit))^2)
sort(c("w/o arma" = rss.kds.light, "w arma" = rss.kds.arima.light))



# For CO2: 
# Kernel density estimation
kds.co2 = ksmooth(data.occ.train$id.rel, data.occ.train$CO2, "normal", bandwidth=20)
plot(data.occ.train$CO2, type="l", col="antiquewhite4", ylab = "CO2", 
     main = "Kernel w/o ARMA errors")
lines(kds.co2, col="cadetblue2")

# Fourier series to model seasonality
fourier.co2 = fourier(ts.co2, 3)
plot(data.occ.train$CO2, type="l", col="antiquewhite4", ylab="CO2", main="Kernel with Fourier")
lines(rowSums(fourier.co2)+kds.co2$y, type="l", col = 4)

# Use RSS to compare KDE and KDE + Fourier
sum((kds.co2$y-data.occ.train$CO2)^2)
sum((rowSums(fourier.co2)+kds.co2$y-data.occ.train$CO2)^2)
# Result: KDE alone has smaller RSS. Use it to model CO2.

# Get residuals and plot them
kds.res.co2 = data.occ.train$CO2-kds.co2$y
plot(kds.res.co2, type="l")

# Model residuals using ARIMA
kds.res.co2.arima = auto.arima(kds.res.co2, seasonal = FALSE)
kds.res.co2.arima.fit = kds.res.co2.arima$fitted

# Model temperature using KDE and ARIMA errors
plot(data.occ.train$CO2, type="l", col="antiquewhite4", ylab="CO2",
     main = "Kernel with ARMA errors")
lines(kds.co2$y + kds.res.co2.arima.fit, col=4)

# Use RSS to compare KDS with KDS + ARIMA errors
rss.kds.co2 = sum((data.occ.train$CO2-kds.co2$y)^2)
rss.kds.arima.co2 = sum((data.occ.train$CO2-(kds.co2$y + kds.res.co2.arima.fit))^2)
sort(c("w/o arma" = rss.kds.co2, "w arma" = rss.kds.arima.co2))



### 2. Model time series with spline smoothing + Fourier series 
###    and spline smoothing + ARIMA for residuals

params = seq(from = 0.01, to = 0.1, by=0.01)

# For temperature:
# Spline smoothing
# Choose best smoothing parameter using residual sum of squares
ssr = c()
j=1
for (i in params){
  spline = smooth.spline(data.occ.train$id.rel, data.occ.train$Temperature, 
                         w = rep.int(1, nrow(data.occ.train)),
                         df = nrow(data.occ.train), i)
  ssr[j] = sum((data.occ.train$Temperature - spline$y)^2)
}

# Find index of smoothing parameter with smallest RSS
i.temp = params[match(min(ssr), ssr)]

# Generate model using best smoothing parameter
spline.temp = smooth.spline(data.occ.train$id.rel, data.occ.train$Temperature, 
                            w = rep.int(1, nrow(data.occ.train)),
                            df = nrow(data.occ.train), i.temp)

# Plot time series and fitted spline
plot(data.occ.train$Temperature, type="l", col="antiquewhite4", ylab = "Temperature", 
     main = "Spline w/o ARMA")
lines(spline.temp, col = 4)

# Fourier series to model seasonality
fourier.temp = fourier(ts.temp, 3)
plot(data.occ.train$Temperature, type="l", col="antiquewhite4")
lines(rowSums(fourier.temp)+spline.temp$y, type="l", col = 4)
# Result: Fourier + spline not usable for temp. Use spline only.

# Get residuals and plot them
spline.res.temp = data.occ.train$Temperature-spline.temp$y
plot(spline.res.temp, type="l")

# Model residuals using ARIMA
spline.res.temp.arima = auto.arima(spline.res.temp, seasonal = FALSE)
spline.res.temp.arima.fit = spline.res.temp.arima$fitted

# Model temperature using spline and ARIMA errors
plot(data.occ.train$Temperature, type="l", col="antiquewhite4", ylab ="Temperature",
     main = "Spline with ARMA errors")
lines(spline.temp$y + spline.res.temp.arima.fit, col=4)

# Use RSS to compare spline with spline + ARIMA errors
rss.spline.temp = sum((data.occ.train$Temperature-spline.temp$y)^2)
rss.spline.arima.temp = sum((data.occ.train$Temperature-(spline.temp$y + spline.res.temp.arima.fit))^2)
sort(c("w/o arma" = rss.spline.temp, "w arma" = rss.spline.arima.temp))




# For humidity:
# Spline smoothing
# Choose best smoothing parameter using residual sum of squares
ssr = c()
j=1
for (i in params){
  spline = smooth.spline(data.occ.train$id.rel, data.occ.train$Humidity, 
                         w = rep.int(1, nrow(data.occ.train)),
                         df = nrow(data.occ.train), i)
  ssr[j] = sum((data.occ.train$Humidity - spline$y)^2)
}

# Find index of smoothing parameter with smallest RSS
i.hum = params[match(min(ssr), ssr)]

# Generate model using best smoothing parameter
spline.hum = smooth.spline(data.occ.train$id.rel, data.occ.train$Humidity, 
                            w = rep.int(1, nrow(data.occ.train)),
                            df = nrow(data.occ.train), i.hum)

# Plot time series and fitted spline
plot(data.occ.train$Humidity, type="l", col="antiquewhite4")
lines(spline.hum, col = 4)

# Fourier series to model seasonality
fourier.hum = fourier(ts.hum, 3)
plot(data.occ.train$Humidity, type="l", col="antiquewhite4")
lines(rowSums(fourier.hum)+spline.hum$y, type="l", col = 4)
# Result: Fourier + spline not usable for humidity. Use spline only.

# Get residuals and plot them
spline.res.hum = data.occ.train$Humidity-spline.hum$y
plot(spline.res.hum, type="l")

# Model residuals using ARIMA
spline.res.hum.arima = auto.arima(spline.res.hum, seasonal = FALSE)
spline.res.hum.arima.fit = spline.res.hum.arima$fitted

# Model temperature using spline and ARIMA errors
plot(data.occ.train$Humidity, type="l", col="antiquewhite4")
plot(spline.hum$y + spline.res.hum.arima.fit, col=4)

# Use RSS to compare spline with KDS + ARIMA errors
rss.spline.hum = sum((data.occ.train$Humidity-spline.hum$y)^2)
rss.spline.arima.hum = sum((data.occ.train$Humidity-(spline.hum$y + spline.res.hum.arima.fit))^2)
sort(c("w/o arma" = rss.spline.hum, "w arma" = rss.spline.arima.hum))




# For light:
# Spline smoothing
# Choose best smoothing parameter using residual sum of squares
ssr = c()
j=1
for (i in params){
  spline = smooth.spline(data.occ.train$id.rel, data.occ.train$Light, 
                         w = rep.int(1, nrow(data.occ.train)),
                         df = nrow(data.occ.train), i)
  ssr[j] = sum((data.occ.train$Light - spline$y)^2)
}

# Find index of smoothing parameter with smallest RSS
i.light = params[match(min(ssr), ssr)]

# Generate model using best smoothing parameter
spline.light = smooth.spline(data.occ.train$id.rel, data.occ.train$Light, 
                            w = rep.int(1, nrow(data.occ.train)),
                            df = nrow(data.occ.train), i.light)

# Plot time series and fitted spline
plot(data.occ.train$Light, type="l", col="antiquewhite4")
lines(spline.light, col="cadetblue2")

# Fourier series to model seasonality
fourier.light = fourier(ts.light, 5)
plot(data.occ.train$Light, type="l", col="antiquewhite4")
lines(rowSums(fourier.light) + spline.light$y, type="l", col=4)

# Use RSS to compare spline and spline + Fourier
sum((spline.light$y-data.occ.train$Light)^2)
sum((rowSums(fourier.light) + spline.light$y-data.occ.train$Light)^2)
# Result: spline alone has smaller RSS. Use that to model light.

# Get residuals and plot them
spline.res.light = data.occ.train$Light-spline.light$y
plot(spline.res.light, type="l")

# Model residuals using ARIMA
spline.res.light.arima = auto.arima(spline.res.light, seasonal = FALSE)
spline.res.light.arima.fit = spline.res.light.arima$fitted

# Model temperature using spline and ARIMA errors
plot(data.occ.train$Light, type="l", col="antiquewhite4")
plot(spline.light$y + spline.res.light.arima.fit, col=4)

# Use RSS to compare spline with spline + ARIMA errors
rss.spline.light = sum((data.occ.train$Light-spline.light$y)^2)
rss.spline.arima.light = sum((data.occ.train$Light-(spline.light$y + spline.res.light.arima.fit))^2)
sort(c("w/o arma" = rss.spline.light, "w arma" = rss.spline.arima.light))



# For CO2:
# Spline smoothing
# Choose best smoothing parameter using residual sum of squares
ssr = c()
j=1
for (i in params){
  spline = smooth.spline(data.occ.train$id.rel, data.occ.train$CO2, 
                         w = rep.int(1, nrow(data.occ.train)),
                         df = nrow(data.occ.train), i)
  ssr[j] = sum((data.occ.train$CO2 - spline$y)^2)
}

# Find index of smoothing parameter with smallest RSS
i.co2 = params[match(min(ssr), ssr)]

# Generate model using best smoothing parameter
spline.co2 = smooth.spline(data.occ.train$id.rel, data.occ.train$CO2, 
                            w = rep.int(1, nrow(data.occ.train)),
                            df = nrow(data.occ.train), i.co2)

# Plot time series and fitted spline
plot(data.occ.train$CO2, type="l", col="antiquewhite4")
lines(spline.co2, col = "cadetblue2")

# Fourier series to model seasonality
fourier.co2 = fourier(ts.co2, 3)
plot(data.occ.train$CO2, type="l", col="antiquewhite4")
plot(rowSums(fourier.co2)+spline.co2$y, type="l", col = 4)

# Use RSS to compare spline and spline + Fourier
sum((spline.co2$y-data.occ.train$CO2)^2)
sum((rowSums(fourier.co2)+spline.co2$y-data.occ.train$CO2)^2)
# Result: spline alone has smaller RSS. Use that to model CO2.

# Get residuals and plot them
spline.res.co2 = data.occ.train$CO2-spline.co2$y
plot(spline.res.co2, type="l")

# Model residuals using ARIMA
spline.res.co2.arima = auto.arima(spline.res.co2, seasonal = FALSE)
spline.res.co2.arima.fit = spline.res.co2.arima$fitted

# Model temperature using KDE and ARIMA errors
plot(data.occ.train$CO2, type="l", col="antiquewhite4")
plot(spline.co2$y + spline.res.co2.arima.fit, col=4)

# Use RSS to compare KDS with KDS + ARIMA errors
rss.spline.co2 = sum((data.occ.train$CO2-spline.co2$y)^2)
rss.spline.arima.co2 = sum((data.occ.train$CO2-(spline.co2$y + spline.res.co2.arima.fit))^2)
sort(c("w/o arma" = rss.spline.co2, "w arma" = rss.spline.arima.co2))



### 3. Model time series with Lowess smoothing only + Fourier terms and
###    Lowess smoothing + ARIMA errors

params = seq(from=0.000001,to=0.00001, by=0.0000001)

# For temperature:
# Lowess smoothing
lr = c()
j = 1
for (i in params) {
  low = lowess(data.occ.train$id.rel, data.occ.train$Temperature, i)
  lr[j] = sum((data.occ.train$Temperature-low$y)^2)
  j = j + 1
}

# Get smoothing parameter with smallest RSS
i.temp = params[match(min(lr), lr)]

# Generate model using best smoothing parameter
lowess.temp = lowess(data.occ.train$id.rel, data.occ.train$Temperature, i.temp)

# Plot time series and fitted lowess curve
plot(data.occ.train$Temperature, type="l", col="antiquewhite4", ylab = "Temperature",
     main="Lowess w/o ARMA errors")
lines(lowess.temp, lwd=2, col="cadetblue2")

# Fourier series to model seasonality
fourier.temp = fourier(ts.temp, 3)
plot(rowSums(fourier.temp), type="l", col = 4)

# Plot Lowess + Fourier
plot(data.occ.train$Temperature, type="l", col="antiquewhite4")
lines(lowess.temp$y+rowSums(fourier.temp), lwd=2, col="cadetblue2")
# Lowess + Fourier series not usable for temperature. Use Lowess alone.

# Get residuals and plot them
lowess.res.temp = data.occ.train$Temperature-lowess.temp$y
plot(lowess.res.temp, type="l")

# Model residuals using ARIMA
lowess.res.temp.arima = auto.arima(lowess.res.temp, seasonal = FALSE)
lowess.res.temp.arima.fit = lowess.res.temp.arima$fitted

# Model temperature using KDE and ARIMA errors
plot(data.occ.train$Temperature, type="l", col="antiquewhite4", ylab = "Temperature",
     main = "Lowess with ARMA errors")
lines(lowess.temp$y + lowess.res.temp.arima.fit, col=4)

# Use RSS to compare KDS with KDS + ARIMA errors
rss.lowess.temp = sum((data.occ.train$Temperature-lowess.temp$y)^2)
rss.lowess.arima.temp = sum((data.occ.train$Temperature-(lowess.temp$y + lowess.res.temp.arima.fit))^2)
sort(c("w/o arma" = rss.lowess.temp, "w arma" = rss.lowess.arima.temp))



# For humidity:
# Lowess smoothing
lr = c()
j = 1
for (i in params) {
  low = lowess(data.occ.train$id.rel, data.occ.train$Humidity, i)
  lr[j] = sum((data.occ.train$Humidity-low$y)^2)
  j = j + 1
}

# Get index of smoothing parameter with smallest RSS
i.hum = params[match(min(lr), lr)]

# Generate model using best smoothing parameter
lowess.hum = lowess(data.occ.train$id.rel, data.occ.train$Humidity, i.hum)

# Plot time series and fitted lowess curve
plot(data.occ.train$Humidity, type="l", col="antiquewhite4", ylab= "Humidity",
     main = "Lowess w/o ARMA errors")
lines(lowess.hum, lwd=2, col="cadetblue2")

# Fourier series to model seasonality
fourier.hum = fourier(ts.hum, 5)
plot(rowSums(fourier.hum), type="l", col = 4)

# Plot Lowess + Fourier
plot(data.occ.train$Humidity, type="l", col="antiquewhite4")
lines(lowess.hum$y+rowSums(fourier.hum), lwd=2, col="cadetblue2")
# Result: Fourier does not usable for humidity. Use Lowess alone.

# Get residuals and plot them
lowess.res.hum = data.occ.train$Humidity-lowess.hum$y
plot(lowess.res.hum, type="l")

# Model residuals using ARIMA
lowess.res.hum.arima = auto.arima(lowess.res.hum, seasonal = FALSE)
lowess.res.hum.arima.fit = lowess.res.hum.arima$fitted

# Model temperature using KDE and ARIMA errors
plot(data.occ.train$Humidity, type="l", col="antiquewhite4", ylab="Humidity", 
     main="Lowess with ARMA errors")
lines(lowess.hum$y + lowess.res.hum.arima.fit, col=4)

# Use RSS to compare KDS with KDS + ARIMA errors
rss.lowess.hum = sum((data.occ.train$Humidity-lowess.hum$y)^2)
rss.lowess.arima.hum = sum((data.occ.train$Humidity-(lowess.hum$y + lowess.res.hum.arima.fit))^2)
sort(c("w/o arma" = rss.lowess.hum, "w arma" = rss.lowess.arima.hum))



# For light:
# Lowess smoothing
lr = c()
j = 1
for (i in params) {
  low = lowess(data.occ.train$id.rel, data.occ.train$Light, i)
  lr[j] = sum((data.occ.train$Light-low$y)^2)
  j = j + 1
}

# Get index of smoothing parameter with smallest RSS
i.light = params[match(min(lr), lr)]

# Generate model using best smoothing parameter
lowess.light = lowess(data.occ.train$id.rel, data.occ.train$Light, i.light)

# Plot time series and fitted lowess curve
plot(data.occ.train$Light, type="l", col="antiquewhite4", ylab = "Light",
     main="Lowess w/o ARMA errors")
lines(lowess.light, lwd=2, col="cadetblue2")

# Fourier series to model seasonality
fourier.light = fourier(ts.light, 5)
plot(data.occ.train$Light, type="l", col="antiquewhite4")
lines(rowSums(fourier.light) + lowess.light$y, type="l", col=4)

# Use RSS to compare Lowess to Lowess + Fourier
sum((lowess.light$y-data.occ.train$Light)^2)
sum((rowSums(fourier.light)+lowess.light$y-data.occ.train$Light)^2)
# Result: Lowess alone has smaller RSS. Use that to model light.

# Get residuals and plot them
lowess.res.light = data.occ.train$Light-lowess.light$y
plot(lowess.res.light, type="l")

# Model residuals using ARIMA
lowess.res.light.arima = auto.arima(lowess.res.light, seasonal = FALSE)
lowess.res.light.arima.fit = lowess.res.light.arima$fitted

# Model temperature using lowess and ARIMA errors
plot(data.occ.train$Light, type="l", col="antiquewhite4", ylab = "Light",
     main="Lowess with ARMA errors")
lines(lowess.light$y + lowess.res.light.arima.fit, col=4)

# Use RSS to compare lowess with lowess + ARIMA errors
rss.lowess.light = sum((data.occ.train$Light-lowess.light$y)^2)
rss.lowess.arima.light = sum((data.occ.train$Light-(lowess.light$y + lowess.res.light.arima.fit))^2)
sort(c("w/o arma" = rss.lowess.light, "w arma" = rss.lowess.arima.light))




# For CO2:
# Lowess smoothing
lr = c()
j = 1
for (i in params) {
  low = lowess(data.occ.train$id.rel, data.occ.train$CO2, i)
  lr[j] = sum((data.occ.train$CO2-low$y)^2)
  j = j + 1
}

# Get index of smoothing parameter with smallest RSS
i.co2 = params[match(min(lr), lr)]

# Generate model using best smoothing parameter
lowess.co2 = lowess(data.occ.train$id.rel, data.occ.train$CO2, i.co2)

# Plot time series and fitted lowess curve
plot(data.occ.train$CO2, type="l", col="antiquewhite4", ylab="CO2", main="Lowess w/o ARMA errors")
lines(lowess.co2, lwd=2, col="cadetblue2")

# Fourier series to model seasonality
fourier.co2 = fourier(ts.co2, 5)
plot(data.occ.train$CO2, type="l", col="antiquewhite4")
lines(rowSums(fourier.co2)+lowess.co2$y, type="l", col = 4)

# Use RSS to compare Lowess to Lowess + Fourier 
sum((lowess.co2$y-data.occ.train$CO2)^2)
sum((rowSums(fourier.co2)+lowess.co2$y-data.occ.train$CO2)^2)
# Result: Lowess alone has smaller RSS. Model CO2 using that.

# Get residuals and plot them
lowess.res.co2 = data.occ.train$CO2-lowess.co2$y
plot(lowess.res.co2, type="l")

# Model residuals using ARIMA
lowess.res.co2.arima = auto.arima(lowess.res.co2, seasonal = FALSE)
lowess.res.co2.arima.fit = lowess.res.co2.arima$fitted

# Model CO2 using KDE and ARIMA errors
plot(data.occ.train$CO2, type="l", col="antiquewhite4", ylab ="CO2", main="Lowess with ARMA errors")
lines(lowess.co2$y + lowess.res.co2.arima.fit, col=4)

# Use RSS to compare lowess with lowess + ARIMA errors
rss.lowess.co2 = sum((data.occ.train$CO2-lowess.co2$y)^2)
rss.lowess.arima.co2 = sum((data.occ.train$CO2-(lowess.co2$y + lowess.res.co2.arima.fit))^2)
sort(c("w/o arma" = rss.lowess.co2, "w arma" = rss.lowess.arima.co2))



# Compare RSS between KDE/spline/lowess + ARIMA
# For temp:
sort(c("KDS" = rss.kds.arima.temp, "Spline" = rss.spline.arima.temp, "Lowess" = rss.lowess.arima.temp))

# For humidity:
sort(c("KDS" = rss.kds.arima.hum, "Spline" = rss.spline.arima.hum, "Lowess" = rss.lowess.arima.hum))

# For light:
sort(c("KDS" = rss.kds.arima.light, "Spline" = rss.spline.arima.light, "Lowess" = rss.lowess.arima.light))
     
# For CO2:
sort(c("KDS" = rss.kds.arima.co2, "Spline" = rss.spline.arima.co2, "Lowess" = rss.lowess.arima.co2))




### 4. Model time series with KDE, spline, Lowess for trend + Fourier terms for seasonality

# For temp:
# Kernel density estimation to find trend
kds.temp.trend = ksmooth(data.occ.train$id.rel, data.occ.train$Temperature, "normal", bandwidth=1500)
plot(data.occ.train$Temperature, type="l", col="antiquewhite4", ylab="Temperature", 
     main="Kernel trend")
lines(kds.temp.trend, col=4)

# Fourier terms for seasonality
temp.fourier = fourier(ts.temp, 3)
plot(data.occ.train$Temperature, type="l", col="antiquewhite4")
lines(kds.temp.trend$y+rowSums(temp.fourier))
# Result: Trend + Fourier does not resemble time series at all. 


# For humidity:
# Kernel density estimation to find trend
kds.hum.trend = ksmooth(data.occ.train$id.rel, data.occ.train$Humidity, "normal", bandwidth=1500)
plot(data.occ.train$Humidity, type="l", col="antiquewhite4", ylab="Humidity", main="Kernel Trend")
lines(kds.hum.trend, col=4)

# Fourier terms for seasonality
hum.fourier = fourier(ts.hum, 3)
plot(data.occ.train$Humidity, type="l", col="antiquewhite4")
lines(rowSums(hum.fourier)+ kds.hum.trend$y)
# Result: Trend + Fourier does not resemble time series at all. 



# For light:
# Kernel density estimation to find trend
kds.light.trend = ksmooth(data.occ.train$id.rel, data.occ.train$Light, "normal", bandwidth=1500)
plot(data.occ.train$Light, type="l", col="antiquewhite4", ylab="Light", main="Kernel trend")
lines(kds.light.trend, col=4) 

# Fourier terms for seasonality
light.fourier = fourier(ts.light, 5)
plot(data.occ.train$Light, type="l", col="antiquewhite4")
lines(rowSums(light.fourier)+kds.light.trend$y)
# Result: Trend + Fourier does not resemble time series at all. 



# For CO2: 
# Kernel density estimation to find trend
kds.co2.trend = ksmooth(data.occ.train$id.rel, data.occ.train$CO2, "normal", bandwidth=1600)
plot(data.occ.train$CO2, type="l", col="antiquewhite4", ylab="CO2", main="Kernel trend")
lines(kds.co2.trend, col=4)

# Fourier terms for seasonality
co2.fourier = fourier(ts.co2, 4)
plot(data.occ.train$CO2, type="l", col="antiquewhite4")
lines(rowSums(co2.fourier)+kds.co2.trend$y)
# Result: Trend + Fourier does not resemble time series at all.


# Spline smoothing to find trend
# For temp:
spline.temp.trend = smooth.spline(data.occ.train$id.rel, data.occ.train$Temperature, 
                            w = rep.int(1, nrow(data.occ.train)),
                            df = nrow(data.occ.train), 1.05)
plot(data.occ.train$Temperature, type="l", col="antiquewhite4", ylab="Temperature", main="Spline trend")
lines(spline.temp.trend, col = 4)

# For humidity:
spline.hum.trend = smooth.spline(data.occ.train$id.rel, data.occ.train$Humidity, 
                           w = rep.int(1, nrow(data.occ.train)),
                           df = nrow(data.occ.train), 1.1)
plot(data.occ.train$Humidity, type="l", col="antiquewhite4", ylab="Humidity", main="Spline trend")
lines(spline.hum.trend, col = 4)

# For light:
spline.light.trend = smooth.spline(data.occ.train$id.rel, data.occ.train$Light, 
                             w = rep.int(1, nrow(data.occ.train)),
                             df = nrow(data.occ.train), 1)
plot(data.occ.train$Light, type="l", col="antiquewhite4", ylab="Light", main="Spline trend")
lines(spline.light.trend, col = 4)

# For CO2:
spline.co2.trend = smooth.spline(data.occ.train$id.rel, data.occ.train$CO2, 
                           w = rep.int(1, nrow(data.occ.train)),
                           df = nrow(data.occ.train), 1)
plot(data.occ.train$CO2, type="l", col="antiquewhite4", ylab="CO2", main="Spline trend")
lines(spline.co2.trend, col = 4)



# Lowess smoothing to find trend
# For temperature:
lowess.temp.trend = lowess(data.occ.train$id.rel, data.occ.train$Temperature, 0.5)
plot(data.occ.train$Temperature, type="l", col="antiquewhite4", ylab="Temperature", main="Lowess trend")
lines(lowess.temp.trend, lwd=2, col="cadetblue2")

# For humidity:
lowess.hum.trend = lowess(data.occ.train$id.rel, data.occ.train$Humidity, 0.3)
plot(data.occ.train$Humidity, type="l", col="antiquewhite4", ylab="Humidity", main="Lowess trend")
lines(lowess.hum.trend, lwd=2, col="cadetblue2")

# For light:
lowess.light.trend = lowess(data.occ.train$id.rel, data.occ.train$Light, 0.25)
plot(data.occ.train$Light, type="l", col="antiquewhite4", ylab="Light", main="Lowess trend")
lines(lowess.light.trend, lwd=2, col="cadetblue2")

# For CO2:
lowess.co2.trend = lowess(data.occ.train$id.rel, data.occ.train$CO2, 0.2)
plot(data.occ.train$CO2, type="l", col="antiquewhite4", ylab="CO2", main="Lowess trend")
lines(lowess.co2.trend, lwd=2, col="cadetblue2")



### 5. Model time series with changepoint detection to estimate intervals of common mean and variance
###    + KDE + ARIMA errors 

# Fourier series were not usable for modelling temp. Use KDS + Arima because it has the lowest RSS 
# between KDS, spline and Lowess.

# For temperature:
# Changepoint method to detect changes in mean and var
chgpt.temp = cpt.meanvar(ts.temp)
plot(chgpt.temp)
chgpt.temp@cpts

# KDS on the changepoint intervals using same bandwidth as KDS in part 1
kds.chgpt.temp.1 = ksmooth(1:(chgpt.temp@cpts[1]-1), data.occ.train$Temperature[1:(chgpt.temp@cpts[1]-1)], 
                           kernel = "normal", bandwidth = 20)
kds.chgpt.temp.2 = ksmooth(chgpt.temp@cpts[1]:chgpt.temp@cpts[2], 
                           data.occ.train$Temperature[chgpt.temp@cpts[1]:chgpt.temp@cpts[2]], kernel = "normal",
                           bandwidth = 20)

# Get residuals
res.kds.chgpt.temp = data.occ.train$Temperature-c(kds.chgpt.temp.1$y, kds.chgpt.temp.2$y)

# Use ARIMA to model residuals
res.kds.chgpt.arima.temp = auto.arima(res.kds.chgpt.temp, seasonal = FALSE)

# Plot time series and fitted KDS + ARIMA values
plot(data.occ.train$Temperature, type="l", col="antiquewhite4")
lines(c(kds.chgpt.temp.1$y, kds.chgpt.temp.2$y)+res.kds.chgpt.arima.temp$fitted, col=4)

# Get RSS and compare to RSS for KDS + ARIMA without changepoint
rss.kds.chgpt.temp = sum((data.occ.train$Temperature-
                       (c(kds.chgpt.temp.1$y, kds.chgpt.temp.2$y)+res.kds.chgpt.arima.temp$fitted))^2)
sort(c("with chgpt" = rss.kds.chgpt.temp, "w/o chgpt" = rss.kds.arima.temp))


# For humidity:
# Changepoint method to detect changes in mean and var
chgpt.hum = cpt.meanvar(ts.hum)
plot(chgpt.hum)
chgpt.hum@cpts

# KDS on the changepoint intervals using same bandwidth as KDS in part 1
kds.chgpt.hum.1 = ksmooth(1:(chgpt.hum@cpts[1]-1), data.occ.train$Humidity[1:(chgpt.hum@cpts[1]-1)], 
                           kernel = "normal", bandwidth = 20)
kds.chgpt.hum.2 = ksmooth(chgpt.hum@cpts[1]:chgpt.hum@cpts[2], 
                           data.occ.train$Humidity[chgpt.hum@cpts[1]:chgpt.hum@cpts[2]], kernel = "normal",
                           bandwidth = 20)

# Get residuals
res.kds.chgpt.hum = data.occ.train$Humidity-c(kds.chgpt.hum.1$y, kds.chgpt.hum.2$y)

# Use ARIMA to model residuals
res.kds.chgpt.arima.hum = auto.arima(res.kds.chgpt.hum, seasonal = FALSE)

# Plot time series and fitted KDS + ARIMA values
plot(data.occ.train$Humidity, type="l", col="antiquewhite4")
lines(c(kds.chgpt.hum.1$y, kds.chgpt.hum.2$y)+res.kds.chgpt.arima.hum$fitted, col=4)

# Get RSS and compare to RSS for KDS + ARIMA without changepoint
rss.kds.chgpt.hum = sum((data.occ.train$Humidity-
                            (c(kds.chgpt.hum.1$y, kds.chgpt.hum.2$y)+res.kds.chgpt.arima.hum$fitted))^2)
sort(c("with chgpt" = rss.kds.chgpt.hum, "w/o chgpt" = rss.kds.arima.hum))




# For light:
# Changepoint method to detect changes in mean and var
chgpt.light = cpt.meanvar(ts.light)
plot(chgpt.light)
chgpt.light@cpts


# KDS on the changepoint intervals using same bandwidth as KDS in part 1
kds.chgpt.light.1 = ksmooth(1:(chgpt.light@cpts[1]-1), data.occ.train$Light[1:(chgpt.light@cpts[1]-1)], 
                          kernel = "normal", bandwidth = 20)
kds.chgpt.light.2 = ksmooth(chgpt.light@cpts[1]:chgpt.light@cpts[2], 
                          data.occ.train$Light[chgpt.light@cpts[1]:chgpt.light@cpts[2]], kernel = "normal",
                          bandwidth = 20)

# Get residuals
res.kds.chgpt.light = data.occ.train$Light-c(kds.chgpt.light.1$y, kds.chgpt.light.2$y)

# Use ARIMA to model residuals
res.kds.chgpt.arima.light = auto.arima(res.kds.chgpt.light, seasonal = FALSE)

# Plot time series and fitted KDS + ARIMA values
plot(data.occ.train$Light, type="l", col="antiquewhite4")
lines(c(kds.chgpt.light.1$y, kds.chgpt.light.2$y)+res.kds.chgpt.arima.light$fitted, col=4)

# Get RSS and compare to RSS for KDS + ARIMA without changepoint
rss.kds.chgpt.light = sum((data.occ.train$Light-
                           (c(kds.chgpt.light.1$y, kds.chgpt.light.2$y)+res.kds.chgpt.arima.light$fitted))^2)
sort(c("with chgpt" = rss.kds.chgpt.light, "w/o chgpt" = rss.kds.arima.light))



# For CO2:
# Changepoint method to detect changes in mean and var
chgpt.co2 = cpt.meanvar(ts.co2)
plot(chgpt.co2)
chgpt.co2@cpts


# KDS on the changepoint intervals using same bandwidth as KDS in part 1
kds.chgpt.co2.1 = ksmooth(1:(chgpt.co2@cpts[1]-1), data.occ.train$CO2[1:(chgpt.co2@cpts[1]-1)], 
                          kernel = "normal", bandwidth = 20)
kds.chgpt.co2.2 = ksmooth(chgpt.co2@cpts[1]:chgpt.co2@cpts[2], 
                          data.occ.train$CO2[chgpt.co2@cpts[1]:chgpt.co2@cpts[2]], kernel = "normal",
                          bandwidth = 20)

# Get residuals
res.kds.chgpt.co2 = data.occ.train$CO2-c(kds.chgpt.co2.1$y, kds.chgpt.co2.2$y)

# Use ARIMA to model residuals
res.kds.chgpt.arima.co2 = auto.arima(res.kds.chgpt.co2, seasonal = FALSE)

# Plot time series and fitted KDS + ARIMA values
plot(data.occ.train$CO2, type="l", col="antiquewhite4")
lines(c(kds.chgpt.co2.1$y, kds.chgpt.co2.2$y)+res.kds.chgpt.arima.co2$fitted, col=4)

# Get RSS and compare to RSS for KDS + ARIMA without changepoint
rss.kds.chgpt.co2 = sum((data.occ.train$CO2-
                           (c(kds.chgpt.co2.1$y, kds.chgpt.co2.2$y)+res.kds.chgpt.arima.co2$fitted))^2)
sort(c("with chgpt" = rss.kds.chgpt.co2, "w/o chgpt" = rss.kds.arima.co2))




### 6. Model time series using Fourier terms with ARMA errors
# For temperature:
# Test multiple numbers of sin and cos pairs (K parameter) to find best fit
fits.temp = c()
for (i in 1:10) {
  fit = auto.arima(ts.temp, xreg = fourier(ts.temp, K = i), seasonal = FALSE, lambda = NULL)
  fits.temp[i] = fit$aicc
}

# Find index of model with lowest AICc
k.temp = match(min(fits.temp), fits.temp)

# Generate model using that index
f.a.temp = auto.arima(ts.temp, xreg = fourier(ts.temp, K = k.temp), seasonal = FALSE, lambda = NULL)

# Plot time series and model
plot(data.occ.train$Temperature, type="l", col = "antiquewhite4")
plot(f.a.temp$fitted, lwd=2, col = "cadetblue2", ylab="Temperature", main="Fourier with ARMA errors")


# For humidity:
# Test multiple numbers of sin and cos pairs (K parameter) to find best fit
fits.hum = c()
for (i in 1:10) {
  fit = auto.arima(ts.hum, xreg = fourier(ts.hum, K = i), seasonal = FALSE, lambda = NULL)
  fits.hum[i] = fit$aicc
}

# Find index of model with lowerst AICc
k.hum = match(min(fits.hum), fits.hum)

# Generate model using that index
f.a.hum = auto.arima(ts.hum, xreg = fourier(ts.hum, K = k.hum), seasonal = FALSE, lambda = NULL)

# Plot time series and model
plot(data.occ.train$Humidity, type="l", col = "antiquewhite4")
plot(f.a.hum$fitted, lwd=2, col = "cadetblue2", ylab="Humidity", main="Fourier with ARMA errors")


# For light:
# Test multiple numbers of sin and cos pairs (K parameter) to find best fit
fits.light = c()
for (i in 1:10) {
  fit = auto.arima(ts.light, xreg = fourier(ts.light, K = i), seasonal = FALSE, lambda = NULL)
  fits.light[i] = fit$aicc
}

# Find index of model with lowerst AICc
k.light = match(min(fits.light), fits.light)

# Generate model using that index
f.a.light = auto.arima(ts.light, xreg = fourier(ts.light, K = k.light), seasonal = FALSE, lambda = NULL)

# Plot time series and model
plot(data.occ.train$Light, type="l", col = "antiquewhite4")
plot(f.a.light$fitted, lwd=2, col = "cadetblue2", ylab="Light", main="Fourier with ARMA errors")


# For CO2:
# Test multiple numbers of sin and cos pairs (K parameter) to find best fit
fits.co2 = c()
for (i in 1:10) {
  fit = auto.arima(ts.co2, xreg = fourier(ts.co2, K = i), seasonal = FALSE, lambda = NULL)
  fits.co2[i] = fit$aicc
}

# Find index of model with lowest AICc
k.co2 = match(min(fits.co2), fits.co2)

# Generate model using that index
f.a.co2 = auto.arima(ts.co2, xreg = fourier(ts.co2, K = k.co2), seasonal = FALSE, lambda = NULL)

# Plot time series and model
plot(data.occ.train$CO2, type="l", col = "antiquewhite4")
plot(f.a.co2$fitted, lwd=2, col = "cadetblue2", ylab="CO2", main="Fourier with ARMA errors")



### Classification models for predicting the response 

# Make datasets from the fitted time series
data.1 = as.data.frame(cbind("Occupancy" = as.factor(data.occ.train$Occupancy), 
               "Temperature" = kds.temp$y + kds.res.temp.arima.fit,
               "Humidity" = kds.hum$y + kds.res.hum.arima.fit, 
               "Light" = kds.light$y + kds.res.light.arima.fit,
               "CO2" = kds.co2$y + kds.res.co2.arima.fit))
data.1$Occupancy = as.factor(data.occ.train$Occupancy)

data.2 = as.data.frame(cbind("Occupancy" = as.factor(data.occ.train$Occupancy), 
               "Temperature" = spline.temp$y + spline.res.temp.arima.fit,
               "Humidity" = spline.hum$y + spline.res.hum.arima.fit, 
               "Light" = spline.light$y + spline.res.light.arima.fit,
               "CO2" = spline.co2$y + spline.res.co2.arima.fit))
data.2$Occupancy = as.factor(data.occ.train$Occupancy)


data.3 = as.data.frame(cbind("Occupancy" = as.factor(data.occ.train$Occupancy), 
               "Temperature" = lowess.temp$y + lowess.res.temp.arima.fit,
               "Humidity" = lowess.hum$y + lowess.res.hum.arima.fit, 
               "Light" = lowess.light$y + lowess.res.light.arima.fit,
               "CO2" = lowess.co2$y + lowess.res.co2.arima.fit))
data.3$Occupancy = as.factor(data.occ.train$Occupancy)


data.4 = as.data.frame(cbind("Occupancy" = as.factor(data.occ.train$Occupancy), 
               "Temperature" = c(kds.chgpt.temp.1$y, kds.chgpt.temp.2$y)+res.kds.chgpt.arima.temp$fitted,
               "Humidity" = c(kds.chgpt.hum.1$y, kds.chgpt.hum.2$y)+res.kds.chgpt.arima.hum$fitted,
               "Light" = c(kds.chgpt.light.1$y, kds.chgpt.light.2$y)+res.kds.chgpt.arima.light$fitted,
               "CO2" = c(kds.chgpt.co2.1$y, kds.chgpt.co2.2$y)+res.kds.chgpt.arima.co2$fitted))
data.4$Occupancy = as.factor(data.occ.train$Occupancy)


data.5 = as.data.frame(cbind("Occupancy" = as.factor(data.occ.train$Occupancy), 
               "Temperature" = f.a.temp$fitted,
               "Humidity" = f.a.hum$fitted,
               "Light" = f.a.light$fitted,
               "CO2" = f.a.co2$fitted))
data.5$Occupancy = as.factor(data.occ.train$Occupancy)


data.6 = data.occ.train[,c(2, 3, 4, 5, 7)]
data.6$Occupancy = as.factor(data.6$Occupancy)



### 1. Use decision trees to predict occupancy
set.seed(6636)

# Generate decision trees
rpart.1 = rpart(Occupancy ~ ., data = data.1, method = "class", minsplit = 2, minbucket = 1, cp = -1)
rpart.2 = rpart(Occupancy ~ ., data = data.2, method = "class", minsplit = 2, minbucket = 1, cp = -1)
rpart.3 = rpart(Occupancy ~ ., data = data.3, method = "class", minsplit = 2, minbucket = 1, cp = -1)
rpart.4 = rpart(Occupancy ~ ., data = data.4, method = "class", minsplit = 2, minbucket = 1, cp = -1)
rpart.5 = rpart(Occupancy ~ ., data = data.5, method = "class", minsplit = 2, minbucket = 1, cp = -1)
rpart.6 = rpart(Occupancy ~ ., data = data.6, method = "class", minsplit = 2, minbucket = 1, cp = -1)

# View trees
rpart.1
rpart.2
rpart.3
rpart.4
rpart.5
rpart.6

# Check variable importance
rpart.1$variable.importance
rpart.2$variable.importance
rpart.3$variable.importance
rpart.4$variable.importance
rpart.5$variable.importance
rpart.6$variable.importance

# Get cross-validation results to find size of tree optimal for classification
printcp(rpart.1)
printcp(rpart.2)
printcp(rpart.3)
printcp(rpart.4)
printcp(rpart.5)
printcp(rpart.6)

# Prune trees to optimal size
rpart.1.p = prune(rpart.1, cp=0.00058)
rpart.2.p = prune(rpart.2, cp=0.00029)
rpart.3.p = prune(rpart.3, cp=0.00174)
rpart.4.p = prune(rpart.4, cp=0.0016)
rpart.5.p = prune(rpart.5, cp=0.00521)
rpart.6.p = prune(rpart.6, cp=0.00116)

# View variable importance 
rpart.1.p$variable.importance
rpart.2.p$variable.importance
rpart.3.p$variable.importance
rpart.4.p$variable.importance
rpart.5.p$variable.importance
rpart.6.p$variable.importance


### 2. Use random forests (RF) to predict occupancy
rf.1 = randomForest(Occupancy ~ ., data = data.1, importance = TRUE)
rf.2 = randomForest(Occupancy ~ ., data = data.2, importance = TRUE)
rf.3 = randomForest(Occupancy ~ ., data = data.3, importance = TRUE)
rf.4 = randomForest(Occupancy ~ ., data = data.4, importance = TRUE)
rf.5 = randomForest(Occupancy ~ ., data = data.5, importance = TRUE)
rf.6 = randomForest(Occupancy ~ ., data = data.6, importance = TRUE)

# View RFs
rf.1
rf.2
rf.3
rf.4
rf.5
rf.6

# Check variable importance in each RF
rf.1$importance
rf.2$importance
rf.3$importance
rf.4$importance
rf.5$importance
rf.6$importance

# Check which variables were used in each RF
varUsed(rf.1)
varUsed(rf.2)
varUsed(rf.3)
varUsed(rf.4)
varUsed(rf.5)
varUsed(rf.6)


### 3. Use support vector machines (SVM) to predict occupancy

# Convert 0 to -1 in response variable (required for SVM algorithm)
occ.svm = as.factor(ifelse(data.occ.train$Occupancy == 0, -1, 1))
data.1.svm = data.1
data.1.svm$Occupancy = occ.svm
data.2.svm = data.2
data.2.svm$Occupancy = occ.svm
data.3.svm = data.3
data.3.svm$Occupancy = occ.svm
data.4.svm = data.4
data.4.svm$Occupancy = occ.svm
data.5.svm = data.5
data.5.svm$Occupancy = occ.svm
data.6.svm = data.occ.train[,c(2, 3, 4, 5, 7)]
data.6.svm$Occupancy = occ.svm

# Apply SVM to data
svm.1.a = svm(Occupancy ~ ., data = data.1.svm, kernel = "linear")
svm.1.b = svm(Occupancy ~ ., data = data.1.svm)
svm.1.c = svm(Occupancy ~ ., data = data.1.svm, kernel = "polynomial")

svm.2.a = svm(Occupancy ~ ., data = data.2.svm, kernel = "linear")
svm.2.b = svm(Occupancy ~ ., data = data.2.svm)
svm.2.c = svm(Occupancy ~ ., data = data.2.svm, kernel = "polynomial")

svm.3.a = svm(Occupancy ~ ., data = data.3.svm, kernel = "linear")
svm.3.b = svm(Occupancy ~ ., data = data.3.svm)
svm.3.c = svm(Occupancy ~ ., data = data.3.svm, kernel = "polynomial")

svm.4.a = svm(Occupancy ~ ., data = data.4.svm, kernel = "linear")
svm.4.b = svm(Occupancy ~ ., data = data.4.svm)
svm.4.c = svm(Occupancy ~ ., data = data.4.svm, kernel = "polynomial")

svm.5.a = svm(Occupancy ~ ., data = data.5.svm, kernel = "linear")
svm.5.b = svm(Occupancy ~ ., data = data.5.svm)
svm.5.c = svm(Occupancy ~ ., data = data.5.svm, kernel = "polynomial")

svm.6.a = svm(Occupancy ~ ., data = data.6.svm, kernel = "linear")
svm.6.b = svm(Occupancy ~ ., data = data.6.svm)
svm.6.c = svm(Occupancy ~ ., data = data.6.svm, kernel = "polynomial")

# Show models
svm.1.a 
svm.1.b 
svm.1.c 

svm.2.a 
svm.2.b 
svm.2.c 

svm.3.a 
svm.3.b 
svm.3.c 

svm.4.a 
svm.4.b 
svm.4.c 

svm.5.a 
svm.5.b 
svm.5.c 

svm.6.a 
svm.6.b 
svm.6.c

# Sort models by number of support vectors
sort(c("a" = svm.1.a$tot.nSV, "b" = svm.1.b$tot.nSV, "c" = svm.1.c$tot.nSV))
sort(c("a" = svm.2.a$tot.nSV, "b" = svm.2.b$tot.nSV, "c" = svm.2.c$tot.nSV))
sort(c("a" = svm.3.a$tot.nSV, "b" = svm.3.b$tot.nSV, "c" = svm.3.c$tot.nSV))
sort(c("a" = svm.4.a$tot.nSV, "b" = svm.4.b$tot.nSV, "c" = svm.4.c$tot.nSV))
sort(c("a" = svm.5.a$tot.nSV, "b" = svm.5.b$tot.nSV, "c" = svm.5.c$tot.nSV))
sort(c("a" = svm.6.a$tot.nSV, "b" = svm.6.b$tot.nSV, "c" = svm.6.c$tot.nSV))



### 4. Use gradient boosting machines (GBM) to predict occupancy
# GBMs cause RStudio to crash. The code is here but results are impossible to obtain using what is 
# computationally available.

# # Generate GBMs
# gbm.1 = gbm(Occupancy ~., data = data.1, n.trees = 1000,  cv.folds = 5)
# gbm.2 = gbm(Occupancy ~., data = data.2, n.trees = 1000,  cv.folds = 5)
# gbm.3 = gbm(Occupancy ~., data = data.3, n.trees = 1000,  cv.folds = 5)
# gbm.4 = gbm(Occupancy ~., data = data.4, n.trees = 1000,  cv.folds = 5)
# gbm.5 = gbm(Occupancy ~., data = data.5, n.trees = 1000,  cv.folds = 5)
# gbm.6 = gbm(Occupancy ~., data = data.6, n.trees = 1000,  cv.folds = 5)
# 
# # check the above cv error rates, then change the interaction.depth to 3
# 
# # Print results of each tree
# print(gbm.1)
# print(gbm.2)
# print(gbm.3)
# print(gbm.4)
# print(gbm.5)
# print(gbm.6)
# 
# # Get MSE and RMSE from models
# sort(c("RMSE gbm 1" = sqrt(min(gbm.1$cv.error)), "RMSE gbm 2" = sqrt(min(gbm.2$cv.error)), 
#   "RMSE gbm 3" = sqrt(min(gbm.3$cv.error)), "RMSE gbm 4" = sqrt(min(gbm.4$cv.error)), 
#   "RMSE gbm 5" = sqrt(min(gbm.5$cv.error)), "RMSE gbm 6" = sqrt(min(gbm.6$cv.error))))
# 
# # Plot squared error loss for each tree
# gbm.perf(gbm.1, method = "cv")
# gbm.perf(gbm.2, method = "cv")
# gbm.perf(gbm.3, method = "cv")
# gbm.perf(gbm.4, method = "cv")
# gbm.perf(gbm.5, method = "cv")
# gbm.perf(gbm.6, method = "cv")


### 5. Use artificial neural net (ANN) to predict occupancy
# To decrease computational demand, run ANN only on original data
# Note: min-max scaling has already been applied to the data

# Generate ANNs
# Use different parameters to find lowest training errors and see which ones converge
ann.1 = neuralnet(Occupancy ~ Temperature + Humidity + Light + CO2, data = data.occ.train[,c(2, 3, 4, 5, 7)], hidden = 3, 
                  threshold = 0.002, linear.output = FALSE, likelihood = TRUE)
ann.2 = neuralnet(Occupancy ~ Temperature + Humidity + Light + CO2, data = data.occ.train[,c(2, 3, 4, 5, 7)], hidden = c(2,1), 
                  threshold = 0.003, linear.output = FALSE, likelihood = TRUE)
ann.3 = neuralnet(Occupancy ~ Temperature + Humidity + Light + CO2, data = data.occ.train[,c(2, 3, 4, 5, 7)], hidden = c(3,2), 
                  threshold = 0.01, linear.output = FALSE, likelihood = TRUE)
ann.4 = neuralnet(Occupancy ~ Temperature + Humidity + Light + CO2, data = data.occ.train[,c(2, 3, 4, 5, 7)], hidden = c(3,1), 
                  threshold = 0.01, linear.output = FALSE, likelihood = TRUE)
ann.5 = neuralnet(Occupancy ~ Temperature + Humidity + Light + CO2, data = data.occ.train[,c(2, 3, 4, 5, 7)], hidden = 2, 
                  threshold = 0.002, linear.output = FALSE, likelihood = TRUE)
ann.6 = neuralnet(Occupancy ~ Temperature + Humidity + Light + CO2, data = data.occ.train[,c(2, 3, 4, 5, 7)], hidden = 5, 
                  threshold = 0.002, linear.output = FALSE, likelihood = TRUE)
ann.7 = neuralnet(Occupancy ~ Temperature + Humidity + Light + CO2, data = data.occ.train[,c(2, 3, 4, 5, 7)], hidden = c(5, 2), 
                  threshold = 0.01, linear.output = FALSE, likelihood = TRUE)

# View training set classification errors 
sort(c("ANN 1" = ann.1$result.matrix[1,1], "ANN 2" = ann.2$result.matrix[1,1], 
       "ANN 3" = ann.3$result.matrix[1,1], "ANN 4" = ann.4$result.matrix[1,1], 
       "ANN 5" = ann.5$result.matrix[1,1], "ANN 6" = ann.6$result.matrix[1,1],
       "ANN 7" = ann.7$result.matrix[1,1]))

# Plot ANNs
plot(ann.1)
plot(ann.2)
plot(ann.3)
plot(ann.4)
plot(ann.5)
plot(ann.6)
plot(ann.7)


### 6. Use logistic regression to model occupancy
log.1 = glm(Occupancy ~ ., family = binomial, data = data.1)
log.2 = glm(Occupancy ~ ., family = binomial, data = data.2)
log.3 = glm(Occupancy ~ ., family = binomial, data = data.3)
log.4 = glm(Occupancy ~ ., family = binomial, data = data.4)
log.5 = glm(Occupancy ~ ., family = binomial, data = data.5)
log.6 = glm(Occupancy ~ ., family = binomial, data = data.6)
# Note: 3 and 5 did not converge.

# Make training confusion matrices
t.l.1 = table(data.1$Occupancy, log.1$y)
t.l.2 = table(data.2$Occupancy, log.2$y)
t.l.4 = table(data.4$Occupancy, log.4$y)
t.l.6 = table(data.6$Occupancy, log.6$y)

# Get training classification accuracy
sort(c("log.1" = (t.l.1[1]+t.l.1[4])/sum(t.l.1), "log.2" = (t.l.2[1]+t.l.2[4])/sum(t.l.2), 
       "log.4" = (t.l.4[1]+t.l.4[4])/sum(t.l.4), "log.6" = (t.l.6[1]+t.l.6[4])/sum(t.l.6)))

# Show significant predictors in each converged model
summary(log.1)
summary(log.2)
summary(log.4)
summary(log.6)


### Apply min-max scaling to test data
data.test.1 = data.occ.test[,c(2, 3, 4, 5, 7)]
data.test.2 = data.occ.test.2[,c(2, 3, 4, 5, 7)]
data.test.1$Temperature = (data.test.1$Temperature-min(data.test.1$Temperature))/
  (max(data.test.1$Temperature)-min(data.test.1$Temperature))
data.test.1$Humidity = (data.test.1$Humidity-min(data.test.1$Humidity))/
  (max(data.test.1$Humidity)-min(data.test.1$Humidity))
data.test.1$Light = (data.test.1$Light-min(data.test.1$Light))/
  (max(data.test.1$Light)-min(data.test.1$Light))
data.test.1$CO2 = (data.test.1$CO2-min(data.test.1$CO2))/
  (max(data.test.1$CO2)-min(data.test.1$CO2))
data.test.2$Temperature = (data.test.2$Temperature-min(data.test.2$Temperature))/
  (max(data.test.2$Temperature)-min(data.test.2$Temperature))
data.test.2$Humidity = (data.test.2$Humidity-min(data.test.2$Humidity))/
  (max(data.test.2$Humidity)-min(data.test.2$Humidity))
data.test.2$Light = (data.test.2$Light-min(data.test.2$Light))/
  (max(data.test.2$Light)-min(data.test.2$Light))
data.test.2$CO2 = (data.test.2$CO2-min(data.test.2$CO2))/
  (max(data.test.2$CO2)-min(data.test.2$CO2))



### Make predictions on test sets and generate confusion matrices 

# Decision trees
# Apply to test set 1
cm.rpart.1.p.1 = table(data.test.1$Occupancy, predict(rpart.1.p, newdata = data.test.1, type = "class"))
cm.rpart.2.p.1 = table(data.test.1$Occupancy, predict(rpart.2.p, newdata = data.test.1, type = "class"))
cm.rpart.3.p.1 = table(data.test.1$Occupancy, predict(rpart.3.p, newdata = data.test.1, type = "class"))
cm.rpart.4.p.1 = table(data.test.1$Occupancy, predict(rpart.4.p, newdata = data.test.1, type = "class"))
cm.rpart.5.p.1 = table(data.test.1$Occupancy, predict(rpart.5.p, newdata = data.test.1, type = "class"))
cm.rpart.6.p.1 = table(data.test.1$Occupancy, predict(rpart.6.p, newdata = data.test.1, type = "class"))

# Add title to each CM
names(dimnames(cm.rpart.1.p.1)) = c("", "DT1p: TS1")
names(dimnames(cm.rpart.2.p.1)) = c("", "DT2p: TS1")
names(dimnames(cm.rpart.3.p.1)) = c("", "DT3p: TS1")
names(dimnames(cm.rpart.4.p.1)) = c("", "DT4p: TS1")
names(dimnames(cm.rpart.5.p.1)) = c("", "DT5p: TS1")
names(dimnames(cm.rpart.6.p.1)) = c("", "DT6p: TS1")

# Apply to test set 2
cm.rpart.1.p.2 = table(data.test.2$Occupancy, predict(rpart.1.p, newdata = data.test.2, type = "class"))
cm.rpart.2.p.2 = table(data.test.2$Occupancy, predict(rpart.2.p, newdata = data.test.2, type = "class"))
cm.rpart.3.p.2 = table(data.test.2$Occupancy, predict(rpart.3.p, newdata = data.test.2, type = "class"))
cm.rpart.4.p.2 = table(data.test.2$Occupancy, predict(rpart.4.p, newdata = data.test.2, type = "class"))
cm.rpart.5.p.2 = table(data.test.2$Occupancy, predict(rpart.5.p, newdata = data.test.2, type = "class"))
cm.rpart.6.p.2 = table(data.test.2$Occupancy, predict(rpart.6.p, newdata = data.test.2, type = "class"))

# Add title to each CM
names(dimnames(cm.rpart.1.p.2)) = c("", "DT1p: TS2")
names(dimnames(cm.rpart.2.p.2)) = c("", "DT2p: TS2")
names(dimnames(cm.rpart.3.p.2)) = c("", "DT3p: TS2")
names(dimnames(cm.rpart.4.p.2)) = c("", "DT4p: TS2")
names(dimnames(cm.rpart.5.p.2)) = c("", "DT5p: TS2")
names(dimnames(cm.rpart.6.p.2)) = c("", "DT6p: TS2")

# Show CMs
cbind(cm.rpart.1.p.1, cm.rpart.2.p.1, cm.rpart.3.p.1, cm.rpart.4.p.1, cm.rpart.5.p.1, cm.rpart.6.p.1)
cbind(cm.rpart.1.p.2, cm.rpart.2.p.2, cm.rpart.3.p.2, cm.rpart.4.p.2, cm.rpart.5.p.2, cm.rpart.6.p.2)


# RFs
# Apply to test set 1
cm.rf.1.1 = table(data.test.1$Occupancy, predict(rf.1, newdata = data.test.1, type = "class"))
cm.rf.2.1 = table(data.test.1$Occupancy, predict(rf.2, newdata = data.test.1, type = "class"))
cm.rf.3.1 = table(data.test.1$Occupancy, predict(rf.3, newdata = data.test.1, type = "class"))
cm.rf.4.1 = table(data.test.1$Occupancy, predict(rf.4, newdata = data.test.1, type = "class"))
cm.rf.5.1 = table(data.test.1$Occupancy, predict(rf.5, newdata = data.test.1, type = "class"))
cm.rf.6.1 = table(data.test.1$Occupancy, predict(rf.6, newdata = data.test.1, type = "class"))

# Add titles
names(dimnames(cm.rf.1.1)) = c("", "RF1: TS1")
names(dimnames(cm.rf.2.1)) = c("", "RF2: TS1")
names(dimnames(cm.rf.3.1)) = c("", "RF3: TS1")
names(dimnames(cm.rf.4.1)) = c("", "RF4: TS1")
names(dimnames(cm.rf.5.1)) = c("", "RF5: TS1")
names(dimnames(cm.rf.6.1)) = c("", "RF6: TS1")

# Apply to test set 2
cm.rf.1.2 = table(data.test.2$Occupancy, predict(rf.1, newdata = data.test.2, type = "class"))
cm.rf.2.2 = table(data.test.2$Occupancy, predict(rf.2, newdata = data.test.2, type = "class"))
cm.rf.3.2 = table(data.test.2$Occupancy, predict(rf.3, newdata = data.test.2, type = "class"))
cm.rf.4.2 = table(data.test.2$Occupancy, predict(rf.4, newdata = data.test.2, type = "class"))
cm.rf.5.2 = table(data.test.2$Occupancy, predict(rf.5, newdata = data.test.2, type = "class"))
cm.rf.6.2 = table(data.test.2$Occupancy, predict(rf.6, newdata = data.test.2, type = "class"))

# Add titles
names(dimnames(cm.rf.1.2)) = c("", "RF1: TS2")
names(dimnames(cm.rf.2.2)) = c("", "RF2: TS2")
names(dimnames(cm.rf.3.2)) = c("", "RF3: TS2")
names(dimnames(cm.rf.4.2)) = c("", "RF4: TS2")
names(dimnames(cm.rf.5.2)) = c("", "RF5: TS2")
names(dimnames(cm.rf.6.2)) = c("", "RF6: TS2")

# Show CMs
cbind(cm.rf.1.1, cm.rf.2.1, cm.rf.3.1, cm.rf.4.1, cm.rf.5.1, cm.rf.6.1)
cbind(cm.rf.1.2, cm.rf.2.2, cm.rf.3.2, cm.rf.4.2, cm.rf.5.2, cm.rf.6.2)


# SVMs
# Apply to test set 1
cm.svm.1.a.1 = table(data.test.1$Occupancy, predict(svm.1.a, newdata = data.test.1, type = "class"))
cm.svm.2.a.1 = table(data.test.1$Occupancy, predict(svm.2.a, newdata = data.test.1, type = "class"))
cm.svm.3.a.1 = table(data.test.1$Occupancy, predict(svm.3.a, newdata = data.test.1, type = "class"))
cm.svm.4.a.1 = table(data.test.1$Occupancy, predict(svm.4.a, newdata = data.test.1, type = "class"))
cm.svm.5.a.1 = table(data.test.1$Occupancy, predict(svm.5.a, newdata = data.test.1, type = "class"))
cm.svm.6.a.1 = table(data.test.1$Occupancy, predict(svm.6.a, newdata = data.test.1, type = "class"))
cm.svm.1.b.1 = table(data.test.1$Occupancy, predict(svm.1.b, newdata = data.test.1, type = "class"))
cm.svm.2.b.1 = table(data.test.1$Occupancy, predict(svm.2.b, newdata = data.test.1, type = "class"))
cm.svm.3.b.1 = table(data.test.1$Occupancy, predict(svm.3.b, newdata = data.test.1, type = "class"))
cm.svm.4.b.1 = table(data.test.1$Occupancy, predict(svm.4.b, newdata = data.test.1, type = "class"))
cm.svm.5.b.1 = table(data.test.1$Occupancy, predict(svm.5.b, newdata = data.test.1, type = "class"))
cm.svm.6.b.1 = table(data.test.1$Occupancy, predict(svm.6.b, newdata = data.test.1, type = "class"))
cm.svm.1.c.1 = table(data.test.1$Occupancy, predict(svm.1.c, newdata = data.test.1, type = "class"))
cm.svm.2.c.1 = table(data.test.1$Occupancy, predict(svm.2.c, newdata = data.test.1, type = "class"))
cm.svm.3.c.1 = table(data.test.1$Occupancy, predict(svm.3.c, newdata = data.test.1, type = "class"))
cm.svm.4.c.1 = table(data.test.1$Occupancy, predict(svm.4.c, newdata = data.test.1, type = "class"))
cm.svm.5.c.1 = table(data.test.1$Occupancy, predict(svm.5.c, newdata = data.test.1, type = "class"))
cm.svm.6.c.1 = table(data.test.1$Occupancy, predict(svm.6.c, newdata = data.test.1, type = "class"))

# Add titles
names(dimnames(cm.svm.1.a.1)) = c("", "SVM1a: TS1")
names(dimnames(cm.svm.2.a.1)) = c("", "SVM2a: TS1")
names(dimnames(cm.svm.3.a.1)) = c("", "SVM3a: TS1")
names(dimnames(cm.svm.4.a.1)) = c("", "SVM4a: TS1")
names(dimnames(cm.svm.5.a.1)) = c("", "SVM5a: TS1")
names(dimnames(cm.svm.6.a.1)) = c("", "SVM6a: TS1")
names(dimnames(cm.svm.1.b.1)) = c("", "SVM1b: TS1")
names(dimnames(cm.svm.2.b.1)) = c("", "SVM2b: TS1")
names(dimnames(cm.svm.3.b.1)) = c("", "SVM3b: TS1")
names(dimnames(cm.svm.4.b.1)) = c("", "SVM4b: TS1")
names(dimnames(cm.svm.5.b.1)) = c("", "SVM5b: TS1")
names(dimnames(cm.svm.6.b.1)) = c("", "SVM6b: TS1")
names(dimnames(cm.svm.1.c.1)) = c("", "SVM1c: TS1")
names(dimnames(cm.svm.2.c.1)) = c("", "SVM2c: TS1")
names(dimnames(cm.svm.3.c.1)) = c("", "SVM3c: TS1")
names(dimnames(cm.svm.4.c.1)) = c("", "SVM4c: TS1")
names(dimnames(cm.svm.5.c.1)) = c("", "SVM5c: TS1")
names(dimnames(cm.svm.6.c.1)) = c("", "SVM6c: TS1")


# Apply to test set 2
cm.svm.1.a.2 = table(data.test.2$Occupancy, predict(svm.1.a, newdata = data.test.2, type = "class"))
cm.svm.2.a.2 = table(data.test.2$Occupancy, predict(svm.2.a, newdata = data.test.2, type = "class"))
cm.svm.3.a.2 = table(data.test.2$Occupancy, predict(svm.3.a, newdata = data.test.2, type = "class"))
cm.svm.4.a.2 = table(data.test.2$Occupancy, predict(svm.4.a, newdata = data.test.2, type = "class"))
cm.svm.5.a.2 = table(data.test.2$Occupancy, predict(svm.5.a, newdata = data.test.2, type = "class"))
cm.svm.6.a.2 = table(data.test.2$Occupancy, predict(svm.6.a, newdata = data.test.2, type = "class"))
cm.svm.1.b.2 = table(data.test.2$Occupancy, predict(svm.1.b, newdata = data.test.2, type = "class"))
cm.svm.2.b.2 = table(data.test.2$Occupancy, predict(svm.2.b, newdata = data.test.2, type = "class"))
cm.svm.3.b.2 = table(data.test.2$Occupancy, predict(svm.3.b, newdata = data.test.2, type = "class"))
cm.svm.4.b.2 = table(data.test.2$Occupancy, predict(svm.4.b, newdata = data.test.2, type = "class"))
cm.svm.5.b.2 = table(data.test.2$Occupancy, predict(svm.5.b, newdata = data.test.2, type = "class"))
cm.svm.6.b.2 = table(data.test.2$Occupancy, predict(svm.6.b, newdata = data.test.2, type = "class"))
cm.svm.1.c.2 = table(data.test.2$Occupancy, predict(svm.1.c, newdata = data.test.2, type = "class"))
cm.svm.2.c.2 = table(data.test.2$Occupancy, predict(svm.2.c, newdata = data.test.2, type = "class"))
cm.svm.3.c.2 = table(data.test.2$Occupancy, predict(svm.3.c, newdata = data.test.2, type = "class"))
cm.svm.4.c.2 = table(data.test.2$Occupancy, predict(svm.4.c, newdata = data.test.2, type = "class"))
cm.svm.5.c.2 = table(data.test.2$Occupancy, predict(svm.5.c, newdata = data.test.2, type = "class"))
cm.svm.6.c.2 = table(data.test.2$Occupancy, predict(svm.6.c, newdata = data.test.2, type = "class"))

# Add titles
names(dimnames(cm.svm.1.a.2)) = c("", "SVM1a: TS2")
names(dimnames(cm.svm.2.a.2)) = c("", "SVM2a: TS2")
names(dimnames(cm.svm.3.a.2)) = c("", "SVM3a: TS2")
names(dimnames(cm.svm.4.a.2)) = c("", "SVM4a: TS2")
names(dimnames(cm.svm.5.a.2)) = c("", "SVM5a: TS2")
names(dimnames(cm.svm.6.a.2)) = c("", "SVM6a: TS2")
names(dimnames(cm.svm.1.b.2)) = c("", "SVM1b: TS2")
names(dimnames(cm.svm.2.b.2)) = c("", "SVM2b: TS2")
names(dimnames(cm.svm.3.b.2)) = c("", "SVM3b: TS2")
names(dimnames(cm.svm.4.b.2)) = c("", "SVM4b: TS2")
names(dimnames(cm.svm.5.b.2)) = c("", "SVM5b: TS2")
names(dimnames(cm.svm.6.b.2)) = c("", "SVM6b: TS2")
names(dimnames(cm.svm.1.c.2)) = c("", "SVM1c: TS2")
names(dimnames(cm.svm.2.c.2)) = c("", "SVM2c: TS2")
names(dimnames(cm.svm.3.c.2)) = c("", "SVM3c: TS2")
names(dimnames(cm.svm.4.c.2)) = c("", "SVM4c: TS2")
names(dimnames(cm.svm.5.c.2)) = c("", "SVM5c: TS2")
names(dimnames(cm.svm.6.c.2)) = c("", "SVM6c: TS2")

# Show CMs
cbind(cm.svm.1.a.1, cm.svm.2.a.1, cm.svm.3.a.1, cm.svm.4.a.1, cm.svm.5.a.1, cm.svm.6.a.1)
cbind(cm.svm.1.b.1, cm.svm.2.b.1, cm.svm.3.b.1, cm.svm.4.b.1, cm.svm.5.b.1, cm.svm.6.b.1)
cbind(cm.svm.1.c.1, cm.svm.2.c.1, cm.svm.3.c.1, cm.svm.4.c.1, cm.svm.5.c.1, cm.svm.6.c.1)
cbind(cm.svm.1.a.2, cm.svm.2.a.2, cm.svm.3.a.2, cm.svm.4.a.2, cm.svm.5.a.2, cm.svm.6.a.2)
cbind(cm.svm.1.b.2, cm.svm.2.b.2, cm.svm.3.b.2, cm.svm.4.b.2, cm.svm.5.b.2, cm.svm.6.b.2)
cbind(cm.svm.1.c.2, cm.svm.2.c.2, cm.svm.3.c.2, cm.svm.4.c.2, cm.svm.5.c.2, cm.svm.6.c.2)



# # GBMs - commented out because GBMs were computationally impossible
# # Apply to test set 1
# cm.gbm.1.1 = table(data.test.1$Occupancy, predict(gbm.1, newdata = data.test.1, type = "class"))
# cm.gbm.2.1 = table(data.test.1$Occupancy, predict(gbm.2, newdata = data.test.1, type = "class"))
# cm.gbm.3.1 = table(data.test.1$Occupancy, predict(gbm.3, newdata = data.test.1, type = "class"))
# cm.gbm.4.1 = table(data.test.1$Occupancy, predict(gbm.4, newdata = data.test.1, type = "class"))
# cm.gbm.5.1 = table(data.test.1$Occupancy, predict(gbm.5, newdata = data.test.1, type = "class"))
# cm.gbm.6.1 = table(data.test.1$Occupancy, predict(gbm.6, newdata = data.test.1, type = "class"))
# 
# # Apply to test set 2
# cm.gbm.1.2 = table(data.test.2$Occupancy, predict(gbm.1, newdata = data.test.2, type = "class"))
# cm.gbm.2.2 = table(data.test.2$Occupancy, predict(gbm.2, newdata = data.test.2, type = "class"))
# cm.gbm.3.2 = table(data.test.2$Occupancy, predict(gbm.3, newdata = data.test.2, type = "class"))
# cm.gbm.4.2 = table(data.test.2$Occupancy, predict(gbm.4, newdata = data.test.2, type = "class"))
# cm.gbm.5.2 = table(data.test.2$Occupancy, predict(gbm.5, newdata = data.test.2, type = "class"))
# cm.gbm.6.2 = table(data.test.2$Occupancy, predict(gbm.6, newdata = data.test.2, type = "class"))


# ANN
# Apply to test set 1
cm.ann.1.1 = table(data.test.1$Occupancy, 
                   ifelse(predict(ann.1, newdata = data.test.1) >= 0.5, 1, 0))
cm.ann.2.1 = table(data.test.1$Occupancy,
                   ifelse(predict(ann.2, newdata = data.test.1) >= 0.5, 1, 0))
cm.ann.3.1 = table(data.test.1$Occupancy, 
                   ifelse(predict(ann.3, newdata = data.test.1) >= 0.5, 1, 0))
cm.ann.4.1 = table(data.test.1$Occupancy, 
                   ifelse(predict(ann.4, newdata = data.test.1) >= 0.5, 1, 0))
cm.ann.5.1 = table(data.test.1$Occupancy, 
                   ifelse(predict(ann.5, newdata = data.test.1) >= 0.5, 1, 0))
cm.ann.6.1 = table(data.test.1$Occupancy, 
                   ifelse(predict(ann.6, newdata = data.test.1) >= 0.5, 1, 0))
cm.ann.7.1 = table(data.test.1$Occupancy, 
                   ifelse(predict(ann.7, newdata = data.test.1) >= 0.5, 1, 0))

# Add titles
names(dimnames(cm.ann.1.1)) = c("", "ANN1: TS1")
names(dimnames(cm.ann.2.1)) = c("", "ANN2: TS1")
names(dimnames(cm.ann.3.1)) = c("", "ANN3: TS1")
names(dimnames(cm.ann.4.1)) = c("", "ANN4: TS1")
names(dimnames(cm.ann.5.1)) = c("", "ANN5: TS1")
names(dimnames(cm.ann.6.1)) = c("", "ANN6: TS1")
names(dimnames(cm.ann.7.1)) = c("", "ANN7: TS1")

# Apply to test set 2
cm.ann.1.2 = table(data.test.2$Occupancy, 
                   ifelse(predict(ann.1, newdata = data.test.2) >= 0.5, 1, 0))
cm.ann.2.2 = table(data.test.2$Occupancy, 
                   ifelse(predict(ann.2, newdata = data.test.2) >= 0.5, 1, 0))
cm.ann.3.2 = table(data.test.2$Occupancy, 
                   ifelse(predict(ann.3, newdata = data.test.2) >= 0.5, 1, 0))
cm.ann.4.2 = table(data.test.2$Occupancy, 
                   ifelse(predict(ann.4, newdata = data.test.2) >= 0.5, 1, 0))
cm.ann.5.2 = table(data.test.2$Occupancy, 
                   ifelse(predict(ann.5, newdata = data.test.2) >= 0.5, 1, 0))
cm.ann.6.2 = table(data.test.2$Occupancy, 
                   ifelse(predict(ann.6, newdata = data.test.2) >= 0.5, 1, 0))
cm.ann.7.2 = table(data.test.2$Occupancy, 
                   ifelse(predict(ann.7, newdata = data.test.2) >= 0.5, 1, 0))

# Add titles
names(dimnames(cm.ann.1.2)) = c("", "ANN1: TS2")
names(dimnames(cm.ann.2.2)) = c("", "ANN2: TS2")
names(dimnames(cm.ann.3.2)) = c("", "ANN3: TS2")
names(dimnames(cm.ann.4.2)) = c("", "ANN4: TS2")
names(dimnames(cm.ann.5.2)) = c("", "ANN5: TS2")
names(dimnames(cm.ann.6.2)) = c("", "ANN6: TS2")
names(dimnames(cm.ann.7.2)) = c("", "ANN7: TS2")

# Show CMs
cbind(cm.ann.1.1, cm.ann.2.1, cm.ann.3.1, cm.ann.4.1, cm.ann.5.1, cm.ann.6.1, cm.ann.7.1)
cbind(cm.ann.1.2, cm.ann.2.2, cm.ann.3.2, cm.ann.4.2, cm.ann.5.2, cm.ann.6.2, cm.ann.7.2)



# Logistic Regression
# Test set 1
cm.log.1.1 = table(data.test.1$Occupancy, 
                   ifelse(predict(log.1, newdata = data.test.1, type = "response") >= 0.5, 1, 0))
cm.log.2.1 = table(data.test.1$Occupancy,
                   ifelse(predict(log.2, newdata = data.test.1, type = "response") >= 0.5, 1, 0))
cm.log.4.1 = table(data.test.1$Occupancy,
                   ifelse(predict(log.4, newdata = data.test.1, type = "response") >= 0.5, 1, 0))
cm.log.6.1 = table(data.test.1$Occupancy,
                   ifelse(predict(log.6, newdata = data.test.1, type = "response") >= 0.5, 1, 0))

# Add titles
names(dimnames(cm.log.1.1)) = c("", "LR1: TS1")
names(dimnames(cm.log.2.1)) = c("", "LR2: TS1")
names(dimnames(cm.log.4.1)) = c("", "LR4: TS1")
names(dimnames(cm.log.6.1)) = c("", "LR6: TS1")


# Test set 2
cm.log.1.2 = table(data.test.2$Occupancy,
                   ifelse(predict(log.1, newdata = data.test.2, type = "response") >= 0.5, 1, 0))
cm.log.2.2 = table(data.test.2$Occupancy,
                   ifelse(predict(log.2, newdata = data.test.2, type = "response") >= 0.5, 1, 0))
cm.log.4.2 = table(data.test.2$Occupancy,
                   ifelse(predict(log.4, newdata = data.test.2, type = "response") >= 0.5, 1, 0))
cm.log.6.2 = table(data.test.2$Occupancy,
                   ifelse(predict(log.6, newdata = data.test.2, type = "response") >= 0.5, 1, 0))

# Add titles
names(dimnames(cm.log.1.2)) = c("", "LR1: TS2")
names(dimnames(cm.log.2.2)) = c("", "LR2: TS2")
names(dimnames(cm.log.4.2)) = c("", "LR4: TS2")
names(dimnames(cm.log.6.2)) = c("", "LR6: TS2")

# Add CMs
cbind(cm.log.1.1, cm.log.2.1, cm.log.4.1, cm.log.6.1)
cbind(cm.log.1.2, cm.log.2.2, cm.log.4.2, cm.log.6.2)


### Calculate specificity
# Decision trees
# test set 1
cbind(cm.rpart.1.1[1]/(cm.rpart.1.1[1]+cm.rpart.1.1[3]), 
cm.rpart.2.1[1]/(cm.rpart.2.1[1]+cm.rpart.2.1[3]),
cm.rpart.3.1[1]/(cm.rpart.3.1[1]+cm.rpart.3.1[3]),
cm.rpart.4.1[1]/(cm.rpart.4.1[1]+cm.rpart.4.1[3]),
cm.rpart.5.1[1]/(cm.rpart.5.1[1]+cm.rpart.5.1[3]),
cm.rpart.6.1[1]/(cm.rpart.6.1[1]+cm.rpart.6.1[3]))

# test set 2
cbind(cm.rpart.1.2[1]/(cm.rpart.1.2[1]+cm.rpart.1.2[3]),
cm.rpart.2.2[1]/(cm.rpart.2.2[1]+cm.rpart.2.2[3]),
cm.rpart.3.2[1]/(cm.rpart.3.2[1]+cm.rpart.3.2[3]),
cm.rpart.4.2[1]/(cm.rpart.4.2[1]+cm.rpart.4.2[3]),
cm.rpart.5.2[1]/(cm.rpart.5.2[1]+cm.rpart.5.2[3]),
cm.rpart.6.2[1]/(cm.rpart.6.2[1]+cm.rpart.6.2[3]))


# RFs
# test set 1
cbind(cm.rf.1.1[1]/(cm.rf.1.1[1]+cm.rf.1.1[3]),
cm.rf.2.1[1]/(cm.rf.2.1[1]+cm.rf.2.1[3]),
cm.rf.3.1[1]/(cm.rf.3.1[1]+cm.rf.3.1[3]),
cm.rf.4.1[1]/(cm.rf.4.1[1]+cm.rf.4.1[3]),
cm.rf.5.1[1]/(cm.rf.5.1[1]+cm.rf.5.1[3]),
cm.rf.6.1[1]/(cm.rf.6.1[1]+cm.rf.6.1[3]))

# test set 2
cbind(cm.rf.1.2[1]/(cm.rf.1.2[1]+cm.rf.1.2[3]),
cm.rf.2.2[1]/(cm.rf.2.2[1]+cm.rf.2.2[3]),
cm.rf.3.2[1]/(cm.rf.3.2[1]+cm.rf.3.2[3]),
cm.rf.4.2[1]/(cm.rf.4.2[1]+cm.rf.4.2[3]),
cm.rf.5.2[1]/(cm.rf.5.2[1]+cm.rf.5.2[3]),
cm.rf.6.2[1]/(cm.rf.6.2[1]+cm.rf.6.2[3]))


# SVMs
# test set 1
cbind(cm.svm.1.a.1[1]/(cm.svm.1.a.1[1]+cm.svm.1.a.1[3]),
cm.svm.2.a.1[1]/(cm.svm.2.a.1[1]+cm.svm.2.a.1[3]),
cm.svm.3.a.1[1]/(cm.svm.3.a.1[1]+cm.svm.3.a.1[3]),
cm.svm.4.a.1[1]/(cm.svm.4.a.1[1]+cm.svm.4.a.1[3]),
cm.svm.5.a.1[1]/(cm.svm.5.a.1[1]+cm.svm.5.a.1[3]),
cm.svm.6.a.1[1]/(cm.svm.6.a.1[1]+cm.svm.6.a.1[3]),
cm.svm.1.b.1[1]/(cm.svm.1.b.1[1]+cm.svm.1.b.1[3]),
cm.svm.2.b.1[1]/(cm.svm.2.b.1[1]+cm.svm.2.b.1[3]),
cm.svm.3.b.1[1]/(cm.svm.3.b.1[1]+cm.svm.3.b.1[3]),
cm.svm.4.b.1[1]/(cm.svm.4.b.1[1]+cm.svm.4.b.1[3]),
cm.svm.5.b.1[1]/(cm.svm.5.b.1[1]+cm.svm.5.b.1[3]),
cm.svm.6.b.1[1]/(cm.svm.6.b.1[1]+cm.svm.6.b.1[3]),
cm.svm.1.c.1[1]/(cm.svm.1.c.1[1]+cm.svm.1.c.1[3]),
cm.svm.2.c.1[1]/(cm.svm.2.c.1[1]+cm.svm.2.c.1[3]),
cm.svm.3.c.1[1]/(cm.svm.3.c.1[1]+cm.svm.3.c.1[3]),
cm.svm.4.c.1[1]/(cm.svm.4.c.1[1]+cm.svm.4.c.1[3]),
cm.svm.5.c.1[1]/(cm.svm.5.c.1[1]+cm.svm.5.c.1[3]),
cm.svm.6.c.1[1]/(cm.svm.6.c.1[1]+cm.svm.6.c.1[3]))

# test set 2
cbind(cm.svm.1.a.2[1]/(cm.svm.1.a.2[1]+cm.svm.1.a.2[3]),
cm.svm.2.a.2[1]/(cm.svm.2.a.2[1]+cm.svm.2.a.2[3]),
cm.svm.3.a.2[1]/(cm.svm.3.a.2[1]+cm.svm.3.a.2[3]),
cm.svm.4.a.2[1]/(cm.svm.4.a.2[1]+cm.svm.4.a.2[3]),
cm.svm.5.a.2[1]/(cm.svm.5.a.2[1]+cm.svm.5.a.2[3]),
cm.svm.6.a.2[1]/(cm.svm.6.a.2[1]+cm.svm.6.a.2[3]),
cm.svm.1.b.2[1]/(cm.svm.1.b.2[1]+cm.svm.1.b.2[3]),
cm.svm.2.b.2[1]/(cm.svm.2.b.2[1]+cm.svm.2.b.2[3]),
cm.svm.3.b.2[1]/(cm.svm.3.b.2[1]+cm.svm.3.b.2[3]),
cm.svm.4.b.2[1]/(cm.svm.4.b.2[1]+cm.svm.4.b.2[3]),
cm.svm.5.b.2[1]/(cm.svm.5.b.2[1]+cm.svm.5.b.2[3]),
cm.svm.6.b.2[1]/(cm.svm.6.b.2[1]+cm.svm.6.b.2[3]),
cm.svm.1.c.2[1]/(cm.svm.1.c.2[1]+cm.svm.1.c.2[3]),
cm.svm.2.c.2[1]/(cm.svm.2.c.2[1]+cm.svm.2.c.2[3]),
cm.svm.3.c.2[1]/(cm.svm.3.c.2[1]+cm.svm.3.c.2[3]),
cm.svm.4.c.2[1]/(cm.svm.4.c.2[1]+cm.svm.4.c.2[3]),
cm.svm.5.c.2[1]/(cm.svm.5.c.2[1]+cm.svm.5.c.2[3]),
cm.svm.6.c.2[1]/(cm.svm.5.c.2[1]+cm.svm.5.c.2[3]))


# GBMs
# cm.gbm.1.1[1]/(cm.gbm.1.1[1]+cm.gbm.1.1[3]) # test set 1
# cm.gbm.2.1[1]/(cm.gbm.2.1[1]+cm.gbm.2.1[3])
# cm.gbm.3.1[1]/(cm.gbm.3.1[1]+cm.gbm.3.1[3])
# cm.gbm.4.1[1]/(cm.gbm.4.1[1]+cm.gbm.4.1[3])
# cm.gbm.5.1[1]/(cm.gbm.5.1[1]+cm.gbm.5.1[3])
# cm.gbm.6.1[1]/(cm.gbm.6.1[1]+cm.gbm.6.1[3])
# 
# cm.gbm.1.2[1]/(cm.gbm.1.2[1]+cm.gbm.1.2[3]) # test set 2
# cm.gbm.2.2[1]/(cm.gbm.2.2[1]+cm.gbm.2.2[3])
# cm.gbm.3.2[1]/(cm.gbm.3.2[1]+cm.gbm.3.2[3]) 
# cm.gbm.4.2[1]/(cm.gbm.4.2[1]+cm.gbm.4.2[3])
# cm.gbm.5.2[1]/(cm.gbm.5.2[1]+cm.gbm.5.2[3])
# cm.gbm.6.2[1]/(cm.gbm.6.2[1]+cm.gbm.6.2[3])

# ANNs
# test set 1
cbind(cm.ann.1.1[1]/(cm.ann.1.1[1]+cm.ann.1.1[3]),
cm.ann.2.1[1]/(cm.ann.2.1[1]+cm.ann.2.1[3]),
cm.ann.3.1[1]/(cm.ann.3.1[1]+cm.ann.3.1[3]),
cm.ann.4.1[1]/(cm.ann.4.1[1]+cm.ann.4.1[3]),
cm.ann.5.1[1]/(cm.ann.5.1[1]+cm.ann.5.1[3]),
cm.ann.6.1[1]/(cm.ann.6.1[1]+cm.ann.6.1[3]),
cm.ann.7.1[1]/(cm.ann.7.1[1]+cm.ann.7.1[3]))

# test set 2
cbind(cm.ann.1.2[1]/(cm.ann.1.2[1]+cm.ann.1.2[3]),
cm.ann.2.2[1]/(cm.ann.2.2[1]+cm.ann.2.2[3]),
cm.ann.3.2[1]/(cm.ann.3.2[1]+cm.ann.3.2[3]),
cm.ann.4.2[1]/(cm.ann.4.2[1]+cm.ann.4.2[3]),
cm.ann.5.2[1]/(cm.ann.5.2[1]+cm.ann.5.2[3]),
cm.ann.6.2[1]/(cm.ann.6.2[1]+cm.ann.6.2[3]),
cm.ann.7.2[1]/(cm.ann.7.2[1]+cm.ann.7.2[3]))


# Logistic regression
# test set 1
cbind(cm.log.1.1[1]/(cm.log.1.1[1]+cm.log.1.1[3]),
cm.log.2.1[1]/(cm.log.2.1[1]+cm.log.2.1[3]),
cm.log.4.1[1]/(cm.log.4.1[1]+cm.log.4.1[3]),
cm.log.6.1[1]/(cm.log.6.1[1]+cm.log.6.1[3]))

# test set 2
cbind(cm.log.1.2[1]/(cm.log.1.2[1]+cm.log.1.2[3]),
cm.log.2.2[1]/(cm.log.2.2[1]+cm.log.2.2[3]),
cm.log.4.2[1]/(cm.log.4.2[1]+cm.log.4.2[3]),
cm.log.6.2[1]/(cm.log.6.2[1]+cm.log.6.2[3]))


### Calculate sensitivity
# Decision trees
# test set 1
cbind(cm.rpart.1.1[4]/(cm.rpart.1.1[2]+cm.rpart.1.1[4]),
cm.rpart.2.1[4]/(cm.rpart.2.1[2]+cm.rpart.2.1[4]),
cm.rpart.3.1[4]/(cm.rpart.3.1[2]+cm.rpart.3.1[4]),
cm.rpart.4.1[4]/(cm.rpart.4.1[2]+cm.rpart.4.1[4]),
cm.rpart.5.1[4]/(cm.rpart.5.1[2]+cm.rpart.5.1[4]),
cm.rpart.6.1[4]/(cm.rpart.6.1[2]+cm.rpart.6.1[4]))

# test set 2
cbind(cm.rpart.1.2[4]/(cm.rpart.1.2[2]+cm.rpart.1.2[4]),
cm.rpart.2.2[4]/(cm.rpart.2.2[2]+cm.rpart.2.2[4]),
cm.rpart.3.2[4]/(cm.rpart.3.2[2]+cm.rpart.3.2[4]),
cm.rpart.4.2[4]/(cm.rpart.4.2[2]+cm.rpart.4.2[4]),
cm.rpart.5.2[4]/(cm.rpart.5.2[2]+cm.rpart.5.2[4]),
cm.rpart.6.2[4]/(cm.rpart.6.2[2]+cm.rpart.6.2[4]))


# RFs
# test set 1
cbind(cm.rf.1.1[4]/(cm.rf.1.1[2]+cm.rf.1.1[4]),
cm.rf.2.1[4]/(cm.rf.2.1[2]+cm.rf.2.1[4]),
cm.rf.3.1[4]/(cm.rf.3.1[2]+cm.rf.3.1[4]),
cm.rf.4.1[4]/(cm.rf.4.1[2]+cm.rf.4.1[4]),
cm.rf.5.1[4]/(cm.rf.5.1[2]+cm.rf.5.1[4]),
cm.rf.6.1[4]/(cm.rf.6.1[2]+cm.rf.6.1[4]))

# test set 2
cbind(cm.rf.1.2[4]/(cm.rf.1.2[2]+cm.rf.1.2[4]),
cm.rf.2.2[4]/(cm.rf.2.2[2]+cm.rf.2.2[4]),
cm.rf.3.2[4]/(cm.rf.3.2[2]+cm.rf.3.2[4]),
cm.rf.4.2[4]/(cm.rf.4.2[2]+cm.rf.4.2[4]),
cm.rf.5.2[4]/(cm.rf.5.2[2]+cm.rf.5.2[4]),
cm.rf.6.2[4]/(cm.rf.6.2[2]+cm.rf.6.2[4]))


# SVMs
# test set 1
cbind(cm.svm.1.a.1[4]/(cm.svm.1.a.1[2]+cm.svm.1.a.1[4]), 
cm.svm.2.a.1[4]/(cm.svm.2.a.1[2]+cm.svm.2.a.1[4]),
cm.svm.3.a.1[4]/(cm.svm.3.a.1[2]+cm.svm.3.a.1[4]),
cm.svm.4.a.1[4]/(cm.svm.4.a.1[2]+cm.svm.4.a.1[4]),
cm.svm.5.a.1[4]/(cm.svm.5.a.1[2]+cm.svm.5.a.1[4]),
cm.svm.6.a.1[4]/(cm.svm.6.a.1[2]+cm.svm.6.a.1[4]),
cm.svm.1.b.1[4]/(cm.svm.1.b.1[2]+cm.svm.1.b.1[4]),
cm.svm.2.b.1[4]/(cm.svm.2.b.1[2]+cm.svm.2.b.1[4]),
cm.svm.3.b.1[4]/(cm.svm.3.b.1[2]+cm.svm.3.b.1[4]),
cm.svm.4.b.1[4]/(cm.svm.4.b.1[2]+cm.svm.4.b.1[4]),
cm.svm.5.b.1[4]/(cm.svm.5.b.1[2]+cm.svm.5.b.1[4]),
cm.svm.6.b.1[4]/(cm.svm.6.b.1[2]+cm.svm.6.b.1[4]),
cm.svm.1.c.1[4]/(cm.svm.1.c.1[2]+cm.svm.1.c.1[4]),
cm.svm.2.c.1[4]/(cm.svm.2.c.1[2]+cm.svm.2.c.1[4]),
cm.svm.3.c.1[4]/(cm.svm.3.c.1[2]+cm.svm.3.c.1[4]),
cm.svm.4.c.1[4]/(cm.svm.4.c.1[2]+cm.svm.4.c.1[4]),
cm.svm.5.c.1[4]/(cm.svm.5.c.1[2]+cm.svm.5.c.1[4]),
cm.svm.6.c.1[4]/(cm.svm.6.c.1[2]+cm.svm.6.c.1[4]))

# test set 2
cbind(cm.svm.1.a.2[4]/(cm.svm.1.a.2[2]+cm.svm.1.a.2[4]), 
cm.svm.2.a.2[4]/(cm.svm.2.a.2[2]+cm.svm.2.a.2[4]),
cm.svm.3.a.2[4]/(cm.svm.3.a.2[2]+cm.svm.3.a.2[4]),
cm.svm.4.a.2[4]/(cm.svm.4.a.2[2]+cm.svm.4.a.2[4]),
cm.svm.5.a.2[4]/(cm.svm.5.a.2[2]+cm.svm.5.a.2[4]),
cm.svm.6.a.2[4]/(cm.svm.6.a.2[2]+cm.svm.6.a.2[4]),
cm.svm.1.b.2[4]/(cm.svm.1.b.2[2]+cm.svm.1.b.2[4]),
cm.svm.2.b.2[4]/(cm.svm.2.b.2[2]+cm.svm.2.b.2[4]),
cm.svm.3.b.2[4]/(cm.svm.3.b.2[2]+cm.svm.3.b.2[4]),
cm.svm.4.b.2[4]/(cm.svm.4.b.2[2]+cm.svm.4.b.2[4]),
cm.svm.5.b.2[4]/(cm.svm.5.b.2[2]+cm.svm.5.b.2[4]),
cm.svm.6.b.2[4]/(cm.svm.6.b.2[2]+cm.svm.6.b.2[4]),
cm.svm.1.c.2[4]/(cm.svm.1.c.2[2]+cm.svm.1.c.2[4]),
cm.svm.2.c.2[4]/(cm.svm.2.c.2[2]+cm.svm.2.c.2[4]),
cm.svm.3.c.2[4]/(cm.svm.3.c.2[2]+cm.svm.3.c.2[4]),
cm.svm.4.c.2[4]/(cm.svm.4.c.2[2]+cm.svm.4.c.2[4]),
cm.svm.5.c.2[4]/(cm.svm.5.c.2[2]+cm.svm.5.c.2[4]),
cm.svm.6.c.2[4]/(cm.svm.5.c.2[2]+cm.svm.5.c.2[4]))


# # GBMs 
# cm.gbm.1.1[4]/(cm.gbm.1.1[2]+cm.gbm.1.1[4]) # test set 1
# cm.gbm.2.1[4]/(cm.gbm.2.1[2]+cm.gbm.2.1[4])
# cm.gbm.3.1[4]/(cm.gbm.3.1[2]+cm.gbm.3.1[4])
# cm.gbm.4.1[4]/(cm.gbm.4.1[2]+cm.gbm.4.1[4])
# cm.gbm.5.1[4]/(cm.gbm.5.1[2]+cm.gbm.5.1[4])
# cm.gbm.6.1[4]/(cm.gbm.6.1[2]+cm.gbm.6.1[4])
# 
# cm.gbm.1.2[4]/(cm.gbm.1.2[2]+cm.gbm.1.2[4]) # test set 2
# cm.gbm.2.2[4]/(cm.gbm.2.2[2]+cm.gbm.2.2[4])
# cm.gbm.3.2[4]/(cm.gbm.3.2[2]+cm.gbm.3.2[4]) 
# cm.gbm.4.2[4]/(cm.gbm.4.2[2]+cm.gbm.4.2[4])
# cm.gbm.5.2[4]/(cm.gbm.5.2[2]+cm.gbm.5.2[4])
# cm.gbm.6.2[4]/(cm.gbm.6.2[2]+cm.gbm.6.2[4])


# ANNs
# test set 1
cbind(cm.ann.1.1[4]/(cm.ann.1.1[2]+cm.ann.1.1[4]),
cm.ann.2.1[4]/(cm.ann.2.1[2]+cm.ann.2.1[4]),
cm.ann.3.1[4]/(cm.ann.3.1[2]+cm.ann.3.1[4]),
cm.ann.4.1[4]/(cm.ann.4.1[2]+cm.ann.4.1[4]),
cm.ann.5.1[4]/(cm.ann.5.1[2]+cm.ann.5.1[4]),
cm.ann.6.1[4]/(cm.ann.6.1[2]+cm.ann.6.1[4]),
cm.ann.7.1[4]/(cm.ann.7.1[2]+cm.ann.7.1[4]))

# test set 2
cbind(cm.ann.1.2[4]/(cm.ann.1.2[2]+cm.ann.1.2[4]),
cm.ann.2.2[4]/(cm.ann.2.2[2]+cm.ann.2.2[4]),
cm.ann.3.2[4]/(cm.ann.3.2[2]+cm.ann.3.2[4]),
cm.ann.4.2[4]/(cm.ann.4.2[2]+cm.ann.4.2[4]),
cm.ann.5.2[4]/(cm.ann.5.2[2]+cm.ann.5.2[4]),
cm.ann.6.2[4]/(cm.ann.6.2[2]+cm.ann.6.2[4]),
cm.ann.7.2[4]/(cm.ann.7.2[2]+cm.ann.7.2[4]))


# Logistic regression
# test set 1
cbind(cm.log.1.1[4]/(cm.log.1.1[2]+cm.log.1.1[4]),
cm.log.2.1[4]/(cm.log.2.1[2]+cm.log.2.1[4]),
cm.log.4.1[4]/(cm.log.4.1[2]+cm.log.4.1[4]),
cm.log.6.1[4]/(cm.log.6.1[2]+cm.log.6.1[4]))

# test set 2
cbind(cm.log.1.2[4]/(cm.log.1.2[2]+cm.log.1.2[4]), 
cm.log.2.2[4]/(cm.log.2.2[2]+cm.log.2.2[4]),
cm.log.4.2[4]/(cm.log.4.2[2]+cm.log.4.2[4]),
cm.log.6.2[4]/(cm.log.6.2[2]+cm.log.6.2[4]))