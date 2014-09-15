library(ggplot2)
qplot(magnet_belt_y, accel_belt_y, data=trainClean, col=classe)

summary(as.factor(trainClean$classe))

axis <- names(trainClean)[7:9]
plot(trainClean[, 8:14])
cor(trainClean[, 8:14])



pp <- preProcess(trainSet[, 8:59], "pca")
pp
pca <- predict(pp, trainSet[, 8:59])
