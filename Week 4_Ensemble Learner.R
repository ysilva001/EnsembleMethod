###################
# Title : Pima Indians Diabetes Data Set
# Data Source: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes  
# Author : Yesenia Silva
# MSDS664x70_Stat Infer Predictive Analytics
###################

##############Packages#############
library(mlbench)
library(ggplot2)
library(dplyr)
library(adabag) ##bagging or boosting method 
library(ipred) ##bagging method 
library(corrplot)
library(gridExtra)
library(e1071)
library(mice) ##imputis missing values with plausible data values drawn from a dist for each missing datapoint
library(pROC) #Display and Analyze ROC Curves
library(caret) ##boosting method

data("PimaIndiansDiabetes")
pima <- rename(PimaIndiansDiabetes)

#####The Objective is to predict the class of diabetes


##DEA
str(pima)
summary(pima)
summary(pima$diabetes)

##Histogram of all variables
par(mfrow = c(3, 3))
hist(pima$pregnant)
hist(pima$age)
hist(pima$glucose)
hist(pima$mass)
hist(pima$pressure)
hist(pima$triceps)
hist(pima$insuliln)
hist(pima$pedigree)
hist(pima$diabetes)

##We can see some zeros - let's get a count
biological_data <- pima[,setdiff(names(pima), c('diabetes', 'pregnant'))]
features_miss_num <- apply(biological_data, 2, function(x) sum(x<=0))
features_miss <- names(biological_data)[ features_miss_num > 0]
features_miss_num

##Count of rows that have zeros
rows_errors <- apply(biological_data, 1, function(x) sum(x<=0)>1) 
sum(rows_errors)

# removing those observation rows with 0 in any of the variables
for (i in 2:6) {
  pima <- pima[-which(pima[, i] == 0), ]
}

# scale the covariates for easier comparison of coefficient posteriors
for (i in 1:8) {
  pima[i] <- scale(pima[i])
}

# Partition the data set into training set and test set with 80:20 ratios 
set.seed(2)
ind = sample(2, nrow(pima), replace = TRUE, prob=c(0.8, 0.2))
trainset = pima[ind == 1,]
testset = pima[ind == 2,]

dim(trainset)
dim(testset)

####Bagging Method using adabag######
set.seed(2)
pima.bagging = bagging(diabetes ~ ., data=trainset, mfinal=10)
pima.predbagging$importance

#use the predicted results from the testing dataset
pima.predbagging <- predict.bagging(pima.bagging, newdata=testset)

#From the predicted results, you can obtain a confusion/classification table
pima.predbagging$confusion

#retrieve the average error of the bagging result
(100 - (pima.predbagging$error))
pima.predbagging$

#confusion matrix
confusionMatrix(table(pima.predbagging, testset$diabetes))

#10-fold cross-validation using bagging 
pima.baggingcv <- bagging.cv(diabetes ~ ., v=10, data=trainset, mfinal=10)
#obtain the confusion matrix from the cross-validation results
pima.baggingcv$confusion
pima.baggingcv$error

####Boosting Method using adabag######
set.seed(2)
pima.boost <- boosting(diabetes ~.,data=trainset,mfinal=10, coeflearn="Freund", boos=FALSE , control=rpart.control(maxdepth=3))

#predict the model 
pima.boost.pred <- predict.boosting(pima.boost,newdata=testset)

#confusion matrix
pima.boost.pred$confusion

#Accuracy
(100 - (pima.boost.pred$error))
#Average Error from predicted results
pima.boost.pred$error

#10-fold cross-validate the training data using boosting
pima.boostcv <- boosting.cv(diabetes ~ ., v=10, data=trainset, mfinal=5,control=rpart.control(cp=0.01))
#obtain confusion matrix
pima.boostcv$confusion
#Accuracy 
(100 - (pima.boostcv$error))
#Error
pima.boostcv$error

###########Calculating the margins of a classifier

##BOOSTING##
#caclculate the margin of boosting ensemble learner
boost.margins <- margins(pima.boost, trainset)
boost.pred.margins <- margins(pima.boost.pred, testset)

#plot a marginal cumulative distribution graph of the boosting classifiers
plot(sort(boost.margins[[1]]), (1:length(boost.margins[[1]]))/length(boost.margins[[1]]), type="l",xlim=c(-1,1),main="Boosting: Margin cumulative distribution graph", xlab="margin", ylab="% observations", col = "blue")
lines(sort(boost.pred.margins[[1]]), (1:length(boost.pred.margins[[1]]))/length(boost.pred.margins[[1]]), type="l", col = "green")
abline(v=0, col="red",lty=2)

#calculate the percentage of negative margin matches training errors and the percentage of negative margin matches test errors
boosting.training.margin = table(boost.margins[[1]] > 0)
boosting.negative.training = as.numeric(boosting.training.margin[1]/boosting.training.margin[2])
boosting.negative.training

boosting.testing.margin = table(boost.pred.margins[[1]] > 0)
boosting.negative.testing = as.numeric(boosting.testing.margin[1]/boosting.testing.margin[2])
boosting.negative.testing

##BAGGING##

#caclculate the margin of bagging ensemble learner
bagging.margins <- margins(pima.bagging, trainset)
bagging.pred.margins <- margins(pima.predbagging, testset)

#plot a marginal cumulative distribution graph of the bagging classifiers
plot(sort(bagging.margins[[1]]), (1:length(bagging.margins[[1]]))/length(bagging.margins[[1]]), type="l",xlim=c(-1,1),main="Bagging: Margin cumulative distribution graph", xlab="margin", ylab="% observations", col = "blue")
lines(sort(bagging.pred.margins[[1]]), (1:length(bagging.pred.margins[[1]]))/length(bagging.pred.margins[[1]]), type="l", col = "green")
abline(v=0, col="red",lty=2)

#compute the percentage of negative margin matches training errors and the percentage of negative margin matches test errors
bagging.training.margin <- table(bagging.margins[[1]] > 0)
bagging.negative.training <- as.numeric(bagging.training.margin[1]/bagging.training.margin[2])
bagging.negative.training

bagging.testing.margin = table(bagging.pred.margins[[1]] > 0)
bagging.negative.testing = as.numeric(bagging.testing.margin[1]/bagging.testing.margin[2])
bagging.negative.testing

#calculate the error evolution of the bossting classifier
boosting.evol.train = errorevol(pima.boost, trainset)
boosting.evol.test = errorevol(pima.boost, testset)
plot(boosting.evol.test$error, type = "l", ylim = c(0, 1),
              main = "Boosting error versus number of trees", xlab = "Iterations",
              ylab = "Error", col = "red", lwd = 2)
lines(boosting.evol.train$error, cex = .5, col = "blue", lty = 2, lwd = 2)
legend("topright", c("test", "train"), col = c("red", "blue"), lty = 1:2, lwd = 2)

##calculate the error evolution of the bagging classifier
bagging.evol.train = errorevol(pima.bagging, trainset)
bagging.evol.test = errorevol(pima.bagging, testset)
plot(bagging.evol.test$error, type = "l", ylim = c(0, 1),
              main = "Bagging error versus number of trees", xlab = "Iterations",
              ylab = "Error", col = "red", lwd = 2)
lines(bagging.evol.train$error, cex = .5, col = "blue", lty = 2, lwd = 2)
legend("topright", c("test", "train"), col = c("red", "blue"), lty = 1:2, lwd = 2)