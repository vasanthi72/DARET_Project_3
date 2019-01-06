library(caret)
library(MASS)
library(car)
library(pROC)
library(InformationValue)
library(C50)
library(mltools)
library(rpart)
library(randomForest)
library(ggplot2)
library(reshape2)
library(corrplot)
library(tree)
`EEG_data.(1)` <- read.csv("D:/Abhyayanam/DARET/Projects/Project 3 - EEG/EEG_data (1).csv", stringsAsFactors=FALSE)
eeg <- `EEG_data.(1)`
str(eeg)
eeg$SubjectID <- as.factor(eeg$SubjectID)
eeg$VideoID <- as.factor(eeg$VideoID)
eeg$predefinedlabel <- as.factor(eeg$predefinedlabel)
eeg$user.definedlabeln <- as.factor(eeg$user.definedlabeln)
str(eeg)
sum(is.na(eeg))
summary(eeg)
colnames(eeg)
qntAttention <- quantile(eeg$Attention, probs=c(.25, .75))
qntMediation <- quantile(eeg$Mediation, probs=c(.25, .75))
qntRaw <- quantile(eeg$Raw, probs=c(.25,.75))
qntDelta <- quantile(eeg$Delta, probs=c(.25,.75))
qntTheta <- quantile(eeg$Theta, probs=c(.25,.75))
qntAlpha1 <- quantile(eeg$Alpha1, probs=c(.25,.75))
qntAlpha2 <- quantile(eeg$Alpha2, probs=c(.25,.75))
qntBeta1 <- quantile(eeg$Beta1, probs=c(.25,.75))
qntBeta2 <- quantile(eeg$Beta2, probs=c(.25,.75))
qntGamma1 <- quantile(eeg$Gamma1, probs=c(.25,.75))
qntGamma2 <- quantile(eeg$Gamma2, probs=c(.25,.75))
HAttention <- 1.5 * IQR(eeg$Attention)
HMediation <- 1.5 * IQR(eeg$Mediation)
HRaw <- 1.5 * IQR(eeg$Raw)
HDelta <- 1.5 * IQR(eeg$Delta)
HTheta <- 1.5 * IQR(eeg$Theta)
HAlpha1 <- 1.5 * IQR(eeg$Alpha1)
HAlpha2 <- 1.5 * IQR(eeg$Alpha2)
HBeta1 <- 1.5 * IQR(eeg$Beta1)
HBeta2 <- 1.5 * IQR(eeg$Beta2)
HGamma1 <- 1.5 * IQR(eeg$Gamma1)
HGamma2 <- 1.5 * IQR(eeg$Gamma2)
outeeg <- eeg
outeeg$Attention[outeeg$Attention < (qntAttention[1] - HAttention)] <- qntAttention[1]; outeeg$Attention[outeeg$Attention > (qntAttention[2] + HAttention)] <- qntAttention[2]
outeeg$Mediation[outeeg$Mediation < (qntMediation[1] - HMediation)] <- qntMediation[1]; outeeg$Mediation[outeeg$Mediation > (qntMediation[2] + HMediation)] <- qntMediation[2]
outeeg$Raw[outeeg$Raw < (qntRaw[1] - HRaw)] <- qntRaw[1]; outeeg$Raw[outeeg$Raw > (qntRaw[2] + HRaw)] <- qntRaw[2]
outeeg$Delta[outeeg$Delta < (qntDelta[1] - HDelta)] <- qntDelta[1]; outeeg$Delta[outeeg$Delta > (qntDelta[2] + HDelta)] <- qntDelta[2]
outeeg$Theta[outeeg$Theta < (qntTheta[1] - HTheta)] <- qntTheta[1]; outeeg$Theta[outeeg$Theta > (qntTheta[2] + HTheta)] <- qntTheta[2]
outeeg$Alpha1[outeeg$Alpha1 < (qntAlpha1[1] - HAlpha1)] <- qntAlpha1[1]; outeeg$Alpha1[outeeg$Alpha1 > (qntAlpha1[2] + HAlpha1)] <- qntAlpha1[2]
outeeg$Alpha2[outeeg$Alpha2 < (qntAlpha2[1] - HAlpha2)] <- qntAlpha2[1]; outeeg$Alpha2[outeeg$Alpha2 > (qntAlpha2[2] + HAlpha2)] <- qntAlpha2[2]
outeeg$Beta1[outeeg$Beta1 < (qntBeta1[1] - HBeta1)] <- qntBeta1[1]; outeeg$Beta1[outeeg$Beta1 > (qntBeta1[2] + HBeta1)] <- qntBeta1[2]
outeeg$Beta2[outeeg$Beta2 < (qntBeta2[1] - HBeta2)] <- qntBeta2[1]; outeeg$Beta2[outeeg$Beta2 > (qntBeta2[2] + HBeta2)] <- qntBeta2[2]
outeeg$Gamma1[outeeg$Gamma1 < (qntGamma1[1] - HGamma1)] <- qntGamma1[1]; outeeg$Gamma1[outeeg$Gamma1 > (qntGamma1[2] + HGamma1)] <- qntGamma1[2]
outeeg$Gamma2[outeeg$Gamma2 < (qntGamma2[1] - HGamma2)] <- qntGamma2[1]; outeeg$Gamma2[outeeg$Gamma2 > (qntGamma2[2] + HGamma2)] <- qntGamma2[2]
summary(eeg)
summary(outeeg)
set.seed(246)
smp_size <- floor(0.75 * nrow(outeeg))
train_ind <- sample(seq_len(nrow(outeeg)), size = smp_size)
outeegtrain <- outeeg[train_ind, ]
outeegtest <- outeeg[-train_ind, ]
library(MASS)
logmodel <- glm(pl~.,data = outeegtrain[,-c(1,2,15)], family = binomial(link = 'logit'),control = list (maxit=50))
summary(logmodel)
step_logmodel <- stepAIC(logmodel,method='backward')
summary(step_logmodel)
predlogmodel <- predict(logmodel, newdata = outeegtest, type = "response")
steppredlogmodel <- predict(step_logmodel, newdata = outeegtest, type = "response")
model1 <- tree(pl~., data = outeegtrain[,-c(1,2,15)])
predmodel1 <- predict(model1, outeegtest[,-c(1,2,15)], type = "class")
library(C50)
library(caret)
library(mltools)
library(rpart)
model2 <- rpart(pl~.,data=outeegtrain[,-c(1,2,15)])
model2
predmodel2 <- predict(model2, outeegtest[,-c(1,2,15)], type="class")
train_control <-trainControl(method="cv", number=10)
model3 <-  train(pl~.,data=outeegtrain[,-c(1,2,15)],trControl = train_control,method = "rpart")
predmodel3 <- predict(model3, outeegtest[,-c(1,2,15)])
model4 <- train(pl~.,data=outeegtrain[,-c(1,2,15)],trControl = train_control, method = "C5.0")
predmodel4 <- predict(model4,outeegtest[,-c(1,2,15)])
library(gbm)
library(doParallel)
model5 <- train(pl~.,data=outeegtrain[,-c(1,2,15)],trControl = train_control, method = "bstTree")
predmodel5 <- predict(model5,outeegtest[,-c(1,2,15)])
model6 <- train(pl~.,data=outeegtrain[,-c(1,2,15)],trControl = train_control, method = "C5.0Cost")
predmodel6 <- predict(model6, outeegtest[,-c(1,2,15)])
model7 <- train(pl~.,data=outeegtrain[,-c(1,2,15)],trControl = train_control, method = "C5.0Rules")
predmodel7 <- predict(model7, outeegtest[,-c(1,2,15)])
model8 <- train(pl~.,data=outeegtrain[,-c(1,2,15)],trControl = train_control, method = "C5.0Tree")
predmodel8 <- predict(model8, outeegtest[,-c(1,2,15)])
library(modeltools)
library(strucchange)
library(coin)
library(party)
model9 <- train(pl~.,data=outeegtrain[,-c(1,2,15)],trControl = train_control, method = "ctree")
predmodel9 <- predict(model9, outeegtest[,-c(1,2,15)])
model10 <- train(pl~.,data=outeegtrain[,-c(1,2,15)],trControl = train_control, method = "ctree2")
predmodel10 <- predict(model10, outeegtest[,-c(1,2,15)])
model11 <- train(pl~.,data=outeegtrain[,-c(1,2,15)], trControl = train_control, method = "rf")
predmodel11 <- predict(model11, outeegtest[,-c(1,2,15)])
seed <- 7
metric <- 'Accuracy'
set.seed(seed)
mtry <- sqrt(ncol(outeegtrain[,-c(1,2,15)]))
tunegrid <- expand.grid(.mtry=mtry)
model12 <- train(pl~.,data=outeegtrain[,-c(1,2,15)], method = "rf", metric=metric, tuneGrid=tunegrid, trControl = train_control)
predmodel12 <- predict(model12, outeegtest[,-c(1,2,15)])
control <- trainControl(method='repeatedcv',number=10, repeats=3,search='random')
model13 <- train(pl~.,data=outeegtrain[,-c(1,2,15)], method = "rf", metric=metric, tuneLength=10, trControl = control)
predmodel13 <- predict(model13, outeegtest[,-c(1,2,15)])
control1 <- trainControl(method='repeatedcv',number=5, repeats=3,search='grid')
model14 <- train(pl~.,data=outeegtrain[,-c(1,2,15)], method = "gbm", metric=metric, trControl = control1)
predmodel14 <- predict(model14, outeegtest[,-c(1,2,15)])
Accuracy <- function(mat) {
Accuracy <- ((mat[1,1]+mat[2,2])/(mat[3,3]))
return(Accuracy)
}
RecallorSensitivity <- function(mat) {
RS <- (mat[1,1]/mat[3,1])
return(RS)
}
Specificity <- function(mat) {
Spec <- (mat[2,2]/mat[3,2])
return(Spec)
}
Precision <- function(mat) {
Prec <- (mat[1,1]/mat[1,3])
return(Prec)
}
CM1 <- rbind(CM1, ColTotal=colSums(CM1))
CM1 <- as.matrix(table(predmodel1, outeegtest$pl));
CM1 <- cbind(CM1,RowTotal=rowSums(CM1));
CM1 <- rbind(CM1, ColTotal=colSums(CM1))
Accuracy(CM1);RecallorSensitivity(CM1);Specificity(CM1);Precision(CM1)
CM2 <- rbind(CM2, ColTotal=colSums(CM2))
CM2 <- as.matrix(table(predmodel2, outeegtest$pl));
CM2 <- cbind(CM2,RowTotal=rowSums(CM2));
CM2 <- rbind(CM2, ColTotal=colSums(CM2))
Accuracy(CM2);RecallorSensitivity(CM2);Specificity(CM2);Precision(CM2)
CM3 <- as.matrix(table(predmodel3, outeegtest$pl));
CM3 <- cbind(CM3,RowTotal=rowSums(CM3));
CM3 <- rbind(CM3, ColTotal=colSums(CM3));
Accuracy(CM3);RecallorSensitivity(CM3);Specificity(CM3);Precision(CM3)
CM4 <- as.matrix(table(predmodel4, outeegtest$pl));
CM4 <- cbind(CM4,RowTotal=rowSums(CM4));
CM4 <- rbind(CM4, ColTotal=colSums(CM4));
Accuracy(CM4);RecallorSensitivity(CM4);Specificity(CM4);Precision(CM4)
CM5 <- as.matrix(table(predmodel5, outeegtest$pl));
CM5 <- cbind(CM5,RowTotal=rowSums(CM5));
CM5 <- rbind(CM5, ColTotal=colSums(CM5));
Accuracy(CM5);RecallorSensitivity(CM5);Specificity(CM5);Precision(CM5)
CM6 <- as.matrix(table(predmodel6, outeegtest$pl));
CM6 <- cbind(CM6,RowTotal=rowSums(CM6));
CM6 <- rbind(CM6, ColTotal=colSums(CM6));
Accuracy(CM6);RecallorSensitivity(CM6);Specificity(CM6);Precision(CM6)
CM7 <- as.matrix(table(predmodel7, outeegtest$pl));
CM7 <- cbind(CM7,RowTotal=rowSums(CM7));
CM7 <- rbind(CM7, ColTotal=colSums(CM7));
Accuracy(CM7);RecallorSensitivity(CM7);Specificity(CM7);Precision(CM7)
CM8 <- as.matrix(table(predmodel8, outeegtest$pl));
CM8 <- cbind(CM8,RowTotal=rowSums(CM8));
CM8 <- rbind(CM8, ColTotal=colSums(CM8));
Accuracy(CM8);RecallorSensitivity(CM8);Specificity(CM8);Precision(CM8)
CM9 <- as.matrix(table(predmodel9, outeegtest$pl));
CM9 <- cbind(CM9,RowTotal=rowSums(CM9));
CM9 <- rbind(CM9, ColTotal=colSums(CM9));
Accuracy(CM9);RecallorSensitivity(CM9);Specificity(CM9);Precision(CM9)
CM10 <- as.matrix(table(predmodel10, outeegtest$pl));
CM10 <- cbind(CM10,RowTotal=rowSums(CM10));
CM10 <- rbind(CM10, ColTotal=colSums(CM10));
Accuracy(CM10);RecallorSensitivity(CM10);Specificity(CM10);Precision(CM10)
CM11 <- as.matrix(table(predmodel11, outeegtest$pl));
CM11 <- cbind(CM11,RowTotal=rowSums(CM11));
CM11 <- rbind(CM11, ColTotal=colSums(CM11));
Accuracy(CM11);RecallorSensitivity(CM11);Specificity(CM11);Precision(CM11)
CM12 <- as.matrix(table(predmodel12, outeegtest$pl));
CM12 <- cbind(CM12,RowTotal=rowSums(CM12));
CM12 <- rbind(CM12, ColTotal=colSums(CM12));
Accuracy(CM12);RecallorSensitivity(CM12);Specificity(CM12);Precision(CM12)
CM13 <- as.matrix(table(predmodel13, outeegtest$pl));
CM13 <- cbind(CM13,RowTotal=rowSums(CM13));
CM13 <- rbind(CM13, ColTotal=colSums(CM13));
Accuracy(CM13);RecallorSensitivity(CM13);Specificity(CM13);Precision(CM13)
CM14 <- as.matrix(table(predmodel14, outeegtest$pl));
CM14 <- cbind(CM14,RowTotal=rowSums(CM14));
CM14 <- rbind(CM14, ColTotal=colSums(CM14));
Accuracy(CM14);RecallorSensitivity(CM14);Specificity(CM14);Precision(CM14)
model15 <- train(pl~.,data=outeegtrain[,c(3,4,6,11,12,13,14)], trControl = train_control, method = "rf")
predmodel15 <- predict(model15, outeegtest[,c(3,4,6,11,12,13,14)])
CM15 <- as.matrix(table(predmodel15, outeegtest$pl));
CM15 <- cbind(CM15,RowTotal=rowSums(CM15));
CM15 <- rbind(CM15, ColTotal=colSums(CM15));
Accuracy(CM15);RecallorSensitivity(CM15);Specificity(CM15);Precision(CM15)
varImp(model11)
varImp(model14)
colnames(outeegtrain)
model16 <- train(pl~.,data=outeegtrain[,-c(1:5,15)], trControl = train_control, method = "rf" )
predmodel16 <- predict(model16, outeegtest[,-c(1:5,15)])
CM16 <- as.matrix(table(predmodel16, outeegtest$pl));
CM16 <- cbind(CM16,RowTotal=rowSums(CM16));
CM16 <- rbind(CM16, ColTotal=colSums(CM16));
Accuracy(CM16);RecallorSensitivity(CM16);Specificity(CM16);Precision(CM16)
varImp(model16)
model17 <- train(pl~.,data=outeegtrain[,-c(1:4,10,15)], trControl = train_control, method = "rf" )
predmodel17 <- predict(model17, outeegtest[,-c(1:4,10,15)])
CM17 <- as.matrix(table(predmodel17, outeegtest$pl));
CM17 <- cbind(CM17,RowTotal=rowSums(CM17));
CM17 <- rbind(CM17, ColTotal=colSums(CM17));
Accuracy(CM17);RecallorSensitivity(CM17);Specificity(CM17);Precision(CM17)
model18 <- train(pl~.,data=outeegtrain[,-c(15)], trControl = train_control, method = "rf" )
predmodel18 <- predict(model18, outeegtest[,-c(15)])
CM18 <- as.matrix(table(predmodel18, outeegtest$pl));
CM18 <- cbind(CM18,RowTotal=rowSums(CM18));
CM18 <- rbind(CM18, ColTotal=colSums(CM18));
Accuracy(CM18);RecallorSensitivity(CM18);Specificity(CM18);Precision(CM18)
model20 <- train(pl~.,data=outeegtrain[,-c(1,2)], trControl = train_control, method = "rf")
trainpredmodel20 <- predict(model20, outeegtrain[,-c(1,2)])
predmodel20 <- predict(model20, outeegtest[,-c(1,2)])
CM20 <- as.matrix(table(predmodel20, outeegtest$pl));
CM20 <- cbind(CM20,RowTotal=rowSums(CM20));
CM20 <- rbind(CM20, ColTotal=colSums(CM20));
Accuracy(CM20);RecallorSensitivity(CM20);Specificity(CM20);Precision(CM20)
TCM20 <- as.matrix(table(trainpredmodel20, outeegtrain$pl));
TCM20 <- cbind(TCM20,RowTotal=rowSums(TCM20));
TCM20 <- rbind(TCM20, ColTotal=colSums(TCM20));
Accuracy(TCM20);RecallorSensitivity(TCM20);Specificity(TCM20);Precision(TCM20)
varImp(model20)
varImp(model11)
TCM11
table(outeegtrain$pl)
table(outeegtest$pl)
table(predmodel11)
table(trainpredmodel11)
