setwd("~/Documents/DSA1101/Data")
data = read.csv("heart-disease-dsa.csv")
dim(data)
head(data)


#####PART I -- EDA: EXPLORING THE VARIABLES & ASSOCIATION#####

#identify categorical variables
data$sex = as.factor(data$sex)
data$chest.pain = as.factor(data$chest.pain)
data$fbs = as.factor(data$fbs)
data$rest.ecg = as.factor(data$rest.ecg)
data$angina = as.factor(data$angina)
data$vessels = as.factor(data$vessels)
data$blood.disorder = as.factor(data$blood.disorder)
data$disease = as.factor(data$disease)

#understanding distribution of quantitative input variables
hist(data$age,
     freq = FALSE,
     main = "Histogram of age",
     xlab = "age",
     ylab = "density",
     col = "skyblue")
hist(data$bp,
     freq = FALSE,
     main = "Histogram of blood pressure",
     xlab = "bp",
     ylab = "density",
     col = "skyblue")
hist(data$chol,
     freq = FALSE,
     main = "Histogram of cholesterol",
     xlab = "chol",
     ylab = "density",
     col = "skyblue")
hist(data$heart.rate,
     freq = FALSE,
     main = "Histogram of heart.rate",
     xlab = "heart.rate",
     ylab = "density",
     col = "skyblue")
hist(data$st.depression,
     freq = FALSE,
     main = "Histogram of st.depression",
     xlab = "st.depression",
     ylab = "density",
     col = "skyblue")
summary(age)
summary(bp)
summary(chol)
summary(heart.rate)
summary(st.depression)

#understanding distribution of quantitative input variables
table(sex)
table(chest.pain)
table(fbs)
table(rest.ecg)
table(angina)
table(vessels)
table(blood.disorder)
table(disease)

###(3)###

#check association between response variable(categorical) VS quantitative input variables
#-> boxplots
age.box = boxplot(data$age~data$disease,
                  main = "Boxplot of Age",
                  xlab = "disease",
                  ylab = "age",
                  col = "skyblue",
                  pch = 20) ; age.box
bp.box = boxplot(data$bp~data$disease,
                 main = "Boxplot of blood pressure",
                 xlab = "disease",
                 ylab = "bp",
                 col = "skyblue",
                 pch = 20) ; bp.box
chol.box = boxplot(data$chol~data$disease,
                   main = "Boxplot of cholesterol",
                   xlab = "disease",
                   ylab = "cholesterol",
                   col = "skyblue",
                   pch = 20) ; chol.box
heart.rate.box = boxplot(data$heart.rate~data$disease,
                         main = "Boxplot of heart rate",
                         xlab = "disease",
                         ylab = "heart rate",
                         col = "skyblue",
                         pch = 20) ; heart.rate.box
st.depression.box = boxplot(data$st.depression~data$disease,
                            main = "Boxplot of ST depression ",
                            xlab = "disease",
                            ylab = "ST depression",
                            col = "skyblue",
                            pch = 20) ; st.depression.box

#check association between response variable(categorical) VS categorical input variables
#-> barplot
#-> pie chart
#-> contingency table of frequency/ proportion
#-> odds ratio

#contingency tables of categorical input variables VS categorical response variable
attach(data)
tab1 = table(sex, disease)
prob.tab1 = prop.table(tab1, "sex"); prob.tab1
tab2 = table(chest.pain, disease)
prob.tab2 = prop.table(tab2, "chest.pain"); prob.tab2
tab3 = table(fbs, disease)
prob.tab3 = prop.table(tab3, "fbs"); prob.tab3
tab4 = table(rest.ecg, disease)
prob.tab4 = prop.table(tab4, "rest.ecg"); prob.tab4
tab5 = table(angina, disease)
prob.tab5 = prop.table(tab5, "angina"); prob.tab5
tab6 = table(vessels, disease)
prob.tab6 = prop.table(tab6, "vessels"); prob.tab6
tab7 = table(blood.disorder, disease)
prob.tab7 = prop.table(tab7, "blood.disorder"); prob.tab7

#odds ratio for 2x2 categorical variables (sex, fbs, angina)
#sex
odds_male_yes = prob.tab1[4]/ (1-prob.tab1[4]); odds_male_yes
odds_female_yes = prob.tab1[3]/ (1-prob.tab1[3]); odds_female_yes
#fbs
odds_fbs1_yes = prob.tab3[4]/ (1-prob.tab3[4]); odds_fbs1_yes
odds_fbs0_yes = prob.tab3[3]/ (1-prob.tab3[3]); odds_fbs0_yes
#angina
odds_angina1_yes = prob.tab5[4]/ (1-prob.tab5[4]); odds_angina1_yes
odds_angina0_yes = prob.tab5[3]/ (1-prob.tab5[3]); odds_angina0_yes

#barplots of categorical input variables VS categorical response variable
barplot(tab1,
        beside = T,
        legend = T,
        col = c("pink", "purple"),
        main = "sex vs disease")
barplot(tab2,
        beside = T,
        legend = T,
        col = c("pink", "purple", "blue", "yellow"),
        main = "chest pain vs disease")
barplot(tab3,
        beside = T,
        legend = T,
        col = c("pink", "purple"),
        main = "fbs vs disease")
barplot(tab4,
        beside = T,
        legend = T,
        col = c("pink", "purple", "blue"),
        main = "rest ecg vs disease")
barplot(tab5,
        beside = T,
        legend = T,
        col = c("pink", "purple"),
        main = "angina vs disease")
barplot(tab6,
        beside = T,
        legend = T,
        col = c("pink", "purple", "blue", "yellow", "green"),
        main = "vessels vs disease")
barplot(tab7,
        beside = T,
        legend = T,
        col = c("pink", "purple", "blue", "yellow"),
        main = "blood disorder vs disease")

#------------------------------------------------------------------------------------------------------------------#


#####PART II -- METHODS: KNN, DT & LR CLASSIFIERS#####

###(4)###

##(i)##
set.seed(1101)
library(class)
n_folds = 5
n = nrow(data) ; n
fold_id = sample(rep(1:n_folds, length.out = n))
table(fold_id)

#deal with missing values in column blood.disorder
table(data$blood.disorder) #blood.disorder = 2 has the highest frequency
data$blood.disorder[which(data$blood.disorder == 0)] = 2



###KNN CLASSIFIER

#create new dataframe for input variables and response variables
#from Part I -> infer from bp.box and chol.box that bp & chol do not have much association with response variable
#            -> infer from contingency table that fbs does not have much association with response variable
X = data[, c("age", "sex", "chest.pain", "heart.rate", "angina", "st.depression", "vessels", "blood.disorder")]
head(X)
Y = data[, c("disease")]
head(Y)
X$age = as.numeric(as.character(X$age))
X$sex = as.numeric(as.character(X$sex))
X$chest.pain = as.numeric(as.character(X$chest.pain))
X$heart.rate = as.numeric(as.character(X$heart.rate))
X$angina = as.numeric(as.character(X$angina))
X$st.depression = as.numeric(as.character(X$st.depression))
X$vessels = as.numeric(as.character(X$vessels))
X$blood.disorder = as.numeric(as.character(X$blood.disorder))

standardised.X = scale(X)
head(standardised.X)

##(ii)##

#check for best value of k
k_values = c(seq(1, floor(sqrt(n)), 2)) ; k_values

##(iii)##
#check best version of KNN using TPR
tpr_values = numeric(n_folds)
avg.tpr_knn = numeric(length(k_values))
library(class)
set.seed(1101)
for(i in seq_along(k_values)){
  for(j in 1:n_folds){
    test.j = which(fold_id == j)
    pred = knn(standardised.X[-test.j,], 
               standardised.X[test.j,], 
               Y[-test.j], 
               k = k_values[i] )
    confusion.matrix = table(Y[test.j], pred)
    tpr_values[j] = confusion.matrix[2,2] / sum(confusion.matrix[2,])
  }
  avg.tpr_knn[i] = mean(tpr_values)
}
best_k = k_values[which(avg.tpr_knn == max(avg.tpr_knn))]; best_k
#avg_tpr
#[1] 0.7512763 0.7524698 0.7481710 0.7700517 0.7458083 0.7378083 0.7452157 0.7374750 0.7220676
#k value that gives the highest TPR is k = 7 which has TPR = 0.7700517


###(5)###: derive goodness of fit of KNN classifier
knn.model= knn(standardised.X, standardised.X, Y, k = 7, prob = T)
confusion.matrix_knn.model= table(Y, knn.model) ; confusion.matrix_knn.model

#TPR#
tpr_knn.model = confusion.matrix_knn.model[2,2] / sum(confusion.matrix_knn.model[2,]); tpr_knn.model
#TPR of KNN model = 0.826087

#precision
prec_knn.model = confusion.matrix_knn.model[2,2] / sum(confusion.matrix_knn.model[,2]); prec_knn.model
#precision of KNN model = 0.870229

#ROC & AUC
library(ROCR)
prob = ifelse(knn.model == "1", attr(knn.model, "prob"), 1 - attr(knn.model, "prob"))
pred = prediction(prob, Y)
roc = performance(pred, "tpr", "fpr")
auc = performance(pred, measure = "auc")
auc@y.values[[1]]
plot(roc, col = "red", main = paste("Area under the curve:", round(auc@y.values[[1]], 4)))
#AUC value of the KNN model = 0.5535427



###DECISION TREE CLASSIFIER
##4(ii): find best minsplit of DT
data2 = data[, c("age", "sex", "chest.pain", "heart.rate", "angina", "st.depression", "vessels", "blood.disorder", "disease")]
head(data2)
tpr_dt.model = numeric(n_folds)
minsplit = 1:100
avg.tpr_dt = numeric(length(minsplit))
library("rpart")
library("rpart.plot")
for(i in 1:length(minsplit)){
  for(j in 1:n_folds){
    test.j = which(fold_id == j)
    train.data = data2[-test.j,]
    test.data = data2[test.j,]
    fit = rpart(disease ~ . ,
                method = "class",
                data = train.data,
                control = rpart.control(minsplit = minsplit[i]),
                parms = list(split = 'information'))
    pred.2 = predict(fit, newdata = test.data, type = "class")
    confusion.matrix = table(test.data[, 9], pred.2)
    tpr_dt.model[j] = confusion.matrix[2,2] / sum(confusion.matrix[2,])
  }
  avg.tpr_dt[i] = mean(tpr_dt.model)
}
best_minsplit = minsplit[which(avg.tpr_dt == max(avg.tpr_dt))]; best_minsplit
#best minsplit values are 8 and 9
#choose to use best minsplit value = 9

###(5)###: derive goodness of fit of DT classifier

#TPR#
dt.model = rpart(disease ~ .,
           method = "class",
           data = data2,
           control = rpart.control(minsplit = 9),
           parms = list(split = 'information'))
rpart.plot(dt.model) #tree plot
pred_dt.model = predict(dt.model, newdata = data2 , type = "class")
confusion.matrix_dt.model = table(data2[, 9], pred_dt.model)
tpr_dt.model = confusion.matrix_dt.model[2,2] / sum(confusion.matrix_dt.model[2,]); tpr_dt.model
#TPR of DT model = 0.8333333

#precision
prec_dt.model = confusion.matrix_dt.model[2,2] / sum(confusion.matrix_dt.model[,2]); prec_dt.model
#precision of DT model = 0.8984375

#ROC & AUC
library(ROCR)
pred_dt.model_prob = predict(dt.model, newdata = data2, type = "prob")
pred_roc_dt.model = prediction(pred_dt.model_prob[, 2], Y)
perf_dt.model = performance(pred_roc_dt.model, "tpr", "fpr")
plot(perf_dt.model, lwd = 2, col = "pink", main = "ROC Curve for Decision Tree")
auc_dt.model = performance(pred_roc_dt.model, measure = "auc")
auc_dt.model@y.values[[1]]
#AUC value for DT model = 0.9147656


###LR CLASSIFIER
LR.model = glm(disease ~., data = data2,family = binomial(link ="logit"))
summary(LR.model)

#contingency table of proportion
table = table(data2$disease)
prop.table(table)

#TPR#
pred_LR.model = predict(LR.model, newdata = data2, type = "response")
pred_LR.model_outcome = ifelse(pred_LR.model >= 0.46 , 1, 0)
confusion.matrix_LR.model = table(data2[,9], pred_LR.model_outcome)
tpr_LR.model = confusion.matrix_LR.model[2,2] / sum(confusion.matrix_LR.model[2,]); tpr_LR.model
#TPR of LR model = 0.8405797

#precision
prec_LR.model = confusion.matrix_LR.model[2,2] / sum(confusion.matrix_LR.model[,2]) ; prec_LR.model
#precision of LR model = 0.8529412

#ROC & AUC
library(ROCR)
pred.roc_LR.model = prediction(pred_LR.model, Y)
perf_LR.model = performance(pred.roc_LR.model , "tpr", "fpr")
plot(perf_LR.model, lwd = 2, col = "red", add = T)  
auc_LR.model = performance(pred.roc_LR.model , "auc")@y.values[[1]]; auc_LR.model
#AUC value for LR model = 0.9312042

#plotting ROC curve for all 3 classifiers
plot(roc_knn, col = "red", lwd = 2, main = "ROC Curve Comparison")
plot(roc_dt, add = TRUE, col = "pink", lwd = 2)
plot(roc_lr, add = TRUE, col = "blue", lwd = 2)
legend("bottomright", 
       legend = c("KNN", "DT", "LR"),
       col = c("red", "pink", "blue"),
       lwd = 2)
