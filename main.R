census.data <- read.csv("D:/Software/gitRepository/ml_workspace/project760/input/census.data.v6_addedFeats.csv", header=TRUE, na.strings=c("NA", " NA"))
census.test <- read.csv("D:/Software/gitRepository/ml_workspace/project760/input/census.test.v6_addedFeats.csv", header=TRUE, na.strings=c("NA", " NA"))



train_data = census.data[3:42]
test_data = census.test[3:42]

train_labels = as.factor(census.data[[1]])
test_labels = as.factor(census.test[[1]])

tiny.set <- census.data[c(1,2,3,4,5), 35:42]

base_train <- census.data[1:38];
base_test <- census.test[1:38];

# KNN
# 42, 42, 40, 39 - big losers. 38, 37, 36 are oook

knn_train <- na.omit(base_train)[,3:38]
knn_test <- na.omit(base_test)[1:1000,3:38]
knn_train_labels <- as.factor((na.omit(base_train)[,])[[1]])
knn_test_labels <- as.factor((na.omit(base_test)[1:1000,])[[1]])

library(class)
knn_pred <- knn(data.matrix(knn_train), data.matrix(knn_test), knn_train_labels, k=1)

library(gmodels)
CrossTable(x = knn_test_labels, y = knn_pred, prop.chisq=FALSE)

# C4.5
library(RWeka)
c45_train <- na.omit(base_train)[,1:38]
c45_train$instance.weight = 1
c45_train_labels <- as.factor((na.omit(base_train)[1:100,])[[1]])
#control = Weka_control(R = TRUE, M = 10)
model1 <- J48(Label~., data=c45_train)
#model2 <- J48(Label~., data=c45_train, control = Weka_control(R = TRUE, A = TRUE))
model2 <- J48(Label~., data=c45_train, control = Weka_control(U = TRUE))


c45_test <- na.omit(base_test)[1:150000,1:38]
c45_test_labels <- as.factor((na.omit(base_test)[1:10,])[[1]])
c45_test$instance.weight = 1

c45_pred <- predict(model1, c45_test)
table(c45_test[,1], c45_pred)



c45_pred2 <- predict(model2, c45_test)
table(c45_test[,1], c45_pred)

library(gmodels)
CrossTable(x = c45_test[,1], y = c45_pred, prop.chisq=FALSE)
CrossTable(x = c45_test[,1], y = c45_pred2, prop.chisq=FALSE)

