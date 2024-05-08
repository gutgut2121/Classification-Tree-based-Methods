# Classification of dry beans using Tree-based Methods

# Import library
library(tree)
library(ISLR)
library(randomForest)
library(MASS)
library(gbm)
library(readxl)

# Read excel input
beans <- read_excel("C:/Users/path")
beans$Class = factor(beans$Class)

# Sepearate the data into training and testing data
set.seed(123)
n = nrow(beans)
train = sample(1:n, floor(n*4/5))
test = (1:n)[-train]
beans.test = beans[test,]

t0 = proc.time()[3]

# Classification tree 
tree.beans = tree(Class ~ ., beans, subset=train)
plot(tree.beans)
text(tree.beans, cex = 0.55)
tree.pred = predict(tree.beans, beans.test, type = "class")
table(tree.pred, beans$Class[test])
tree.acc = sum(diag(table(tree.pred, beans$Class[test])))/sum(table(tree.pred, beans$Class[test]))
t.tree = proc.time()[3]-t0

# Pruned tree
par(mfrow = c(1,1))
prune.beans = prune.misclass(tree.beans, best = 7)
plot(prune.beans)
text(prune.beans, cex = 0.8)
tree.pred = predict(prune.beans, beans.test, type = "class")
table(tree.pred, beans$Class[test])
prune.acc = sum(diag(table(tree.pred, beans$Class[test])))/sum(table(tree.pred, beans$Class[test]))
t.prune = proc.time()[3]-t.tree-t0

# Bagging
set.seed(123)
bag.beans = randomForest(Class ~ ., data = beans, mtry = dim(beans)[2]-1, importance=TRUE, subset = train)
tree.pred = predict(bag.beans, beans.test, type = "class")
table(tree.pred, beans$Class[test])
bag.acc = sum(diag(table(tree.pred, beans$Class[test])))/sum(table(tree.pred, beans$Class[test]))
t.bag = proc.time()[3]-t.prune-t.tree-t0

# Random Forest
set.seed(123)
rf.beans = randomForest(Class ~ ., data = beans, mtry = floor(sqrt(dim(beans)[2]-1)), importance=TRUE, subset = train)
tree.pred = predict(rf.beans, beans.test, type = "class")
table(tree.pred, beans$Class[test])
rf.acc = sum(diag(table(tree.pred, beans$Class[test])))/sum(table(tree.pred, beans$Class[test]))
t.rf = proc.time()[3]-t.bag-t.prune-t.tree-t0

#Summary of the result
summary(tree.beans)
summary(prune.beans)
importance(bag.beans)
varImpPlot(bag.beans)
importance(rf.beans)
varImpPlot(rf.beans)
# summary(boost.beans)

matrix(c(tree.acc,t.tree, prune.acc,t.prune, bag.acc,t.bag, rf.acc,t.rf),nrow=2)

