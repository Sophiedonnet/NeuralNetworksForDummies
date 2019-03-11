## ----setup, eval=TRUE,echo=FALSE-----------------------------------------
knitr::opts_chunk$set(error = TRUE)
knitr::opts_chunk$set(echo = TRUE)
suppressMessages(library("tidyverse"))
library(keras)
library(tensorflow)
suppressMessages(library(GGally))

## ----data mnist, eval=TRUE, fig.show='asis', fig.keep='all'--------------
#loading the keras inbuilt mnist dataset
data <- dataset_mnist()

#separating train and test file
train_x <- data$train$x
train_y <- data$train$y
test_x <- data$test$x
test_y <- data$test$y
 

# converting a 2D array into a 1D array for feeding into the MLP and normalising the matrix
train_x <- array(train_x, dim = c(dim(train_x)[1], prod(dim(train_x)[-1]))) / 255
test_x <- array(test_x, dim = c(dim(test_x)[1], prod(dim(test_x)[-1]))) / 255

image(matrix(train_x[2,],28,28,byrow=T), axes = FALSE,col=gray.colors(255))

#converting the target variable to once hot encoded vectors using keras inbuilt function
train_y_cat <- to_categorical(train_y,10)
test_y_cat <- to_categorical(test_y,10)
train_y <- train_y_cat
test_y <- test_y_cat

## ----mnist network  initialisation, eval=TRUE, echo=TRUE-----------------
model <- keras_model_sequential()

## ----mnist networkdefinition, eval =TRUE---------------------------------
model %>% 
layer_dense(units = 784, input_shape = 784) %>% 
layer_dropout(rate = 0.4) %>% 
layer_activation(activation = 'relu') %>% 
layer_dense(units = 10) %>% 
layer_activation(activation = 'softmax')
model %>% compile(
loss = 'categorical_crossentropy',
optimizer = 'adam', 
metrics = c('accuracy')
)

### ----learning echo, eval=TRUE, echo=FALSE--------------------------------
#if (!exists("learning_history") & exists('model')) {learning_history <- model %>% fit(train_x, train_y, epochs = 30, batch_size = 1000)}

## ----learning, eval=FALSE------------------------------------------------
learning_history <- model %>% fit(train_x, train_y, epochs = 10, batch_size = 1000)

## ----learning again, eval=FALSE------------------------------------------
learning_history <- model %>% fit(train_x, train_y, epochs = 10, batch_size = 1000)

## ----testing, eval = TRUE------------------------------------------------
predictions <- model %>% predict_classes(test_x)
predictions_proba <- model %>% predict_proba(test_x)
loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = 128)

## ----output--------------------------------------------------------------
summary(model)
plot(learning_history)

## ----saving and loading keras object-------------------------------------
save_model_hdf5(model, "NN_mnist.h5")


## ----transfo data image--------------------------------------------------
d <- dim(data$train$x)
train_x_picture <- array(0,c(d,1)) 
train_x_picture[,,,1] <- data$train$x/255

## ----def CNN-------------------------------------------------------------
model_convol <- keras_model_sequential()
model_convol %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), use_bias = TRUE, activation = 'relu',data_format = 'channels_last') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.2) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 10) %>% 
  layer_activation(activation = 'softmax')

model_convol %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam', 
  metrics = c('accuracy')
)

## ----fit CNN-------------------------------------------------------------
learning_history_convol <- model_convol %>% fit(train_x_picture, train_y, validation_split = 0.1, epochs = 10, batch_size = 1000)

save_model_hdf5(model_convol, "CNN_mnist.h5")


## ----output CNN----------------------------------------------------------
summary(model_convol)
plot(learning_history_convol)

## ----iris----------------------------------------------------------------
data(iris)
x_iris <- 
  iris %>% 
  select(-Species) %>% 
  as.matrix %>% 
  scale 

y_iris <- to_categorical(as.integer(iris$Species)-1)

## ----stratified_train_test_splitting-------------------------------------
set.seed(0)
ntest <- 15 # number of test samples in each class
test.index <-
  tibble(row_number =1:nrow(iris),Species = iris$Species)  %>% group_by(Species) %>% sample_n(ntest) %>% pull(row_number)
train.index <- (1:nrow(iris))[-test.index]

x_iris_train <- x_iris[train.index,]
y_iris_train <- y_iris[train.index,]
x_iris_test <- x_iris[test.index,]
y_iris_test <- y_iris[test.index,]

## ---- eval=FALSE---------------------------------------------------------
## model <- keras_model_sequential()
## model %>%
## layer_dense(units = 4, input_shape = 4) %>%
## layer_dropout(rate=0.1)%>%
## layer_activation(activation = 'relu') %>%
## layer_dense(units = 3) %>%
## layer_activation(activation = 'softmax')
## 
## model %>% compile(
## loss = 'categorical_crossentropy',
## optimizer = 'adam',
## metrics = c('accuracy')
## )
## learning_history <- model %>% fit(x_iris_train, y_iris_train, epochs = 200, validation_split=0.0)
## loss_and_metrics <- model %>% evaluate(x_iris_test, y_iris_test)
## 
## estimation <- apply(predict(model,x_iris_test),1,which.max)
## truth <- apply(y_iris_test,1,which.max)
## table(estimation, truth)

## ----model_for_iris_2_layers, eval = FALSE-------------------------------
## model_autoencoder <- keras_model_sequential()
## 
## model_autoencoder %>%
##   layer_dense(units = 2, activation = 'linear',input_shape = ncol(x_iris),name = "inter_layer") %>%
##  layer_dense(units = 4, activation = 'linear')
## 
## model_autoencoder %>% compile(
##      loss = 'mse',
##      optimizer = 'adam',
##      metrics = 'mse'
##  )
## 
## model_autoencoder %>% fit(
##      x_iris_train,
##      x_iris_train,
##      epochs = 1000,
##   batch_size = 16,
##   shuffle  = TRUE,
##     validation_split = 0.1,
## )
## 
## model_projection = keras_model(inputs = model_autoencoder$input, outputs = get_layer(model_autoencoder,"inter_layer")$output)
## 
## intermediate_output = predict(model_projection,x_iris_train)
## 
## 

## ---- eval = FALSE-------------------------------------------------------
## library(FactoMineR)
## res.pca <- PCA(x_iris_train, graph = FALSE)
## 
## par(mfrow=c(1,2))
## plot(intermediate_output[,1],intermediate_output[,2],col = y_iris_train %*% (1:3))
## plot(res.pca$ind$coord[,1],res.pca$ind$coord[,2], col = y_iris_train %*% (1:3))
## 

## ----saving_loading------------------------------------------------------
save_model_hdf5(model, "my_model.h5")
model <- load_model_hdf5("my_model.h5")

