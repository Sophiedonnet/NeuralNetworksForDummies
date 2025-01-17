#loading the keras inbuilt mnist dataset
data<-dataset_mnist()

#separating train and test file
train_x<-data$train$x
train_y<-data$train$y
test_x<-data$test$x
test_y<-data$test$y

rm(data)

# converting a 2D array into a 1D array for feeding into the MLP and normalising the matrix
train_x <- array(train_x, dim = c(dim(train_x)[1], prod(dim(train_x)[-1]))) / 255
test_x <- array(test_x, dim = c(dim(test_x)[1], prod(dim(test_x)[-1]))) / 255

image(matrix(train_x[2,],28,28,byrow=T), axes = FALSE,col=gray.colors(255))
#converting the target variable to once hot encoded vectors using keras inbuilt function
train_y_cat<-to_categorical(train_y,10)
test_y_cat<-to_categorical(test_y,10)
train_y <- train_y_cat
test_y <- test_y_cat




model <- keras_model_sequential()

#defining the model with 1 input layer[784 neurons], 1 hidden layer[784 neurons] with dropout rate 0.4 and 1 output layer[10 neurons]
#i.e number of digits from 0 to 9

model %>% 
  layer_dense(units = 784, input_shape = 784) %>% 
  layer_dropout(rate=0.4)%>%
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 20) %>%
  layer_activation(activation = 'relu')%>%
  layer_dense(units = 10) %>% 
  layer_activation(activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam', 
  metrics = c('accuracy')
)


learning_history <- model %>% fit(train_x, train_y, epochs = 30, batch_size = 1000)
loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = 128)

# plots
summary(model)
plot(learning_history)

#Comment utiliser le modèle pour prédire ?
  

prediction <-  model %>% predict(test_x) 
image(x=1:5,y=0:9,as.matrix(prediction[1:5,]))

