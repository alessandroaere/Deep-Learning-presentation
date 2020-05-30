## ---- results="hide", message=FALSE-------------------------------------------
devtools::install_github("rstudio/keras")
library(keras)
install_keras()


## -----------------------------------------------------------------------------
library(reticulate)
py_run_string("import numpy as np;
import tensorflow as tf;
import random as python_random;
np.random.seed(123);
python_random.seed(123);
tf.random.set_seed(123);")


## -----------------------------------------------------------------------------
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y


## -----------------------------------------------------------------------------
train_images <- train_images / 255
test_images <- test_images / 255


## -----------------------------------------------------------------------------
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)


## ---- warning=FALSE-----------------------------------------------------------
model <- keras_model_sequential() %>%
layer_dense(units = 256, activation = "relu", input_shape = c(28 * 28)) %>%
layer_dense(units = 128, activation = "relu", input_shape = c(28 * 28)) %>%
layer_dense(units = 64, activation = "relu", input_shape = c(28 * 28)) %>%
layer_dense(units = 10, activation = "softmax")

model


## -----------------------------------------------------------------------------
model %>% compile(
  optimizer = optimizer_adam(lr = 0.001),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)


## ---- message=FALSE, warning=FALSE--------------------------------------------
history <- model %>% fit(
  x = array_reshape(train_images, c(60000, 28 * 28)), 
  y = train_labels, 
  epochs = 10, 
  batch_size = 32,
  validation_split = 0.2,
  verbose = 1
)


## ---- fig.height=5, fig.width=7, out.width='100%', message=FALSE--------------
plot(history)


## -----------------------------------------------------------------------------
results <- model %>% evaluate(
  x = array_reshape(test_images, c(10000, 28 * 28)), 
  y = test_labels,
  verbose = 0
)

print(paste("Loss on test data:", results["loss"]))
print(paste("Accuracy on test data:", results["accuracy"]))


## -----------------------------------------------------------------------------
model <- keras_model_sequential() %>%
layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)) %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
layer_flatten() %>%
layer_dense(units = 64, activation = "relu") %>%
layer_dense(units = 10, activation = "softmax")

model


## -----------------------------------------------------------------------------
model %>% compile(
  optimizer = optimizer_adam(lr = 0.001),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)


## ---- message=FALSE-----------------------------------------------------------
history <- model %>% fit(
  x = array_reshape(train_images, c(60000, 28, 28, 1)), 
  y = train_labels, 
  epochs = 10, 
  batch_size = 32,
  validation_split = 0.2,
  verbose = 1
)


## ---- fig.height=5, fig.width=7, out.width='100%', message=FALSE--------------
plot(history)


## -----------------------------------------------------------------------------
results <- model %>% evaluate(
  x = array_reshape(test_images, c(10000, 28, 28, 1)), 
  y = test_labels,
  verbose = 0
)

print(paste("Loss on test data:", results["loss"]))
print(paste("Accuracy on test data:", results["accuracy"]))

