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
max_features <- 5000
maxlen <- 500

imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, train_labels), c(x_test, test_labels)) %<-% imdb


## -----------------------------------------------------------------------------
vectorize_sequences <- function(sequences, dimension = max_features) {
# Function that applies one-hot-encoding to the sequences.
# Input:
# - sequences: list of sequences; each sequence has length equal to the length of the sentence, and each int value of the sequence corresponds to a word.
# - dimension: max number of words to include; words are ranked by how often they occur (in the training set) and only the most frequent words are kept.
# Output:
# - results: matrix of dim = (n, dimension); each row represents a different sentence, and each column represents a different word; the matrix is filled by 0 and 1 values, depending if the word is present in that sentence or else.
results <- matrix(0, nrow = length(sequences), ncol = dimension)
for (i in 1:length(sequences)) results[i, sequences[[i]]] <- 1
return(results)
}

y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)


## ---- message=FALSE, warning=FALSE--------------------------------------------
model <- keras_model_sequential() %>%
layer_dense(units = 32, activation = "relu", input_shape = c(max_features)) %>%
layer_dense(units = 16, activation = "relu") %>%
layer_dense(units = 1, activation = "sigmoid")

model


## -----------------------------------------------------------------------------
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)


## ----echo=TRUE, warning=FALSE-------------------------------------------------
history <- model %>% fit(
  x = vectorize_sequences(x_train), 
  y = y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 1
)


## ---- fig.height=5, fig.width=7, out.width='100%', message=FALSE--------------
plot(history)


## ---- message = FALSE---------------------------------------------------------
results <- model %>% evaluate(
  x = vectorize_sequences(x_test), 
  y = y_test,
  verbose = 0
)

print(paste("Loss on test data:", results["loss"]))
print(paste("Accuracy on test data:", results["accuracy"]))


## ---- message=FALSE, warning=FALSE--------------------------------------------
model <- keras_model_sequential() %>%
layer_embedding(input_dim = max_features, output_dim = 128) %>%
layer_lstm(units = 32) %>%
layer_dense(units = 1, activation = "sigmoid")

model


## -----------------------------------------------------------------------------
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)


## ----echo=TRUE----------------------------------------------------------------
history <- model %>% fit(
  x = pad_sequences(x_train, maxlen = maxlen), 
  y = y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 1
)


## ---- fig.height=5, fig.width=7, out.width='100%', message=FALSE--------------
plot(history)


## -----------------------------------------------------------------------------
results <- model %>% evaluate(
  x = pad_sequences(x_test, maxlen = maxlen), 
  y = y_test,
  verbose = 0
)

print(paste("Loss on test data:", results["loss"]))
print(paste("Accuracy on test data:", results["accuracy"]))

