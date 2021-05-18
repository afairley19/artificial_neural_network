FLAGS <- flags(
  flag_numeric("nodes", 5),
  flag_numeric("batch_size", 100),
  flag_string("activation", "relu"),
  flag_numeric("learning_rate", 0.01),
  flag_numeric("epochs", 20)
)

model_tune <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes, activation = FLAGS$activation, input_shape = ncol(x_train)) %>%
  layer_dense(units = FLAGS$nodes, activation = FLAGS$activation) %>%
  layer_dense(units = 1)

model_tune %>% compile(
  optimizer = optimizer_adam(lr = FLAGS$learning_rate),
  loss = 'mse',
  metric = list('mean_absolute_error'))

model_tune %>% fit(
  as.matrix(x_train),
  y_train,
  epochs = FLAGS$epochs,
  batch_size = FLAGS$batch_size,
  validation_data = list(as.matrix(x_val), y_val))