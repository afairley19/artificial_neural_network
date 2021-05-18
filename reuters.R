FLAGS <- flags(
  flag_numeric("nodes", 64),
  flag_numeric("batch_size", 100),
  flag_string("activation", "relu"),
  flag_numeric("learning_rate", 0.01),
  flag_numeric("epochs", 20)
)

model_tune <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes, activation = FLAGS$activation, input_shape = c(10000)) %>%
  layer_dense(units = FLAGS$nodes, activation = FLAGS$activation) %>%
  layer_dense(units = 46, activation = "softmax")

model_tune %>% compile(
  optimizer = optimizer_adam(lr = FLAGS$learning_rate),
  loss = 'categorical_crossentropy',
  metrics = c('accuracy'))

model_tune %>% fit(
  training_split_x,
  training_split_y,
  epochs = FLAGS$epochs,
  batch_size = FLAGS$batch_size,
  validation_data = list(validation_split_x, validation_split_y)
)