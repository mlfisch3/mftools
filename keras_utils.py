import tensorflow as tf
from functools import partial
#NOTE:  This cell exits with error "UnknownError:  [_Derived_]  Fail to find the dnn implementation."  unless tensorflow is imported using:
#   import tensorflow as tf

#  create_dense_model(data, N1=64, N2=64, activation="relu", name="dense_model", metrics=["accuracy"]):
#  create_image():
#  dense_model(x_train, y_train, N1=64, N2=64, activation="relu", name="dense_model", metrics=["accuracy"], batch_size=64, epochs=10, validation_split=0.2):
#  get_submodel(layer_name):
#  lag(df, feature_range):
#  leakyReLU_model(x_train, y_train, N1=64, N2=64, leaky_relu_alpha=0.2, name="leakyReLU_model", metrics=["accuracy"], batch_size=64, epochs=10, validation_split=0.2):
#  plot(df):
#  plot_error(df):
#  plot_history(history):
#  plot_image(image, title='random'):
#  prediction(model,test_x,train_x, df):
#  tf_sigmoid(N = 1, activation='sigmoid', weights=1.0):
#  times_series_lstm(df, split_pct=0.9)
#  visualize_filter(layer_name, f_index=None, iters=50, learning_rate=10):

gpus =  tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

	# try:
	#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
	# except RuntimeError as e:
	#     print(e)


# plot_model = partial(tf.keras.utils.plot_model, to_file='model_tmp.png', show_shapes=True, expand_nested=True, show_layer_names=True, rankdir='LR', dpi=70)

# def generate_data(x, y, batch_size=32):
    
#     def create_example(x, y):      
#         c = np.random.randint(0,2)
#         image = 0.5 * np.random.rand(28, 28, 3)
#         image[:,:,c] +=0.5 * x/255.
#         return image, y, c

#     num_examples = len(y)
    
#     while True:
#         x_batch = np.zeros((batch_size, 28, 28, 3))
#         y_batch = np.zeros((batch_size, ))
#         c_batch = np.zeros((batch_size, ))
        
#         for i in range(0, batch_size):
#             index = np.random.randint(0, num_examples)
#             image, digit, color = create_example(x[index], y[index])
#             x_batch[i] = image
#             y_batch[i] = digit
#             c_batch[i] = color
            
#         yield x_batch, [y_batch, c_batch]

# test_gen = generate_data(x_test, y_test)
# train_gen = generate_data(x_train, y_train)

class Logger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        digit_accuracy = logs.get('digit_accuracy')
        color_accuracy = logs.get('color_accuracy')
        val_digit_accuracy = logs.get('val_digit_accuracy')
        val_color_accuracy = logs.get('val_color_accuracy')
        print('='*30, epoch + 1, '='*30)
        print(f'digit_accuracy: {digit_accuracy:.2f}, color_accuracy: {color_accuracy:.2f}')
        print(f'val_digit_accuracy: {val_digit_accuracy:.2f}, val_color_accuracy: {val_color_accuracy:.2f}')

# log_dir = r'C:\Users\DrD\JUPYTER\COURSES\COURSERA\MULTITASK_MODELS_KERAS\LOGS'
# history = model.fit(train_gen, validation_data=test_gen, steps_per_epoch=200, validation_steps=100, epochs=10, callbacks=[Logger(), tf.keras.callbacks.TensorBoard(log_dir=log_dir)], verbose=False)


def tf_sigmoid(N = 1, activation='sigmoid', weights=1.0):
#with tf.init_scope():
    inpt = tf.keras.Input(shape=(1,))
    initializer = tf.keras.initializers.Constant(weights)
    outpt = tf.keras.layers.Dense(N, kernel_initializer=initializer, activation=activation)(inpt)
    modl = tf.keras.Model(inpt, outpt)
    modl._layers[1].use_bias = False

    return modl

# def create_dense_model(data, N1=64, N2=64, n_categories=10, activation="relu", name="dense_model", metrics=["accuracy"]):
#     inputs = tf.keras.Input(shape=(data.shape[1]))
#     dense = tf.keras.layers.Dense(N1, activation=activation)
#     x = dense(inputs)
#     x = tf.keras.layers.Dense(N2, activation=activation)(x)
#     outputs = tf.keras.layers.Dense(n_categories)(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
#     model.compile(
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         optimizer=tf.keras.optimizers.RMSprop(),
#         metrics=metrics)

#     return model

def create_dense_model(train_data, N1=64, N2=64, activation="relu", name="dense_model", metrics=["accuracy"]):
    
    input_shape = train_data[0][1].shape
    n_categories = len(set(train_data[1]))

    input_ = tf.keras.Input(shape=input_shape, name='input')
    x = tf.keras.layers.Dense(N1, activation=activation, name='dense_1')(input_)
    x = tf.keras.layers.Dense(N2, activation=activation, name='dense_2')(x)
    output = tf.keras.layers.Dense(n_categories, name='output')(x)
    model = tf.keras.Model(inputs=input_, outputs=output, name=name)
    
    return model

def compile_dense_model(model, metrics=["accuracy"]):

    loss_param=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt_param=tf.keras.optimizers.RMSprop()
    model.compile(loss=loss_param, optimizer=opt_param, metrics=["accuracy"])

    return model


# def compile_dense_model(model, opt_param='adam', metrics=["accuracy"]):

#     #loss_param=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     #opt_param=tf.keras.optimizers.RMSprop()
#     model.compile(loss={'output':'sparse_categorical_crossentropy'}, optimizer=opt_param, metrics=["accuracy"])

#     return model

    
def fit_dense_model(model, x_train, y_train, batch_size=64, epochs=2, validation_split=0.2):
    return model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)


def dense_model(x_train, y_train, N1=64, N2=64, activation="relu", name="dense_model", metrics=["accuracy"], batch_size=64, epochs=10, validation_split=0.2):
    inputs = tf.keras.Input(shape=(x_train.shape[1]))
    x = tf.keras.layers.Dense(N1, activation=activation)(inputs)
    x = tf.keras.layers.Dense(N2, activation=activation)(x)
    outputs = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=metrics)

    print(model.summary())
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    model.save(name)
    
    del model
    
    return history


def leakyReLU_model(x_train, y_train, N1=64, N2=64, leaky_relu_alpha=0.2, name="leakyReLU_model", metrics=["accuracy"], batch_size=64, epochs=10, validation_split=0.2):
    inputs = tf.keras.Input(shape=(x_train.shape[1]))
    dense = tf.keras.layers.Dense(N1)
    x = dense(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = tf.keras.layers.Dense(N2)(x)
    x = tf.keras.layers.LeakyReLU(alpha=leaky_relu_alpha)(x)    
    outputs = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=metrics)

    print(model.summary())
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    model.save(name)
    
    del model
    
    return history


def get_submodel(layer_name):
    return tf.keras.models.Model(
        model.input,
        model.get_layer(layer_name).output
    )

def create_image():
    return tf.random.uniform((96, 96, 3), minval=-0.5, maxval=0.5)

def plot_image(image, title='random'):
    image = image - tf.math.reduce_min(image) # shift minimum value to 0
    image = image / tf.math.reduce_max(image) # scale to range [0,1]
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()

def visualize_filter(layer_name, f_index=None, iters=50, learning_rate=10):
    submodel = get_submodel(layer_name)
    num_filters = submodel.output.shape[-1]

    if f_index is None:
        f_index = random.randint(0, num_filters - 1)
  
    #assert num_filters > f_index, 'f_index is out of bounds'  #??? why this check?

    image = create_image()
    verbose_step = int(iters / 10)

    for i in range(0, iters):
        with tf.GradientTape() as tape:
            tape.watch(image)
            out = submodel(tf.expand_dims(image, axis=0))[:,:,:,f_index]
            loss = tf.math.reduce_mean(out)

        grads = tape.gradient(loss, image)
        grads = tf.math.l2_normalize(grads)
        image += grads * learning_rate

        if (i + 1) % verbose_step == 0:
            print(f'Iteration: {i + 1}, Loss: {loss.numpy():.4f}') 

    plot_image(image, f'{layer_name}, {f_index}')



def recur_columns(df_data, target_col, lag, col_offset=0):
    
    df = copy(df_data)
    columns = df.columns
    
    for i in range(1, (lag + 1)):
        for j in columns[col_offset:]:
            name = j + '[t-' + str(i) + ']'
            df[name] = df[j].shift((i))
    
    df['Target'] = df[target_col].shift(-1)
    
    return df

def times_series_lstm(df, split_pct=0.9):


    train = df[:int(split_pct * len(df))].drop(columns = 'dt').values
    test = df[int(split_pct * len(df)):].drop(columns = 'dt').values
    scaler = MinMaxScaler(feature_range = (0, 1))
    train  = scaler.fit_transform(train)
    test   = scaler.transform(test)
    # Split the data into input features and targets
    train_x, train_y = train[:,:-1], train[:,-1]
    test_x, test_y = test[:,:-1], test[:,-1]
    # reshape input to be 3D [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    def create_model(train_x):
        # Create the model
        inputs = keras.layers.Input(shape = (train_x.shape[1], train_x.shape[2]))
        x = keras.layers.LSTM(50,return_sequences =  True)(inputs)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.LSTM(50, return_sequences = True)(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.LSTM(50)(x)
        outputs = keras.layers.Dense(1, activation = 'linear')(x)

        model = keras.Model(inputs = inputs, outputs = outputs)
        model.compile(optimizer = 'adam', loss = "mse")
        return model

# >>> model = create_model(train_x)
# >>> model.summary()

# fit the network
# >>> history = model.fit(train_x, train_y, epochs = 50, batch_size = 72, validation_data = (test_x, test_y), shuffle = False)
#   physical_devices = tf.config.list_physical_devices('GPU')
#   tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

## Evaluate LSTM results

def plot_history(history):
    # plot history
    fig, ax = plt.subplots()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.grid()
    plt.legend()
    ax.set(title='History', xlabel='epoch', ylabel='loss')
    plt.show()

# >>> plot_history(history)

def prediction(model,test_x,train_x, df):
    # Predict using the model
    predict =  model.predict(test_x)

    # Reshape test_x and train_x for visualization  and inverse-scaling purpose
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[2]))

    # Concatenate test_x with predicted value
    predict_ = np.concatenate((test_x, predict),axis = 1)

    # Inverse-scaling to get the real values
    predict_ = scaler.inverse_transform(predict_)
    original_ = scaler.inverse_transform(test)

    # Create dataframe to store the predicted and original values
    pred = pd.DataFrame()
    pred['dt'] = df['dt'][-test_x.shape[0]:]
    pred['Original'] = original_[:,-1]
    pred['Predicted'] = predict_[:,-1]

    # Calculate the error 
    pred['Error'] = pred['Original'] - pred['Predicted']
    
    # Create dataframe for visualization
    df = df[['dt','AverageTemperature']][:-test_x.shape[0]]
    df.columns = ['dt','Original']
    original = df.append(pred[['dt','Original']])
    df.columns = ['dt','Predicted']
    predicted = df.append(pred[['dt','Predicted']])
    original = original.merge(predicted, left_on = 'dt',right_on = 'dt')
    return pred, original

# >>> pred, original = prediction(model, test_x, train_x, df_global_monthly )

def plot_error(df):

    # Plotting the Current and Predicted values
    fig = px.line(title = 'Prediction vs. Actual')
    fig.add_scatter(x = df['dt'], y = df['Original'], name = 'Original', opacity = 0.7)
    fig.add_scatter(x = df['dt'], y = df['Predicted'], name = 'Predicted', opacity = 0.5)
    fig.show()

    fig = px.line(title = 'Error')
    fig = fig.add_scatter(x = df['dt'], y = df['Error'])
    fig.show()


def plot(df):
    # Plotting the Current and Predicted values
    fig = px.line(title = 'Prediction vs. Actual')
    fig.add_scatter(x = df['dt'], y = df['Original'], name = 'Original', opacity = 0.7)
    fig.add_scatter(x = df['dt'], y = df['Predicted'], name = 'Predicted', opacity = 0.5)
    fig.show()

# >>> plot(original)
# >>> plot_error(pred)
