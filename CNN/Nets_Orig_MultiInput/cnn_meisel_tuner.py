import tensorflow.keras.callbacks
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.layers import Input, Conv1D, Dense, concatenate, BatchNormalization, ReLU, GlobalAveragePooling1D
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers import MaxPooling1D, Dropout
#import time
#import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization
#from tensorflow.python.keras.models import Functional
#from keras.tuners import Hyperband
#import Hyperband
import preparator_old
from Helper.metrics import Metrics
from Helper.evaluation_th_mov import Evaluation


class CNN_Hyper:
    def __init__(self, result, x_train, y_train, x_valid, y_valid, times_valid, path, k, cw=None, name=""):
        # create all used_vars:
        name = "CNN_meisel_Multi_Change_Hyper" + name
        print(name)
        info = self.set_info()
        self.prep = preparator_old.Preparator(result, x_train, y_train, x_valid, y_valid, times_valid, path, k, cw=cw,
                                          name=name, info=info)
        self.metrics = Metrics().metrics

        # build model
        input_shape = {'acc_x': (self.prep.height_train_acc_x, self.prep.width_train_acc_x),
                       'acc_y': (self.prep.height_train_acc_y, self.prep.width_train_acc_y),
                       'acc_z': (self.prep.height_train_acc_z, self.prep.width_train_acc_z),
                       'temp': (self.prep.height_train_temp, self.prep.width_train_temp),
                       'hr': (self.prep.height_train_hr, self.prep.width_train_hr)}
        train_feat = [self.prep.train_x_acc_x, self.prep.train_x_acc_y, self.prep.train_x_acc_z, self.prep.train_x_hr,
                       self.prep.train_x_temp]
        epochs = 2000
        batch_size = 32


        print('START HYPERMODEL!!!!')
        model = CNNHyperModel(input_shape=input_shape, prep=self.prep, metrics=self.metrics)
        tuner = RandomSearch(model, objective="val_loss", #kt.Objective("val_f1", direction="max"),
                            max_trials=100, executions_per_trial=2, directory="/data/NEW/Final_Models_rs")
        # tuner = BayesianOptimization(model, objective="val_loss", max_trials=500, executions_per_trial=2,
        #                             directory='/data/NEW/Final_Models_Baesian', seed=42)
        # tuner = Hyperband(model, objective="val_loss", max_epochs=500, hyperband_iterations=1,
        #                   directory='/data/NEW/Final_Models/tuning_results')
        #naxepochs=30 fhrt zu 92modellen
        # tuner.search_space.update({'batch_size': [16, 32, 64, 128, 256, 512, 1024,2048]})


        if k == 0:
            print('SEARCH TUNER')
            tuner.search_space_summary()
            tuner.search(train_feat, self.prep.train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1,
                         callbacks=[EarlyStopping(monitor="val_loss", patience=50, verbose=1)])
            tuner.results_summary()
        else:
            tuner.reload()

        # Select best combination of hyper parameters the tuner found!
        best_model = tuner.get_best_models(num_models=1)[0]

        self.model_info(best_model, "BESTMODEL")


        print('FIT THE BEST MODEL')
        history, best_model = self.fit_the_model(best_model, epochs, batch_size, name, train_feat, self.prep.train_y)

        print('EVALUATION OF BEST MODEL!')
        valid_feat = [self.prep.valid_x_acc_x, self.prep.valid_x_acc_y, self.prep.valid_x_acc_z, self.prep.valid_x_hr,
                       self.prep.valid_x_temp]
        Evaluation(best_model, prep=self.prep, x_val=valid_feat, y_val=self.prep.valid_y)


    @staticmethod
    def set_info():
        """
        Information for the text_file
        :return: string of info
        """
        return "CNN from Meisel: https://onlinelibrary.wiley.com/doi/full/10.1111/epi.16719 \n" \
               "Input shape changed: additionally, a Reshape layer. Otherwise there was an error. \n" \
               "Right before the output: Added a Flatten layer; otherwise there was an error. \n"

    def create_model(self, input_shape):
        """Creates a CNN-Model"""
        # MODEL ACC_X
        input_layer_acc_x = Input(input_shape['acc_x'])
        conv1_acc_x = Conv1D(filters=64, kernel_size=2, activation=activations.relu)(input_layer_acc_x)
        conv1_acc_x = MaxPooling1D(pool_size=2)(conv1_acc_x)
        conv1_acc_x = Dropout(rate=0.7)(conv1_acc_x)
        conv2_acc_x = Dense(50, activation=activations.relu)(conv1_acc_x)
        conv2_acc_x = Dropout(0.7)(conv2_acc_x)
        # conv2_acc_x = keras.layers.Flatten()(conv2_acc_x)
        output_layer_acc_x = GlobalAveragePooling1D()(conv2_acc_x)
        acc_x_model = Model(inputs=input_layer_acc_x, outputs=output_layer_acc_x)

        # MODEL ACC_Y
        input_layer_acc_y = Input(input_shape['acc_y'])
        conv1_acc_y = Conv1D(filters=64, kernel_size=2, activation=activations.relu)(input_layer_acc_y)
        conv1_acc_y = MaxPooling1D(pool_size=2)(conv1_acc_y)
        conv1_acc_y = Dropout(rate=0.7)(conv1_acc_y)
        conv2_acc_y = Dense(50, activation=activations.relu)(conv1_acc_y)
        conv2_acc_y = Dropout(0.7)(conv2_acc_y)
        # conv2_acc_y = keras.layers.Flatten()(conv2_acc_y)
        output_layer_acc_y = GlobalAveragePooling1D()(conv2_acc_y)
        acc_y_model = Model(inputs=input_layer_acc_y, outputs=output_layer_acc_y)

        # MODEL ACC_Z
        input_layer_acc_z = Input(input_shape['acc_z'])
        conv1_acc_z = Conv1D(filters=64, kernel_size=2, activation=activations.relu)(input_layer_acc_z)
        conv1_acc_z = MaxPooling1D(pool_size=2)(conv1_acc_z)
        conv1_acc_z = Dropout(rate=0.7)(conv1_acc_z)
        conv2_acc_z = Dense(50, activation=activations.relu)(conv1_acc_z)
        conv2_acc_z = Dropout(0.7)(conv2_acc_z)
        # conv2_acc_z = keras.layers.Flatten()(conv2_acc_z)
        output_layer_acc_z = GlobalAveragePooling1D()(conv2_acc_z)
        acc_z_model = Model(inputs=input_layer_acc_z, outputs=output_layer_acc_z)

        # MODEL HR
        input_layer_hr = Input(input_shape['hr'])
        conv1_hr = Conv1D(filters=64, kernel_size=2, activation=activations.relu)(input_layer_hr)
        conv1_hr = MaxPooling1D(pool_size=2)(conv1_hr)
        conv1_hr = Dropout(rate=0.7)(conv1_hr)
        conv2_hr = Dense(50, activation=activations.relu)(conv1_hr)
        conv2_hr = Dropout(0.7)(conv2_hr)
        # conv2_hr = keras.layers.Flatten()(conv2_hr)
        output_layer_hr = GlobalAveragePooling1D()(conv2_hr)
        hr_model = Model(inputs=input_layer_hr, outputs=output_layer_hr)

        # MODEL temp
        input_layer_temp = Input(input_shape['temp'])
        conv1_temp = Conv1D(filters=64, kernel_size=2, activation=activations.relu)(input_layer_temp)
        conv1_temp = MaxPooling1D(pool_size=2)(conv1_temp)
        conv1_temp = Dropout(rate=0.7)(conv1_temp)
        conv2_temp = Dense(50, activation=activations.relu)(conv1_temp)
        conv2_temp = Dropout(0.7)(conv2_temp)
        # conv2_temp = keras.layers.Flatten()(conv2_temp)
        output_layer_temp = GlobalAveragePooling1D()(conv2_temp)
        temp_model = Model(inputs=input_layer_temp, outputs=output_layer_temp)

        concatenated = concatenate([acc_x_model.output, acc_y_model.output, acc_z_model.output, hr_model.output,
                                    temp_model.output])

        out = Dense(self.prep.num_classes, activation='softmax', name='output_layer')(concatenated)

        model = Model([acc_x_model.input, acc_y_model.input, acc_z_model.input, hr_model.input, temp_model.input], out)

        with open(self.prep.path + '/model_summary.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        return model

    def create_any_model(self, input_shape):
        """Every feature gets same model."""
        input_layer = Input(input_shape)

        conv1 = Conv1D(filters=64, kernel_size=2, activation=activations.relu)(input_layer)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        conv1 = Dropout(rate=0.7)(conv1)

        conv2 = Dense(50, activation=activations.relu)(conv1)
        conv2 = Dropout(0.7)(conv2)

        output_layer = GlobalAveragePooling1D()(conv2)

        return Model(inputs=input_layer, outputs=output_layer)

    def model_info(self, model, name):
        """
        This method prints the model summary and saves the model plot under the name "name"
        """
        print(model.summary())
        plot_model(model, self.prep.path + "/" + name + "_model.png", show_shapes=True)

    def train_model(self, model, epochs, batch_size, x_train, y_train, name):
        """
        Train and configures the model
        :param model: model to be trained
        :param epochs: Number of epochs to train the model.
        :param batch_size: Number of samples per gradient update.
        :param x_train: input data feat
        :param y_train: input data label
        :param name: name of model to save
        :return: history object
        """
        self.prep.result['epochs'] = epochs  # noqa
        self.prep.result['batch_size'] = batch_size

        # Callbacks are used for performing different actions during the training
        callbacks = [
            # for saving the model
            ModelCheckpoint(self.prep.path + "/" + name, save_best_only=True, monitor="val_loss"),
            # Reduce learning rate when a metric has stopped improving.
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
            # Stop training when a monitored metric has stopped improving.
            EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]

        self.prep.result['Callbacks'] = ["ModelCheckpoint(self.prep.path + \"/\" + name, save_best_only=True, "
                                         "monitor=\"val_loss\")",
                                         "ReduceLROnPlateau(monitor=\"val_loss\", factor=0.5, patience=20, "
                                         "min_lr=0.0001)",
                                         "EarlyStopping(monitor=\"val_loss\", patience=50, verbose=1)"]

        # Configures the model for training.
        model.compile(
            optimizer=self.optimizer(),
            loss=losses.BinaryCrossentropy(from_logits=True),
            metrics=self.metrics,
        )
        self.prep.result['optimizer'] = "\"adam\""
        self.prep.result['loss'] = 'losses.BinaryCrossentropy()'
        self.prep.result['metrics'] = self.metrics.__str__()

        # Trains the model for a fixed number of epochs (iterations on a dataset).
        validation_split = 0.2
        verbose = 1
        history = model.fit(
            x_train,
            y_train,
            class_weight=self.prep.class_weight,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            verbose=verbose,
        )
        self.prep.result['validation_split'] = validation_split
        self.prep.result['verbose'] = verbose

        model.save(self.prep.path + '/my_model.h5')

        # The history object contains training loss values and metrics values for epochs
        return history

    def optimizer(self):
        """
        Creates optimizer for model compiling
        :return: optimizer
        """
        opt = Adam(learning_rate=0.001)
        self.prep.result['learning_rate'] = 0.001
        return opt

    def fit_the_model(self, model, epochs, batch_size, name, x_train, y_train):
        self.prep.result['epochs'] = epochs  # noqa
        self.prep.result['batch_size'] = batch_size

        # Callbacks are used for performing different actions during the training
        callbacks = [
            # for saving the model
            ModelCheckpoint(self.prep.path + "/" + name, save_best_only=True, monitor="val_loss"),
            # Reduce learning rate when a metric has stopped improving.
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
            # Stop training when a monitored metric has stopped improving.
            EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]

        self.prep.result['Callbacks'] = ["ModelCheckpoint(self.prep.path + \"/\" + name, save_best_only=True, "
                                         "monitor=\"val_loss\")",
                                         "ReduceLROnPlateau(monitor=\"val_loss\", factor=0.5, patience=20, "
                                         "min_lr=0.0001)",
                                         "EarlyStopping(monitor=\"val_loss\", patience=50, verbose=1)"]

        # Trains the model for a fixed number of epochs (iterations on a dataset).
        validation_split = 0.2
        verbose = 1
        history = model.fit(
            x_train,
            y_train,
            class_weight=self.prep.class_weight,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            verbose=verbose,
        )
        self.prep.result['validation_split'] = validation_split
        self.prep.result['verbose'] = verbose

        model.save(self.prep.path + '/my_model.h5')

        # The history object contains training loss values and metrics values for epochs
        return history, model


class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, prep, metrics):
        super().__init__()
        self.input_shape = input_shape
        self.prep = prep
        self.metrics = metrics
        # model_hr.add(Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.9, default=0.7, step=0.05))

    def set_max_pool_size(self, tensor_length, maximum):
        if tensor_length < maximum:
            return tensor_length // 2
        else:
            return maximum

    def build(self, hp):
        # MODEL ACC_X
        input_layer_acc_x = Input(self.input_shape['acc_x'])
        # GROUP 1 START
        conv1_acc_x = Conv1D(filters=hp.Choice('conv_1_filters_acc_x', [32, 64, 128, 256]),
                             kernel_size=hp.Int('conv_1_kernel_size_acc_x', min_value=2, max_value=10),
                             padding='same', activation=activations.relu)(input_layer_acc_x)
        conv1_acc_x = MaxPooling1D(pool_size=hp.Int('conv_1_pool_size_acc_x', min_value=2, max_value=9),
                                   padding='same')(conv1_acc_x)
        conv1_acc_x = Dropout(rate=hp.Float('conv_1_do_rate_acc_x', min_value=0.2, max_value=0.9,
                                            default=0.7, step=0.05))(conv1_acc_x)
        # LOOP START
        i = 1
        for i in range(hp.Int('num_groups_acc_x', 0, 3)):
            max_size = self.set_max_pool_size(conv1_acc_x.shape[1], 10)
            if max_size >= 2:
                conv1_acc_x = Conv1D(filters=hp.Choice(f'conv_{i+1}_filters_acc_x', [32, 64, 128, 256]),
                                     kernel_size=hp.Int(f'conv_{i+1}_kernel_size_acc_x', min_value=2, max_value=max_size),
                                     padding='same', activation=activations.relu)(conv1_acc_x)
                conv1_acc_x = MaxPooling1D(pool_size=hp.Int(f'conv_{i+1}_pool_size_acc_x',
                                                            min_value=2, max_value=max_size),
                                           padding='same')(conv1_acc_x)
                conv1_acc_x = Dropout(rate=hp.Float(f'conv_{i+1}_do_rate_acc_x', min_value=0.2, max_value=0.9,
                                                    default=0.7, step=0.05))(conv1_acc_x)
        # LAST GROUP
        conv2_acc_x = Dense(hp.Int('Dense', min_value=20, max_value=100, step=10), activation=activations.relu)(conv1_acc_x)
        drop_out_rate = hp.Float(f'conv_{i+1}_do_rate_acc_x', min_value=0.2, max_value=0.9, default=0.7, step=0.05)
        conv2_acc_x = Dropout(rate=drop_out_rate)(conv2_acc_x)
        output_layer_acc_x = GlobalAveragePooling1D()(conv2_acc_x)
        acc_x_model = Model(inputs=input_layer_acc_x, outputs=output_layer_acc_x)
        ########################
        # MODEL ACC_Y
        input_layer_acc_y = Input(self.input_shape['acc_y'])
        # GROUP 1 START
        conv1_acc_y = Conv1D(filters=hp.Choice('conv_1_filters_acc_y', [32, 64, 128, 256]),
                             kernel_size=hp.Int('conv_1_kernel_size_acc_y', min_value=2, max_value=10),
                             padding='same', activation=activations.relu)(input_layer_acc_y)
        conv1_acc_y = MaxPooling1D(pool_size=hp.Int('conv_1_pool_size_acc_y', min_value=2, max_value=9),
                                   padding='same')(conv1_acc_y)
        conv1_acc_y = Dropout(rate=hp.Float('conv_1_do_rate_acc_y', min_value=0.2, max_value=0.9,
                                            default=0.7, step=0.05))(conv1_acc_y)
        # LOOP START
        i = 1
        for i in range(hp.Int('num_groups_acc_y', 0, 3)):
            max_size = self.set_max_pool_size(conv1_acc_y.shape[1], 10)
            if max_size >= 2:
                conv1_acc_y = Conv1D(filters=hp.Choice(f'conv_{i + 1}_filters_acc_y', [32, 64, 128, 256]),
                                     kernel_size=hp.Int(f'conv_{i + 1}_kernel_size_acc_y', min_value=2, max_value=max_size),
                                     padding='same', activation=activations.relu)(conv1_acc_y)
                conv1_acc_y = MaxPooling1D(pool_size=hp.Int(f'conv_{i + 1}_pool_size_acc_y',
                                                            min_value=1, max_value=max_size),
                                           padding='same')(conv1_acc_y)
                conv1_acc_y = Dropout(rate=hp.Float(f'conv_{i + 1}_do_rate_acc_y', min_value=0.2, max_value=0.9,
                                                    default=0.7, step=0.05))(conv1_acc_y)
        # LAST GROUP
        conv2_acc_y = Dense(hp.Int('Dense', min_value=20, max_value=100, step=10), activation=activations.relu)(conv1_acc_y)
        drop_out_rate = hp.Float(f'conv_{i + 1}_do_rate_acc_y', min_value=0.2, max_value=0.9, default=0.7, step=0.05)
        conv2_acc_y = Dropout(rate=drop_out_rate)(conv2_acc_y)
        output_layer_acc_y = GlobalAveragePooling1D()(conv2_acc_y)
        acc_y_model = Model(inputs=input_layer_acc_y, outputs=output_layer_acc_y)
        ########################
        # MODEL ACC_Z
        input_layer_acc_z = Input(self.input_shape['acc_z'])
        # GROUP 1 START
        conv1_acc_z = Conv1D(filters=hp.Choice('conv_1_filters_acc_z', [32, 64, 128, 256]),
                             kernel_size=hp.Int('conv_1_kernel_size_acc_z', min_value=2, max_value=10),
                             padding='same', activation=activations.relu)(input_layer_acc_z)
        conv1_acc_z = MaxPooling1D(pool_size=hp.Int('conv_1_pool_size_acc_z', min_value=2, max_value=9),
                                   padding='same')(conv1_acc_z)
        conv1_acc_z = Dropout(rate=hp.Float('conv_1_do_rate_acc_z', min_value=0.2, max_value=0.9,
                                            default=0.7, step=0.05))(conv1_acc_z)
        # LOOP START
        i = 1
        for i in range(hp.Int('num_groups_acc_z', 0, 3)):
            max_size = self.set_max_pool_size(conv1_acc_z.shape[1], 10)
            if max_size >= 2:
                conv1_acc_z = Conv1D(filters=hp.Choice(f'conv_{i + 1}_filters_acc_z', [32, 64, 128, 256]),
                                     kernel_size=hp.Int(f'conv_{i + 1}_kernel_size_acc_z', min_value=2, max_value=max_size),
                                     padding='same', activation=activations.relu)(conv1_acc_z)
                conv1_acc_z = MaxPooling1D(pool_size=hp.Int(f'conv_{i + 1}_pool_size_acc_z',
                                                            min_value=1, max_value=max_size),
                                           padding='same')(conv1_acc_z)
                conv1_acc_z = Dropout(rate=hp.Float(f'conv_{i + 1}_do_rate_acc_z', min_value=0.2, max_value=0.9,
                                                    default=0.7, step=0.05))(conv1_acc_z)
        # LAST GROUP
        conv2_acc_z = Dense(hp.Int('Dense', min_value=20, max_value=100, step=10), activation=activations.relu)(conv1_acc_z)
        drop_out_rate = hp.Float(f'conv_{i + 1}_do_rate_acc_z', min_value=0.2, max_value=0.9, default=0.7, step=0.05)
        conv2_acc_z = Dropout(rate=drop_out_rate)(conv2_acc_z)
        output_layer_acc_z = GlobalAveragePooling1D()(conv2_acc_z)
        acc_z_model = Model(inputs=input_layer_acc_z, outputs=output_layer_acc_z)
        ########################
        # MODEL HR
        input_layer_hr = Input(self.input_shape['hr'])
        # GROUP 1 START
        conv1_hr = Conv1D(filters=hp.Choice('conv_1_filters_hr', [32, 64, 128, 256]),
                          kernel_size=hp.Int('conv_1_kernel_size_hr', min_value=2, max_value=10),
                          padding='same', activation=activations.relu)(input_layer_hr)
        conv1_hr = MaxPooling1D(pool_size=hp.Int('conv_1_pool_size_hr', min_value=2, max_value=9),
                                padding='same')(conv1_hr)
        conv1_hr = Dropout(rate=hp.Float('conv_1_do_rate_hr', min_value=0.2, max_value=0.9,
                                         default=0.7, step=0.05))(conv1_hr)
        # LOOP START
        i = 1
        for i in range(hp.Int('num_groups_hr', 0, 3)):
            max_size = self.set_max_pool_size(conv1_hr.shape[1], 10)
            if max_size >= 2:
                conv1_hr = Conv1D(filters=hp.Choice(f'conv_{i + 1}_filters_hr', [32, 64, 128, 256]),
                                  kernel_size=hp.Int(f'conv_{i + 1}_kernel_size_hr', min_value=2, max_value=max_size),
                                  padding='same', activation=activations.relu)(conv1_hr)
                conv1_hr = MaxPooling1D(pool_size=hp.Int(f'conv_{i + 1}_pool_size_hr',
                                                         min_value=1, max_value=max_size),
                                        padding='same')(conv1_hr)
                conv1_hr = Dropout(rate=hp.Float(f'conv_{i + 1}_do_rate_hr', min_value=0.2, max_value=0.9,
                                                 default=0.7, step=0.05))(conv1_hr)
        # LAST GROUP
        conv2_hr = Dense(hp.Int('Dense', min_value=20, max_value=100, step=10), activation=activations.relu)(conv1_hr)
        drop_out_rate = hp.Float(f'conv_{i + 1}_do_rate_hr', min_value=0.2, max_value=0.9, default=0.7, step=0.05)
        conv2_hr = Dropout(rate=drop_out_rate)(conv2_hr)
        output_layer_hr = GlobalAveragePooling1D()(conv2_hr)
        hr_model = Model(inputs=input_layer_hr, outputs=output_layer_hr)
        ########################
        # MODEL TEMP
        input_layer_temp = Input(self.input_shape['temp'])
        # GROUP 1 START
        conv1_temp = Conv1D(filters=hp.Choice('conv_1_filters_temp', [32, 64, 128, 256]),
                            kernel_size=hp.Int('conv_1_kernel_size_temp', min_value=2, max_value=10),
                            padding='same', activation=activations.relu)(input_layer_temp)
        conv1_temp = MaxPooling1D(pool_size=hp.Int('conv_1_pool_size_temp', min_value=2, max_value=9),
                                  padding='same')(conv1_temp)
        conv1_temp = Dropout(rate=hp.Float('conv_1_do_rate_temp', min_value=0.2, max_value=0.9,
                                           default=0.7, step=0.05))(conv1_temp)
        # LOOP START
        i = 1
        for i in range(hp.Int('num_groups_temp', 0, 3)):
            max_size = self.set_max_pool_size(conv1_temp.shape[1], 10)
            if max_size >= 2:
                conv1_temp = Conv1D(filters=hp.Choice(f'conv_{i + 1}_filters_temp', [32, 64, 128, 256]),
                                    kernel_size=hp.Int(f'conv_{i + 1}_kernel_size_temp', min_value=2, max_value=max_size),
                                    padding='same', activation=activations.relu)(conv1_temp)
                conv1_temp = MaxPooling1D(pool_size=hp.Int(f'conv_{i + 1}_pool_size_temp',
                                                           min_value=1, max_value=max_size),
                                          padding='same')(conv1_temp)
                conv1_temp = Dropout(rate=hp.Float(f'conv_{i + 1}_do_rate_temp', min_value=0.2, max_value=0.9,
                                                   default=0.7, step=0.05))(conv1_temp)
        # LAST GROUP
        conv2_temp = Dense(hp.Int('Dense', min_value=20, max_value=100, step=10), activation=activations.relu)(conv1_temp)
        drop_out_rate = hp.Float(f'conv_{i + 1}_do_rate_temp', min_value=0.2, max_value=0.9, default=0.7, step=0.05)
        conv2_temp = Dropout(rate=drop_out_rate)(conv2_temp)
        output_layer_temp = GlobalAveragePooling1D()(conv2_temp)
        temp_model = Model(inputs=input_layer_temp, outputs=output_layer_temp)

        concatenated = concatenate([acc_x_model.output, acc_y_model.output, acc_z_model.output, hr_model.output,
                                    temp_model.output])

        out = Dense(self.prep.num_classes, activation='softmax', name='output_layer')(concatenated)

        model = Model([acc_x_model.input, acc_y_model.input, acc_z_model.input, hr_model.input, temp_model.input], out)

        with open(self.prep.path + '/model_summary.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))


        model.compile(
            optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)),
            loss=losses.BinaryCrossentropy(from_logits=True),
            metrics=self.metrics
        )

        return model