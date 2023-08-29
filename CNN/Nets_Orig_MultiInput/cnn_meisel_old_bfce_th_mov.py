from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.layers import Input, Conv1D, Dense, concatenate, BatchNormalization, ReLU, GlobalAveragePooling1D
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers import MaxPooling1D, Dropout
import time
import preparator_old
from Helper.metrics import Metrics
from Helper.evaluation_th_mov import Evaluation


class CNN:
    def __init__(self, result, x_train, y_train, x_valid, y_valid, times_valid, path, k, cw=None, name=""):
        # create all used_vars:
        name = "CNN_meisel_Multi_" + name
        print(name)
        info = self.set_info()
        self.prep = preparator_old.Preparator(result, x_train, y_train, x_valid, y_valid, times_valid, path, k, cw=cw,
                                          name=name, info=info)
        self.metrics = Metrics().metrics

        # CREATION OF MODEL
        model = self.create_model()

        self.model_info(model, name)

        # TRAIN MODEL
        epochs = 2000
        batch_size = 2048
        train_feat = [self.prep.train_x_acc_x, self.prep.train_x_acc_y, self.prep.train_x_acc_z, self.prep.train_x_hr,
                      self.prep.train_x_temp]
        start = time.time()
        history = self.train_model(model, epochs=epochs, batch_size=batch_size,
                                   x_train=train_feat, y_train=self.prep.train_y, name="TrainedModel")
        end = time.time()
        result['duration_train'] = end - start

        # EVALUATE MODEL & HANDLE RESULTS
        valid_feat = [self.prep.valid_x_acc_x, self.prep.valid_x_acc_y, self.prep.valid_x_acc_z, self.prep.valid_x_hr,
                      self.prep.valid_x_temp]
        start = time.time()
        Evaluation(model, history, self.prep, x_val=valid_feat, y_val=self.prep.valid_y)
        end = time.time()
        result['duration_eval'] = end - start

    @staticmethod
    def set_info():
        """
        Information for the text_file
        :return: string of info
        """
        return "CNN from Meisel: https://onlinelibrary.wiley.com/doi/full/10.1111/epi.16719 \n" \
               "Input shape changed: additionally, a Reshape layer. Otherwise there was an error. \n" \
               "Right before the output: Added a Flatten layer; otherwise there was an error. \n"

    def create_model(self):
        """Creates a CNN-Model"""
        i_shape = (self.prep.height_train_acc_x, self.prep.width_train_acc_x)
        acc_x_model = self.create_any_model(input_shape=i_shape)
        i_shape = (self.prep.height_train_acc_y, self.prep.width_train_acc_y)
        acc_y_model = self.create_any_model(input_shape=i_shape)
        i_shape = (self.prep.height_train_acc_z, self.prep.width_train_acc_z)
        acc_z_model = self.create_any_model(input_shape=i_shape)

        i_shape = (self.prep.height_train_hr, self.prep.width_train_hr)
        hr_model = self.create_any_model(input_shape=i_shape)

        i_shape = (self.prep.height_train_temp, self.prep.width_train_temp)
        temp_model = self.create_any_model(input_shape=i_shape)

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
        # conv2 = keras.layers.Flatten()(conv2)

        output_layer = GlobalAveragePooling1D()(conv2)

        # output_layer = keras.layers.Dense(self.prep.num_classes, activation="softmax")(output_layer)

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
            # loss=losses.BinaryCrossentropy(from_logits=True),
            loss=losses.BinaryFocalCrossentropy(from_logits=True),
            metrics=self.metrics,
        )
        self.prep.result['optimizer'] = "\"adam\""
        # self.prep.result['loss'] = 'losses.BinaryCrossentropy()'
        self.prep.result['loss'] = 'losses.BinaryFocalCrossentropy()'
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
