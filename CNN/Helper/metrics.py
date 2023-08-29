from tensorflow.keras import metrics as m
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import AUC


class Metrics:
    def __init__(self):
        self.metrics = [
            m.TruePositives(name='tp'),
            m.FalsePositives(name='fp'),
            m.TrueNegatives(name='tn'),
            m.FalseNegatives(name='fn'),
            m.BinaryAccuracy(name='accuracy'),
            m.Precision(name='precision'),
            m.Recall(name='recall'),
            AUC(name='auc'),
            AUC(name='prc', curve='PR'),
            self.f1
        ]

    @staticmethod
    def f1(y_true, y_pred):
        """
        Metric for f_1 score
        :param y_true:
        :param y_pred:
        :return:
        """
        def recall(true, pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(true * pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(true, 0, 1)))
            rec = true_positives / (possible_positives + K.epsilon())
            return rec

        def precision(true, pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(true * pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(pred, 0, 1)))
            prec = true_positives / (predicted_positives + K.epsilon())
            return prec

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
