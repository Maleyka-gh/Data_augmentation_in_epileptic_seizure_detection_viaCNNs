import json
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import Helper.plot_time as plot_time
from Helper import Investigation_test
from numpy import sqrt
from numpy import argmax
from sklearn.metrics import roc_curve




class Evaluation:

    def __init__(self, model, history=None, prep=None, x_val=None, y_val=None):
        self.prep = prep

        if x_val is None and y_val is None:
            x_val = self.prep.valid_x
            y_val = self.prep.valid_y

        self.eval_model(x_val, y_val, model)

        # PLOT RESULT
        self.plot_confusion_matrix(y_val, self.prep.result['predicted_class'], "confusion_matrix")
        self.plot_graph(self.prep.valid_times, self.prep.result['predicted_proba'][:, 1],
                        self.prep.result['predicted_class'])  # noqa
        if history is not None:
            self.plot_loss(history, 'plot_loss')


        # SAVE RESULT
        with open(self.prep.path + '/result.json', 'w') as result_file:
            del self.prep.result['predicted_proba']
            del self.prep.result['predicted_class']
            self.prep.result = self.make_json_serializable(self.prep.result)
            json.dump(self.prep.result, result_file)

    def eval_model(self, val_x, val_y, model):
        """
        Evaluates a CNN-Model on the valid datasets.
        :param val_x: val x
        :param val_y: val y
        :param model: Configured CNN-Model
        """
        self.prep.result['KERAS_RESULTS'] = model.evaluate(val_x, val_y, return_dict=True)
        val_y_1_dim = self.reduce_label_dimension(val_y)
        predicted_proba = model.predict(val_x)
        # predicted_class = predicted_proba.argmax(axis=-1)#uncomment
        # best_threshold = 0.149019#comment
        best_threshold = threshold_moving(true_label=val_y_1_dim,
                                          pred_prob=predicted_proba,
                                          save_path=self.prep.path)
        predicted_class = np.array(probs_to_prediction(predicted_proba, best_threshold))#comment
        self.prep.result['predicted_proba'] = predicted_proba
        self.prep.result['predicted_class'] = predicted_class


        non_seizure_ranges_gt, seizure_ranges_gt = self.seizure_indices(val_y_1_dim)
        non_seizure_ranges_pred, seizure_ranges_pred = self.seizure_indices(predicted_class)

        ## Storing the true and predicted labels
        np.save(self.prep.path+'/true_label.npy',val_y_1_dim)
        np.save(self.prep.path+'/predicted_labels.npy',predicted_class)
        time_to_save = np.concatenate(list(self.prep.valid_times.values()), axis=0)
        np.save(self.prep.path+'/time_values.npy',time_to_save)
        np.save(self.prep.path + '/predicted_proba.npy', predicted_proba) #saving predicted probabilities

        condition_positive_samples, condition_positive_seizures, condition_negative_samples = \
            self.compute_condition_values(val_y, seizure_ranges_gt)
        self.prep.result['condition_positive_samples'] = condition_positive_samples
        self.prep.result['condition_positive_seizures'] = condition_positive_seizures
        self.prep.result['condition_negative_samples'] = condition_negative_samples

        tp_samples, tn_samples, fp_samples, fn_samples, tp_seizures, fn_seizures = \
            self.compute_conf_matrix_values(predicted_class, val_y_1_dim, seizure_ranges_gt)
        # tn_sections, fp_sections = self.compute_conf_matrix_values_sections(val_y_1_dim, non_seizure_ranges_pred,
        #                                                                     seizure_ranges_pred)
        self.prep.result['tp_samples'] = tp_samples
        self.prep.result['tp_seizures'] = tp_seizures
        self.prep.result['tn_samples'] = tn_samples
        # self.prep.result['tn_sections'] = tn_sections
        self.prep.result['fp_samples'] = fp_samples
        # self.prep.result['fp_sections'] = fp_sections
        self.prep.result['fn_samples'] = fn_samples
        self.prep.result['fn_seizures'] = fn_seizures


        # sensitivity = tp / (tp+fn)                                recall, true-positive-rate
        try:
            sensitivity_samples = tp_samples / (tp_samples + fn_samples)
        except ZeroDivisionError:
            sensitivity_samples = 0
        self.prep.result['sensitivity_samples'] = sensitivity_samples
        try:
            sensitivity_seizures = tp_seizures / (tp_seizures + fn_seizures)
        except ZeroDivisionError:
            sensitivity_seizures = 0
        self.prep.result['sensitivity_seizures'] = sensitivity_seizures

        # specificity = tn / (tn+fp)                                selectivity, true-negative-rate
        try:
            specificity_samples = tn_samples / (tn_samples + fp_samples)
        except ZeroDivisionError:
            specificity_samples = 0
        self.prep.result['specificity_samples'] = specificity_samples
        # try:
        #     specificity_sections = tn_sections / (tn_sections + fp_sections)
        # except ZeroDivisionError:
        #     specificity_sections = 0
        # self.prep.result['specificity_sections'] = specificity_sections

        # precision = tp / (tp + fp)                                positive-predicted-value
        try:
            precision_samples = tp_samples / (tp_samples + fp_samples)
        except ZeroDivisionError:
            precision_samples = 0
        self.prep.result['precision_samples'] = precision_samples
        # try:
        #     precision_seizures = tp_seizures / (tp_seizures + fp_sections)
        # except ZeroDivisionError:
        #     precision_seizures = 0
        # self.prep.result['precision_seizures'] = precision_seizures

        self.prep.result['num_recognized_seizures'] = str(tp_seizures) + " of " + str(condition_positive_seizures)

        # negative-predicted-value = tn / (tn + fn)
        try:
            negative_predicted_value_samples = tn_samples / (tn_samples + fn_samples)
        except ZeroDivisionError:
            negative_predicted_value_samples = 0
        self.prep.result['negative_predicted_value_samples'] = negative_predicted_value_samples
        # try:
        #     negative_predicted_value_seizures_sections = tn_sections / (tn_sections + fn_seizures)
        # except ZeroDivisionError:
        #     negative_predicted_value_seizures_sections = 0
        # self.prep.result['negative_predicted_value_seizures_sections'] = negative_predicted_value_seizures_sections

        # false-negative-rate = fn / (fn + tp)                      miss-rate
        try:
            false_negative_rate_samples = fn_samples / (fn_samples + tp_samples)
        except ZeroDivisionError:
            false_negative_rate_samples = 0
        self.prep.result['false_negative_rate_samples'] = false_negative_rate_samples
        try:
            false_negative_rate_seizures = fn_seizures / (fn_seizures + tp_seizures)
        except ZeroDivisionError:
            false_negative_rate_seizures = 0
        self.prep.result['false_negative_rate_seizures'] = false_negative_rate_seizures

        # false-positive-rate = fp / (fp+tn)                        fall-out
        try:
            false_positive_rate_samples = fp_samples / (fp_samples + tn_samples)
        except ZeroDivisionError:
            false_positive_rate_samples = 0
        self.prep.result['false_positive_rate_samples'] = false_positive_rate_samples
        # try:
        #     false_positive_rate_sections = fp_sections / (fp_sections + tn_sections)
        # except ZeroDivisionError:
        #     false_positive_rate_sections = 0
        # self.prep.result['false_positive_rate_sections'] = false_positive_rate_sections

        # false-discovery-rate = fp / (fp+tp)
        try:
            false_discovery_rate_samples = fp_samples / (fp_samples + tp_samples)
        except ZeroDivisionError:
            false_discovery_rate_samples = 0
        self.prep.result['false_discovery_rate_samples'] = false_discovery_rate_samples
        # try:
        #     false_discovery_rate_seizures_section = fp_sections / (fp_sections + tp_seizures)
        # except ZeroDivisionError:
        #     false_discovery_rate_seizures_section = 0
        # self.prep.result['false_discovery_rate_seizures_section'] = false_discovery_rate_seizures_section

        # false-omission-rate = fn / (fn+tn)
        try:
            false_omission_rate_samples = fn_samples / (fn_samples + tn_samples)
        except ZeroDivisionError:
            false_omission_rate_samples = 0
        self.prep.result['false_omission_rate_samples'] = false_omission_rate_samples
        # try:
        #     false_omission_rate_seizures_sections = fn_seizures / (fn_seizures + tn_sections)
        # except ZeroDivisionError:
        #     false_omission_rate_seizures_sections = 0
        # self.prep.result['false_omission_rate_seizures_sections'] = false_omission_rate_seizures_sections

        # prevalence-threshold = sqrt(fpr) / (sqrt(tpr)+sqrt(fpr))
        try:
            prevalence_threshold_samples = math.sqrt(false_positive_rate_samples) /\
                                           (math.sqrt(sensitivity_samples) +
                                            math.sqrt(false_positive_rate_samples))
        except ZeroDivisionError:
            prevalence_threshold_samples = 0
        self.prep.result['prevalence_threshold_samples'] = prevalence_threshold_samples
        # try:
        #     prevalence_threshold_seizures_sections = math.sqrt(false_positive_rate_sections) / \
        #                                              (math.sqrt(sensitivity_seizures) +
        #                                               math.sqrt(false_positive_rate_sections))
        # except ZeroDivisionError:
        #     prevalence_threshold_seizures_sections = 0
        # self.prep.result['prevalence_threshold_seizures_sections'] = prevalence_threshold_seizures_sections

        # threat-score = tp / (tp+fn+fp)
        try:
            threat_score_samples = tp_samples / (tp_samples + fn_samples + fp_samples)
        except ZeroDivisionError:
            threat_score_samples = 0
        self.prep.result['threat_score_samples'] = threat_score_samples
        # try:
        #     threat_score_seizures_sections = tp_seizures / (tp_seizures + fn_seizures + fp_sections)
        # except ZeroDivisionError:
        #     threat_score_seizures_sections = 0
        # self.prep.result['threat_score_seizures_sections'] = threat_score_seizures_sections

        # accuracy = (tp+tn) / (tp+tn+fp+fn)
        try:
            accuracy_samples = (tp_samples + tn_samples) / (tp_samples + tn_samples + fp_samples + fn_samples)
        except ZeroDivisionError:
            accuracy_samples = 0
        self.prep.result['accuracy_samples'] = accuracy_samples
        # try:
        #     accuracy_seizures_sections = (tp_seizures + tn_sections) / \
        #                                  (tp_seizures + + tn_sections + fp_sections + fn_seizures)
        # except ZeroDivisionError:
        #     accuracy_seizures_sections = 0
        # self.prep.result['accuracy_seizures_sections'] = accuracy_seizures_sections

        # balanced-accuracy = (TPR+TNR) / 2
        try:
            balanced_accuracy_samples = (sensitivity_samples + specificity_samples) / 2
        except ZeroDivisionError:
            balanced_accuracy_samples = 0
        self.prep.result['balanced_accuracy_samples'] = balanced_accuracy_samples
        # try:
        #     balanced_accuracy_seizures_sections = (sensitivity_seizures / specificity_sections) / 2
        # except ZeroDivisionError:
        #     balanced_accuracy_seizures_sections = 0
        # self.prep.result['balanced_accuracy_seizures_sections'] = balanced_accuracy_seizures_sections

        # f_beta-score = ((1+beta^2)*tp) / ((1+beta^2)*tp+beta^2*fn+fp)
        beta = 1
        name = 'f' + str(beta) + '_score_'
        try:
            f1_score_samples = self.compute_f_score(beta=beta, tp=tp_samples, fn=fn_samples, fp=fp_samples)
        except ZeroDivisionError:
            f1_score_samples = 0
        self.prep.result[name + 'samples'] = f1_score_samples
        # try:
        #     f1_score_seizures_sections = self.compute_f_score(beta=beta, tp=tp_seizures, fn=fn_seizures, fp=fp_sections)
        # except ZeroDivisionError:
        #     f1_score_seizures_sections = 0
        # self.prep.result[name + 'seizures_sections'] = f1_score_seizures_sections

        # 24 error rate
        try:
            error_rate_24_hrs_samples = compute_error_rate_24hrs(false_alarms=fp_samples,
                                                                      all_non_seizures=self.prep.valid_time_non_seizure)
        except ZeroDivisionError:
            error_rate_24_hrs_samples = 'ZeroDivisionError'
        self.prep.result['error_rate_24_hrs_samples'] = error_rate_24_hrs_samples

        try:
            error_rate_24_hrs_samples_10s = get_false_alarm_ns(self.prep.valid_time_non_seizure,self.prep.times, val_y_1_dim,predicted_class,n='10s')
        except ZeroDivisionError:
            error_rate_24_hrs_samples_10s = 'ZeroDivisionError'
        self.prep.result['error_rate_24_hrs_samples_10s'] = error_rate_24_hrs_samples_10s

        try:
            error_rate_24_hrs_samples_90s = get_false_alarm_ns(self.prep.valid_time_non_seizure,self.prep.times, val_y_1_dim, predicted_class,
                                                               n='90s')
        except ZeroDivisionError:
            error_rate_24_hrs_samples_90s = 'ZeroDivisionError'
        self.prep.result['error_rate_24_hrs_samples_90s'] = error_rate_24_hrs_samples_90s

        # try:
        #     error_rate_24_hrs_sections = compute_error_rate_24hrs(false_alarms=fp_sections,
        #                                                           all_non_seizures=self.prep.valid_time_non_seizure)
        # except ZeroDivisionError:
        #     error_rate_24_hrs_sections = 'ZeroDivisionError'
        # self.prep.result['error_rate_24_hrs_sections'] = error_rate_24_hrs_sections
        # Investigation_test.test_investigate(path=self.prep.original_path,test_x=self.prep.valid_x,
        #                                     times=self.prep.valid_times,t_label=val_y_1_dim,
        #                                     p_label=predicted_class,k=self.prep.k)
    @staticmethod
    def compute_f_score(beta, tp, fn, fp):
        """
        Method to compute the f_score. Formula of https://en.wikipedia.org/wiki/F-score
        :return: F1 - Score
        """
        return ((1 + beta ** 2) * tp) / (((1 + beta ** 2) * tp) + (beta ** 2 * fn) + fp)

    @staticmethod
    def seizure_indices(y):
        """
        This method computes the ranges (indices) of consecutive 0's and 1's in y
        :param y: np array 1d labels
        """
        # compute start and end for every value change in y
        changes = np.where(y[:-1] != y[1:])[0]  # after these values there are changes!
        ends = np.append(changes, y.shape[0])
        starts = changes + 1
        starts = np.insert(starts, 0, 0)

        # stack them into pairs
        result = np.stack([starts, ends], axis=-1)
        # split every 2nd value to second; the rest in first:
        first = np.copy(result[::2, :])
        second = np.copy(result[1::2, :])

        # order for returned values depends on first label in y.
        if y[0] == 0:
            return first, second
        else:
            return second, first

    @staticmethod
    def reduce_label_dimension(y):
        """
        Reduce the dimension of y from 2d to 1d
        :param y: np labels (2d or 1d)
        :return: np labels (binary) in 1d
        """
        if y.ndim == 2:
            y = y[:, 1]
        if y.ndim != 1:
            print('ERROR')
        return y

    @staticmethod
    def compute_condition_values(val_y, seizure_ranges):
        """
        Computes the total of samples and seizures in val_y
        :param val_y: Ground Truth for every sample
        :param seizure_ranges: start and end for all seizures
        :return: num of seizures, num of samples in seizures, num of samples of not seizures
        """
        condition_positive_seizures = seizure_ranges.shape[0]
        column_sums = val_y.sum(axis=0)
        condition_negative_samples = column_sums[0]
        condition_positive_samples = column_sums[1]
        return condition_positive_samples, condition_positive_seizures, condition_negative_samples

    @staticmethod
    def compute_conf_matrix_values(pred, val_y, seizure_ranges):
        """
        Every value which is used for a confusion matrix is computed.
        :param pred: predicted labels
        :param val_y: gt labels
        :param seizure_ranges: ranges where a seizure is (indices)
        :return: tuple of different values
        """
        # samples is for every done sample
        tp_sample = np.sum(np.logical_and(pred == 1, val_y == 1))   # 14
        tn_sample = np.sum(np.logical_and(pred == 0, val_y == 0))   # 4
        fp_sample = np.sum(np.logical_and(pred == 1, val_y == 0))   # 17495
        fn_sample = np.sum(np.logical_and(pred == 0, val_y == 1))   # 0
        # seizures is for every seizure at least one positive predicted!
        tp_seizure, fn_seizure = Evaluation.compute_tp_fn_seizure(seizure_ranges, pred) #2, 0
        return tp_sample, tn_sample, fp_sample, fn_sample, tp_seizure, fn_seizure

    @staticmethod
    def compute_tp_fn_seizure(gt, pred):
        """
        Computes the tp and fn for
        :param gt: seizure ranges (indices) in 2d numpy array with 2columns: 'start_index' and 'end_index' for every
        seizure
        :param pred: prediction of model
        :return: true positive and false negative for every seizure
        """
        tp = 0
        fn = 0
        for seizure in gt:
            pred_temp = pred[seizure[0]:seizure[1]]
            if 1 in pred_temp:
                tp = tp + 1
            else:
                fn = fn + 1
        return tp, fn

    def plot_confusion_matrix(self, y_true, y_pred, name):
        """
        Creates and saves the confusion matrix.
        :param y_true: real labels
        :param y_pred: predicted labels.
        :param name: for saving the plot
        """
        fig = plt.figure()
        classes = [i for i in range(0, self.prep.num_classes)]
        import tensorflow as tf
        y_true = np.argmax(y_true, axis=-1, out=None)
        con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
        con_mat_df = pd.DataFrame(con_mat, index=classes, columns=classes)
        sns.heatmap(con_mat_df, annot=True, fmt='g')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.prep.path + "/" + name)
        plt.clf()
        plt.close(fig)

    def plot_graph(self, times, probabilities, classes):
        """
        Plots timeline of each patient (test, valid) and its class assignment.
        :param times: Timestamps corresponding to the probabilities
        :param probabilities: Probabilities in range 0,1 corresponding to the timestamps
        :param classes: Class assignment of the model
        :return:
        """
        all_data = self.reassign_data(times, probabilities, classes, self.prep.valid_y_orig)
        for pat, data in all_data.items():
            plot_time.Plot(x=data['center'], y=data['probas'], real_seizures=data['real_seizures'],
                           save_path=self.prep.path, pat=pat)

    @staticmethod
    def reassign_data(times, probas, labels, real_seizures):
        """
        Creates for every patient in times (dictionary with key of patients) a dataframe. These dataframes are stored
        in another dictionary with same keys. The values of each patient (in probas, labels and real_seizures) are
        concatenated and not split by patient. Only times contains the correct amount of data for each patient.
        Therefore, a sum s is needed to always cut out the correct section from the values and assign it to the correct
        patient.
        :param times: Dictionary of timestamps for each patient for each window.
        :param probas: Probabilities of class assignment from model
        :param labels: Assigned Labels from model
        :param real_seizures: Ground Truth
        :return: Dictionary of dataframes.
        """
        s = 0
        dfs = dict()
        for key, value in times.items():
            df = pd.DataFrame(value, columns=['start', 'end', 'center'])
            size = value.shape[0]
            df['probas'] = probas[s:size + s]
            df['predictions'] = labels[s:size + s]
            df['real_seizures'] = real_seizures[s:size + s]
            dfs[key] = df
            s += size
        return dfs

    def plot_loss(self, history, name):
        """
        Plots changes over the epochs.
        :param history: History object from fitting the model.
        :param name: for saving the plot
        """
        metric = "loss"
        fig = plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric])
        plt.title("model " + metric)
        plt.ylabel(metric, fontsize="large")
        plt.xlabel("epoch", fontsize="large")
        plt.legend(["train", "val"], loc="best")
        plt.savefig(self.prep.path + "/" + name)
        plt.clf()
        plt.close(fig)

    @staticmethod
    def compute_conf_matrix_values_sections(val_y_1_dim, non_seizure_ranges_pred, seizure_ranges_pred):
        """
        This method computes values for a confusion matrix considering sections and not each label per se.
        :param val_y_1_dim: gt labels
        :param non_seizure_ranges_pred: ranges of consecutive non-seizure (0) predictions
        :param seizure_ranges_pred: ranges of consecutive seizure (1) predictions
        :return: tn and fp of consecutive sections
        """
        tn_sections = 0
        fp_sections = 0
        for non_sei in non_seizure_ranges_pred:
            temp_gt = val_y_1_dim[non_sei[0]:non_sei[1] + 1]
            if not temp_gt.any():  # only zeros
                tn_sections = tn_sections + 1
        for sei in seizure_ranges_pred:
            temp_gt = val_y_1_dim[sei[0]:sei[1] + 1]
            if not temp_gt.any():
                fp_sections = fp_sections + 1

        return tn_sections, fp_sections

    @staticmethod
    def make_json_serializable(dict_to_json):
        """
        To save the results in json we cant use np-data types. This method converts np.floats and np.ints to python ints
        and floats.
        :param dict_to_json: dictionary with different data types
        :return: same dictionary without np.float or np.int data type
        """
        for key, val in dict_to_json.items():
            if isinstance(val, (np.floating, np.integer)):
                dict_to_json[key] = val.item()
            try:
                if math.isnan(dict_to_json[key]):
                    dict_to_json[key] = 'NaN'
            except TypeError:  # it is a list maybe
                pass
        return dict_to_json


def compute_error_rate_24hrs(false_alarms, all_non_seizures):
    """
    Computes the error rate in 24 hours.
    :param false_alarms: amount of false positives
    :param all_non_seizures: summed up time delta of all non seizures
    :return: error rate in 24 hrs
    """
    days = all_non_seizures.total_seconds() / (60 * 60 * 24)
    error_rate = false_alarms / days
    return error_rate


def get_false_alarm_ns(time_,valid_times, y_true, y_pred,n):

    act_pred = np.stack([y_true,y_pred],axis=1)
    # time = pd.date_range("2018-01-01",periods=act_pred.shape[0],freq='10s')
    time = np.concatenate(list(valid_times.values()),axis=0)[:,0]
    df = pd.DataFrame(data=act_pred,index=time,columns=['actual','predict'])
    non_seizure_part = df[df['actual']==0].drop('actual',axis=1)

    total_seconds = time_.total_seconds()
    total_days = total_seconds/(60*60*24)

    def get_false_alarms(window):
        if window.size:
            if 1 in window.values:
                return 1
            else:
                return 0
        else:
            return np.nan

    ns_ns = non_seizure_part.resample(n).agg(get_false_alarms).dropna()
    print(n)
    fa_24 = np.sum(ns_ns.values)/total_days

    return fa_24


def probs_to_prediction(probs, threshold):
    pred = []
    for x in probs[:,1]:
        if x > threshold:
            pred.append(1)
        else:
            pred.append(0)
    return pred


def threshold_moving(true_label, pred_prob,save_path):
    pred_prob = pred_prob[:, 1]  # keeeping the second class(seizure)
    fpr, tpr, threshold = roc_curve(true_label, pred_prob)
    gmeans = sqrt(tpr * (1 - fpr))
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))
    best_threshold = threshold[ix]
    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='CNN')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    # plt.show()
    plt.savefig(save_path + '/roc_curve.png')
    return best_threshold



