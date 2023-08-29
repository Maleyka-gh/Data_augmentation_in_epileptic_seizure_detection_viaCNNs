from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter


class Plot:
    def __init__(self, x, y, real_seizures, save_path, pat):
        self.pat = pat
        self.real_seizures = self.times_of_ground_truth(real_seizures, x)
        self.threshold = pd.Timedelta('00:30:00')
        self.x, self.y = self.split_data(x, y)
        self.save_plots(save_path)

    @staticmethod
    def times_of_ground_truth(gt, times):
        """
        This method computes a list of start and end times (dictionary) of the real seizures in gt.
        :param gt: Series with 0 or 1 ; 1 for seizure, 0 for no seizure
        :param times: Timestamps for every entry in gt
        :return: list of dictionary with two keys; start & end of a seizure
        """
        boxes = list()
        changes = gt[gt.diff() != 0].index.tolist()  # changes between 0 and 1
        changes.append(gt.index[-1])  # append last element
        for c in range(0, len(changes)-1):
            # only consider changes from 1 to 0 and not from 0 to 1
            if gt[changes[c]] == 1 and gt[changes[c+1]] == 0 and gt[changes[c+1]-1] == 1:
                start = times[changes[c]]
                end = times[changes[c+1] - 1]
                boxes.append({'start': start, 'end': end})
        return boxes

    def split_data(self, x, y):
        """
        This method splits x, y depending on the difference between two neighbored timestamps in x. If the difference is
        larger than threshold, x and y are split and stored in two lists. These two lists are returned at the end.
        The idea behind this is that one plot does not have too many gaps.
        Instead, it is better to plot two different plots later.
        :param x: Series of timestamps; corresponding to y
        :param y: Series of values; corresponding to the timestamps of x
        :return: Two lists of Series. One with timestamps, other with values.
        """
        # where is the difference between two neighbored timestamps larger than threshold
        mask = x.diff() > self.threshold
        index = mask[mask].index.to_list()

        # split data:
        x_result = list()
        y_result = list()
        for i in range(1, len(index)-1):
            x_result.append(x[index[i-1]:index[i]])
            y_result.append(y[index[i-1]:index[i]])
        if not x_result and not y_result:
            return [x], [y]
        return x_result, y_result

    def save_plots(self, save_path):
        """
        Plots every plot in list of x or y. Saves the plot.
        """
        # plot every subplot and set x-axis
        for i in range(0, len(self.x)):  # iterate for every plot in the list x
            # plot and sets limits of plt
            plt.plot(self.x[i], self.y[i], marker=".")
            plt.axhline(0.5, color='gray')
            plt.ylim((-0.01, 1.01))
            # turn red if value above 0.5
            top = np.ma.masked_where(self.y[i] < 0.5, self.y[i])
            plt.plot(self.x[i], top, 'r', linewidth=2, marker=".")
            # axvspan the real seizures (ground_truth)
            num_of_seizures = 0
            for seizure in self.real_seizures:
                # seizure is in plot_window
                if self.x[i].iloc[0] < seizure['start'] and self.x[i].iloc[-1] > seizure['end']:
                    plt.axvspan(seizure['start'], seizure['end'], alpha=0.5, color='green')
                    num_of_seizures = num_of_seizures + 1

                    seizure_center = mdates.date2num(((seizure['end'] - seizure['start']) / 2) + seizure['start'])
                    plt.annotate('Seizure',
                                 xy=(seizure_center, 1),
                                 xytext=(seizure_center, 1.05),
                                 arrowprops=dict(arrowstyle="->"),
                                 horizontalalignment="center",
                                 verticalalignment="bottom"
                                 )

            # save plot
            duration = self.x[i].iloc[-1] - self.x[i].iloc[0]
            time = str(duration.seconds // 3600) + "-" + str(duration.seconds // 60 % 60)
            num_of_dots = self.x[i].shape[0]
            name = self.pat + "_" + str(num_of_seizures) + "_" + str(num_of_dots) + "_" + time

            """Determines design and format of the x ticks."""
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(FuncFormatter(self.x_axis_formatter))

            plt.tight_layout()
            plt.savefig(save_path + "/" + name)
            plt.clf()
            plt.close()

    @staticmethod
    def x_axis_formatter(a, _):
        """Formatter for the x_axis. It is used for time_stamps"""
        date = mdates.num2date(a)
        year = str(date.year)[2:]
        month = str(date.month)
        day = str(date.day)
        hour = str(date.hour)
        minute = str(date.minute)
        second = str(date.second)
        millisecond = str(date.microsecond)[:2]
        tick_format = "%s.%s.%s %s:%s:%s.%s" % (day, month, year, hour, minute, second, millisecond)
        return tick_format
