from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(true, pred, title, tick_labels, save=False):
    """shows plot of confusion matrix"""
    plt.clf()
    cm = confusion_matrix(true, pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix' + title)
    ax.xaxis.set_ticklabels(tick_labels)
    ax.yaxis.set_ticklabels(tick_labels)
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    if save:
        plt.savefig('OUTPUT_' + title + '.png')
    plt.show()
    plt.close()