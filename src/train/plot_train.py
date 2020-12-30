import matplotlib.pyplot as plt
import os


def get_all_metrics(history):
    metrics = set()
    for metric in history.history:
        if 'val_' in metric:
            metric = metric.replace(metric, 'val_')
        metrics.add(metric)
    return metrics


def plot_history(history, result_dir, prefix):
    """
    Plots the model training history in each epoch
    :param history: generated during model training
    :param result_dir: save the training history in this folder
    :return: None
    """
    try:
        metrics = get_all_metrics(history)
        for metric in metrics:
            loss_metric = 'val_' + metric
            if metric in history.history and loss_metric in history.history:
                plt.plot(history.history[metric], 'g.-')
                plt.plot(history.history[loss_metric], 'r.-')
                plt.title(metric)
                plt.xlabel('epoch')
                plt.ylabel(metric)
                plt.legend(['train', 'val'])
                plt.savefig(os.path.join(result_dir, prefix + '_' + metric + '.png'))
                plt.close()
    except Exception as e:
        print(e)
