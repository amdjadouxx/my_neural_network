from matplotlib import pyplot as plt
import numpy as np


def disp_loss_graph(err_logs):
    """
    Display the training statistics

    :param err_logs: list of errors
    """
    plt.plot(range(len(err_logs)), err_logs)
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

def disp_accuracy_graph(accuracy_logs):
    """
    Display the training statistics

    :param accuracy_logs: list of accuracies
    """
    if len(accuracy_logs) == 0:
        print("No accuracy logs to display")
        return
    plt.plot(range(len(accuracy_logs)), accuracy_logs)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

def disp_loss_accuracy_graph(err_logs, accuracy_logs):
    """
    Display the training statistics

    :param err_logs: list of errors
    :param accuracy_logs: list of accuracies
    """
    if len(err_logs) == 0:
        print("No loss logs to display")
        return
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(len(err_logs)), err_logs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(len(accuracy_logs)), accuracy_logs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()