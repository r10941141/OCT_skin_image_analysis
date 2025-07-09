from sklearn.metrics import confusion_matrix
import numpy as np


def confusion_matrix(Y_test,Z_test):
    """
    This function calculates and displays the confusion matrix and five commonly used evaluation metrics 
    (Accuracy, IoU, Precision, Recall, Dice). 
    The input is numpy arrays of ground truth mask and predicted mask.
    """

    ground_truth_labels = Y_test.ravel()
    predition_labels = Z_test.ravel()

    r = confusion_matrix(y_true=ground_truth_labels, y_pred=predition_labels)
    r = np.flip(r)

    acc = (r[0][0] + r[-1][-1]) / (r[0][0] + r[0][-1] + r[-1][0]+ r[-1][-1])
    iou = r[0][0] / (r[0][0] + r[0][-1] + r[-1][0])
    precision = r[0][0] / (r[0][0] + r[-1][0])
    recall = r[0][0] / (r[0][0] + r[0][-1])
    dice = 2*(precision*recall/(recall+precision))

    return acc, iou, precision, recall, dice