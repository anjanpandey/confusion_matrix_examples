from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd 
import seaborn as sns 


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10, 7), fontsize = 14):
    """
    Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments:
    -----------
    confusion_matrix: numpy.ndarry 
        The numpy.ndarry object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used. 
    
    class_names: list
        An ordered list of class names, in the order they indec the given confusion matrix. 

    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the outputted figure,
        the second determining the vertical size. Defauls to (10,7)

    
    fontsize: int
        Font size for axes lables. Defaults to 14.

    Returns
    ---------
    matplotlib.figure.Figure 
        The resulting confusion matrix figure 

    """

    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    plt.show()



truth = ["Cancer", "Not a Cancer", "Cancer", "Cancer", "Cancer", "Not a Cancer", "Not a Cancer", "Cancer", "Not a Cancer", "Cancer"]
prediction = ["Cancer", "Cancer",  "Cancer",  "Not a Cancer", "Cancer", "Not a Cancer", "Cancer", "Not a Cancer", "Cancer", "Cancer"]

"""
Accuracy
-----------
How accuracte is the model? 
Total sample = 10
Total correct = 5

accuracy = 5/10 = 0.5

Precision
------------
    For Cancer Class
        True Positive = 4
        False Positive = 3

        Our prediction model predicted 7 items were 'Cancer.' Of which, only four (4) are correct.

        Precision = 4/7 = 0.57

    For Not a Cancer Class
        True Positive = 1
        False Positive = 2

        Our prediction model predicted 3 items were 'Not a Cancer.' Of which, only (1) is correct.

        Precision = 1/3 = 0.33

        
Recall (baseline is always the truth)
--------------
    For Cancer Class
        Total 'Cancer' in the sample = 6
        True Positive = 4

        Out of 6 'Cancer,' our prediction model predicted 4 correctly. 

        Recall = 4/6 = 0.67

    For Not a Cancer Class
        Total 'Not a Cancer' in the sample = 4
        True Positive = 1

        Out of 4 'Not a Cancer,' our prediction model predicted 1 correctly. 

        Recall = 1/4 = 0.25 


Precision = True Positive / (True Positive + False Positive)
Recall = True Positive / (True Positive + False Negative)

"""

cm = confusion_matrix(truth, prediction)
print_confusion_matrix(cm, ["Cancer", "Not a Cancer"])

print(classification_report(truth, prediction))


########
"""
F1 (Harmonic Average)

"""
# extract precision and recall from the report 
# for cancer 
f1_score_cancer = 2 * ((0.57 * 0.67)/(0.57 + 0.67))
print("F1 Score (Cancer)", f1_score_cancer)

# f1 for not a cancer
f1_score_not_a_cancer = 2 * ((0.33 * 0.25)/(0.33 + 0.25))
print("F1 Score (Not a Cancer)", f1_score_not_a_cancer)


    