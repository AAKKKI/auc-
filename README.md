# auc-
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([1,1,0,0,1,1,0])
y_scores = np.array([0.8,0.7,0.5,0.5,0.5,0.5,0.3])
print "y_true is ",y_true
print "y_scores is ",y_scores
print "AUC is",roc_auc_score(y_true, y_scores)
 
 
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print "y_true is ",y_true
print "y_scores is ",y_scores
print "AUC is ",roc_auc_score(y_true, y_scores)
