import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score
y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
# score = precision_recall_fscore_support(y_true, y_pred, average=None, labels=['pig', 'dog', 'cat'])
score = recall_score(y_true, y_pred, average=None, labels=['cat'])
print(score)
