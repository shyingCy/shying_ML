

import numpy as np
# here x and y must be a np array
# the function return the distance of x and y using metric
# attention: x must be a shape(1,1) array, the columns are feature
# the rows are samples number and the dimensions of x and y on column
# must be the same
def get_distance(x, y, metric = 'euclidean'):

    '''
    if x != np.ndarray or y != np.ndarray:
        print 'the type of x or y is not ndarray or both are not!'
        return
    '''
    if x.reshape(1, -1).shape[1] != y.shape[1]:
        print 'features number not the same!'
        return

    if metric == "euclidean":

        difMat = np.tile(x, (y.shape[0], 1)) - y
        sqDif = difMat ** 2
        disMat = sqDif.sum(axis=1)
        return disMat
