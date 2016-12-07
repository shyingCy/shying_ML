from numpy import *


### weak classifier ###
def loadSimpData():
    dataMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


### decision stump ###
def stumpClassify(dataMatrix, dimen, threshVal, threshInq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshInq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    N, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((N, 1)))
    minError = inf
    for ii in range(n):
        rangeMin = dataMatrix[:, ii].min()
        rangeMax = dataMatrix[:, ii].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for jj in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(jj)*stepSize
                predictedVals = stumpClassify(dataMatrix, ii, threshVal, inequal)
                errArr = mat(ones((N, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = ii
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


### AdaBoost training ###
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    N = shape(dataArr)[0]
    D = mat(ones((N, 1))/N)
    aggClassEst = mat(zeros((N, 1)))
    for ii in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)

        alpha = float(0.5 * log((1.0-error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((N,1)))
        errorRate = aggErrors.sum() / N
        if errorRate == 0.0:
            break
    return weakClassArr


### AdaBoost testing ###
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    N = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((N, 1)))
    for ii in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[ii]['dim'], classifierArr[ii]['thresh'], classifierArr[ii]['ineq'])
        aggClassEst += classifierArr[ii]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)


dataM, label = loadSimpData()
wkCA = adaBoostTrainDS(dataM, label)
print adaClassify([2, 1], wkCA)

