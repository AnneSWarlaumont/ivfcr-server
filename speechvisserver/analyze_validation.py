from matplotlib import pyplot
import numpy
from scipy.signal import resample


def plotRatingsOverTime():
    numCoders = len(numpy.unique(validations[:,4]))
    for i, coder in zip(range(1, 1 + numCoders), numpy.unique(validations[:,4])):
        pyplot.subplot(2, int(numCoders / 2), i)
        rows = validations[validations[:,4] == coder]
        scores = 2 * (rows[:,3] == 'CHN') + (rows[:,3] == 'OLN')
        pyplot.plot(range(len(scores)), scores, '.')
        pyplot.plot(numpy.linspace(0, len(scores), 100), resample(scores, 100))
        pyplot.title('Coder: {}'.format(coder))
        pyplot.xlabel('Response Number')
        pyplot.yticks([0, 1, 2], ['REJ', 'OLN', 'CHN'])


def plotCoverageAndOverlap():
    alison = validations[validations[:,4] == 'Alison',0:2]
    a = numpy.vstack(numpy.logical_and(alison[:,0] == row[0], alison[:,1] == row[1]).sum() for row in numbers)
    a = numpy.minimum(a, 1)
    monica = validations[validations[:,4] == 'Monica Mendiola',0:2]
    b = numpy.vstack(numpy.logical_and(monica[:,0] == row[0], monica[:,1] == row[1]).sum() for row in numbers)
    b = numpy.minimum(b, 1)
    coverage = a + b
    values, counts = numpy.unique(coverage, return_counts=True)
    pyplot.bar(values, counts / len(coverage))

def plotRaterAgreement():
    alison = validations[validations[:,4] == 'Alison',0:2]
    a = numpy.vstack(numpy.logical_and(alison[:,0] == row[0], alison[:,1] == row[1]).sum() for row in numbers)
    #a = numpy.minimum(a, 1)
    monica = validations[validations[:,4] == 'Monica Mendiola',0:2]
    b = numpy.vstack(numpy.logical_and(monica[:,0] == row[0], monica[:,1] == row[1]).sum() for row in numbers)
    #b = numpy.minimum(b, 1)
    #coverage = a + b
    coverage = b
    overlapping = numbers[coverage.reshape((len(numbers))) >= 2, :]
    consistent = 0
    for number in overlapping:
        indices = numpy.logical_and(validations[:,0] == number[0], validations[:,1] == number[1])
        rows = validations[indices,3]
        consistent = consistent + (1 if len(numpy.unique(rows)) == 1 else 0)
    print(consistent)
    print(len(overlapping))
