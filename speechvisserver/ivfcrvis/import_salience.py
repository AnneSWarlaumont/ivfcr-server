import os
import numpy


def get_segment_list(directory):
    files = os.listdir(directory)
    numbers = []
    speakers = []
    for file in files:
        fields = file.split('_')
        numbers.append(fields[4])
        speakers.append(fields[5])
    files = numpy.array(files)
    numbers = numpy.array(numbers, dtype='int')
    speakers = numpy.array(speakers)
    index = numbers.argsort()
    return files[index], numbers[index], speakers[index]


def read_segment_data(directory, files, starts, stops):
    recording_data = numpy.zeros((0, 304))
    for file, start, stop in zip(files, starts, stops):
        segment_data = numpy.loadtxt(os.path.join(directory, file), delimiter=',')
        times = numpy.linspace(start, stop, segment_data.shape[1])
        segment_data = numpy.concatenate((times, segment_data.transpose()), axis=1)
        recording_data = numpy.concatenate((recording_data, segment_data), axis=0)
        print(start)
    return recording_data
