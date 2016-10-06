import numpy
import math
from matplotlib import pyplot
from speechvisserver.models import *


def plot_speaker_counts(recording):
    """Plot the number of segments in the recording for each speaker."""
    speakers = recording.lena_segments.values_list('speaker')
    speakers, counts = numpy.unique(speakers, return_counts=True)
    fig = pyplot.figure()
    pyplot.bar(numpy.arange(len(speakers)) + 0.1, counts)
    pyplot.title('Number of Vocalizations by Speaker')
    pyplot.xticks(numpy.arange(len(speakers)) + 0.5, speakers)
    pyplot.xlim(0, len(speakers))
    pyplot.xlabel('Speaker')
    pyplot.ylabel('Count')
    return fig


def plot_durations(recording, speaker):
    """Plot a time series and a histogram of segment durations, optionally filtered for a speaker."""
    records = recording.lena_segments.filter(speaker=speaker)
    values = numpy.array(records.values_list('start', 'end'))
    durations = values[:, 1] - values[:, 0]
    midpoints = values[:, 0] + durations / 2
    fig = pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.plot(midpoints, durations)
    pyplot.title('Vocalization Durations for {0}'.format('ALL' if speaker is None else speaker))
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Duration (s)')
    pyplot.subplot(2, 1, 2)
    pyplot.hist(durations, bins=numpy.logspace(0, 4, 100))
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.xlabel('Duration (s)')
    pyplot.ylabel('Count')
    return fig


def plot_intervals(recording, speaker):
    """Plot a time series and histogram of segment intervals labeled as speaker."""
    records = recording.lena_segments.filter(speaker=speaker)
    values = numpy.array(records.values_list('start', 'end'))
    intervals = values[1:, 0] - values[:-1, 1]
    midpoints = values[:-1, 0] + (values[1:, 1] - values[:-1, 0]) / 2
    fig = pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.plot(midpoints, intervals, '.')
    pyplot.title('Vocalization Intervals for {0}'.format(speaker))
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Interval (s)')
    pyplot.subplot(2, 1, 2)
    pyplot.hist(intervals, bins=numpy.logspace(0, 4, 50))
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.xlabel('Interval (s)')
    pyplot.ylabel('Count')
    return fig


def plot_volubility(recording, speaker):
    """Plot the volubility ratio (proportion of time that speaker is speaking) as a time series and histogram. This
    analysis uses one minute blocks to aggregate segments."""
    records = recording.lena_segments.filter(speaker=speaker)
    values = numpy.array(records.values_list('start', 'end'))
    minutes = math.ceil((values[-1, 1] - values[0, 0]) / 60)
    volubility = numpy.zeros(minutes)
    for m in range(minutes):
        start_minute = 60 * m
        end_minute = 60 * m + 60
        for i in range(len(values)):
            volubility[m] += max(min(end_minute, values[i, 1]) - max(start_minute, values[i, 0]), 0)
    volubility /= 60
    fig = pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.plot(60 * numpy.arange(minutes), volubility)
    pyplot.title('Volubility for {0}'.format(speaker))
    pyplot.xlabel('Time (min)')
    pyplot.ylabel('Vocalized Seconds / Minute')
    pyplot.subplot(2, 1, 2)
    pyplot.hist(volubility, bins=50)
    pyplot.yscale('log')
    pyplot.xlabel('Volubility')
    pyplot.ylabel('Count')
    return fig
