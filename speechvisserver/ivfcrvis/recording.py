import numpy
import math
from matplotlib import pyplot


def plot_speaker_counts(recording):
    """Plot the number of segments in the recording for each speaker."""
    speakers, counts = numpy.unique(recording.speakers, return_counts=True)
    fig = pyplot.figure()
    pyplot.bar(numpy.arange(len(speakers)) + 0.1, counts)
    pyplot.title('Number of Vocalizations by Speaker')
    pyplot.xticks(numpy.arange(len(speakers)) + 0.5, speakers)
    pyplot.xlim(0, len(speakers))
    pyplot.xlabel('Speaker')
    pyplot.ylabel('Count')
    return fig


def plot_durations(recording, speaker=None):
    """Plot a time series and a histogram of segment durations, optionally filtered for a speaker."""
    if speaker is None:
        starts = recording.starts
        ends = recording.ends
    else:
        i, starts, ends = recording.filter_speaker(speaker)
    durations = ends - starts
    fig = pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.plot(starts + durations / 2, durations)
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
    i, starts, ends = recording.filter_speaker(speaker)
    intervals = starts[1:] - ends[:-1]
    fig = pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.plot(starts[1:], intervals)
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
    minutes = math.ceil((recording.ends[-1] - recording.starts[0]) / 60)
    volubility = numpy.zeros(minutes)
    i, starts, ends = recording.filter_speaker(speaker)
    for m in range(minutes):
        start_minute = 60 * m
        end_minute = 60 * m + 60
        for start, end in zip(starts, ends):
            volubility[m] += max(min(end_minute, end) - max(start_minute, start), 0)
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
