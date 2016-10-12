from matplotlib import pyplot
from python_speech_features import mfcc, logfbank
import numpy


def plot_segment(segment, window_size, step_size):
    """Plots the waveform, power over time, spectrogram with formants, and power over frequency of a Segment."""
    fig = pyplot.figure()
    pyplot.suptitle('Vocalization {}'.format(segment.number))
    pyplot.subplot(2, 2, 1)
    pyplot.plot(numpy.linspace(0, float(segment.duration), len(segment.signal)), segment.signal)
    pyplot.xlim(0, segment.duration)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Sound Pressure')
    pyplot.subplot(2, 2, 2)
    steps, power = segment.power(window_size, step_size)
    pyplot.plot(steps, power)
    pyplot.xlim(0, segment.duration)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Power (dB)')
    pyplot.subplot(2, 2, 3)
    pyplot.specgram(segment.signal, NFFT=window_size, Fs=segment.samplerate, noverlap=window_size - step_size)
    windows, pitches = segment.pitch(window_size, step_size)
    pyplot.plot(numpy.array(windows) / segment.samplerate, pitches, 'or')
    pyplot.xlim(0, segment.duration)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Frequency (Hz)')
    pyplot.subplot(2, 2, 4)
    frequencies, spectrum = segment.power_spectrum(window_size, step_size)
    pyplot.plot(frequencies / 1000, 10 * numpy.log10(numpy.mean(spectrum, axis=0)))
    pyplot.xlabel('Frequency (kHz)')
    pyplot.ylabel('Power (dB)')
    return fig

