from django.db import models
from django.db.models import Max
from xml.etree import ElementTree
import numpy
from scipy.io import wavfile
from scipy.signal import butter, lfilter, correlate, savgol_filter
from python_speech_features import logfbank, mfcc
import os.path


class Child(models.Model):
    id = models.CharField(max_length=50, primary_key=True, null=False)
    dob = models.DateField(null=False)
    gender = models.CharField(max_length=1, blank=False, null=False)

    def __str__(self):
        return '{0} {1} {2}'.format(self.id, self.dob, self.gender)


class Recording(models.Model):
    @staticmethod
    def import_recording(id, directory, writeFiles=True):
        """Create a new recording in the database, parse the associated ITS files, create
            rows for each segment, and split the original wave file into segments in subdirectories"""
        recording = Recording(id=id, directory=directory)
        recording.read_child_from_its()
        recording.save()
        recording.read_audio()
        numbers, starts, ends, speakers, categories = recording.split_its()
        recording.create_segments(numbers, starts, ends, speakers, writeFiles)
        recording.create_annotations(numbers, speakers, categories, 'LENA', 'SPLIT_ITS')
        return recording

    child = models.ForeignKey(Child, on_delete=models.CASCADE, null=False)
    id = models.CharField(max_length=50, primary_key=True)
    directory = models.CharField(max_length=200, null=True)

    def read_child_from_its(self):
        tree = ElementTree.parse('{0}/{1}.its'.format(self.directory, self.id))
        root = tree.getroot()
        childElement = next(root.iter('Child'))
        child = Child.objects.filter(id=childElement.get('id'))
        if child.exists():
            child = child.get()
        else:
            child = Child(id=childElement.get('id'), dob=childElement.get('DOB'),
                          gender=childElement.get('Gender'))
            child.save()
        self.child = child

    def split_its(self):
        tree = ElementTree.parse('{0}/{1}.its'.format(self.directory, self.id))
        root = tree.getroot()
        starts = []
        ends = []
        speakers = []
        categories = []
        for segment in root.iter('Segment'):
            starts.append(parse_time(segment.attrib['startTime']))
            ends.append(parse_time(segment.attrib['endTime']))
            speakers.append(segment.attrib['spkr'])
            categories.append(parse_category(segment))
        return range(len(starts)), starts, ends, speakers, categories

    @property
    def filename(self):
        return '{0}.wav'.format(self.id)
    
    @property
    def filepath(self):
        return os.path.join(self.directory, self.filename)

    @property
    def mp3_filename(self):
        return '{1}.mp3'.format(self.directory, self.id)

    def read_audio(self):
        self.samplerate, self.signal = wavfile.read(self.filename)

    def create_segments(self, numbers, starts, ends, speakers, writeFiles):
        if writeFiles:
            segments_directory = os.path.join(self.directory, self.id)
            if not os.path.exists(segments_directory):
                os.makedirs(segments_directory)
            for speaker in set(speakers):
                speaker_directory = os.path.join(segments_directory, speaker)
                if not os.path.exists(speaker_directory):
                    os.makedirs(speaker_directory)
        segments = []
        for number, start, end, speaker in zip(numbers, starts, ends, speakers):
            print('Inserting Segment {0} of {1}'.format(number, len(numbers)))
            if writeFiles:
                segment_audio = self.signal[int(start * self.samplerate):int(end * self.samplerate)]
                filepath = os.path.join(segments_directory, speaker, '{0}.wav'.format(number))
                wavfile.write(filepath, self.samplerate, segment_audio)
            segment = Segment(recording=self, number=number, start=start, end=end)
            segments.append(segment)
        Segment.objects.bulk_create(segments)

    def create_annotations(self, numbers, speakers, categories, coder, method):
        annotations = []
        segments = Segment.objects.filter(recording=self).order_by('number')
        for segment, speaker, category in zip(segments, speakers, categories):
            print('Inserting Annotation {0} of {1}'.format(segment.number, len(segments)))
            annotation = Annotation(segment=segment, speaker=speaker, category=category,
                                    coder=coder, method=method)
            annotations.append(annotation)
        Annotation.objects.bulk_create(annotations)

    def frequency_banks(self, winlen=0.05, winstep=0.025, blockSize=600):
        t = numpy.zeros(0)
        fbanks = numpy.zeros((0, 26))
        start = 0
        while start < len(self.signal):
            end = start + blockSize * self.samplerate
            end = end if end < len(self.signal) else len(self.signal)
            block = self.signal[start:end]
            fbank = logfbank(block, self.samplerate, winlen=winlen, winstep=winstep)
            t = numpy.concatenate((t, numpy.linspace(start / self.samplerate, end / self.samplerate - winstep, len(fbank))))
            fbanks = numpy.concatenate((fbanks, fbank))
            start = end
        return t, fbanks

    @property
    def lena_segments(self):
        return self.segment.filter(annotation__coder='LENA').annotate(
            speaker=Max('annotation__speaker'), category=Max('annotation__category'))

    def __str__(self):
        return '{0} {1}'.format(self.directory, self.id)


def parse_time(formatted, default=0.0):
    if formatted.startswith('PT') and formatted.endswith('S'):
        return float(formatted[2:-1])
    elif formatted.startswith('P') and formatted.endswith('S'):
        return float(formatted[1:-1])
    else:
        return default


def parse_category(segment):
    category = 'OTHER'
    if 'childUttLen' in segment.attrib and parse_time(segment.attrib['childUttLen']) > 0:
        category = 'CHILD_SPEECH'
    elif parse_time(segment.attrib.get('childCryVfxLen', '')) > 0:
        category = 'CHILD_CRY'
    elif (parse_time(segment.attrib.get('femaleAdultUttLen', '')) > 0
          or parse_time(segment.attrib.get('maleAdultUttLen', '')) > 0):
        category = 'ADULT_SPEECH'
    elif (parse_time(segment.attrib.get('femaleAdultNonSpeechLen', '')) > 0
          or parse_time(segment.attrib.get('maleAdultNonSpeechLen', '')) > 0):
        category = 'ADULT_NONSPEECH'
    return category


class Segment(models.Model):
    @staticmethod
    def find_peak(signal):
        diff = numpy.diff(signal)
        extrema = numpy.where(diff[:-1] * diff[1:] <= 0)[0]
        index = numpy.arange(len(signal))
        return extrema[numpy.argmax(signal[extrema])]

    recording = models.ForeignKey(Recording, on_delete=models.CASCADE, related_name='segment')
    number = models.IntegerField()
    start = models.FloatField()
    end = models.FloatField()

    class Meta:
        unique_together = ('recording', 'number')

    @property
    def duration(self):
        return self.end - self.start

    @property
    def lena_speaker(self):
        return self.annotation.get(coder='LENA').speaker

    @property
    def lena_category(self):
        return self.annotation.get(coder='LENA').category

    @property
    def full_path(self):
        return os.path.join(self.recording.directory, self.static_path)
    
    @property
    def static_path(self):
        return os.path.join(self.recording.id, self.lena_speaker,
                            '{}.wav'.format(self.number))
    
    def read_audio(self):
        self.samplerate, self.signal = wavfile.read(self.full_path)

    def acoustic_features(self):
        window_size = 1024
        step_size = 512
        _, power = self.power(window_size, step_size)
        _, pitches = self.pitch(window_size, step_size)
        return {
            'peak_amplitude': power.max(),
            'mean_amplitude': power.mean(),
            'pitch': numpy.mean(pitches)
        }

    def power(self, window_size, step_size):
        """Return a time series corresponding to the acoustic power of the segment waveform within
        each window of the given size, offset by the specified step."""
        steps = numpy.arange(0, self.signal.size - window_size + 1, step_size)
        power = numpy.zeros(len(steps))
        for i, step in zip(range(len(steps)), steps):
            windowed_signal = self.signal[step:step + window_size].astype(int)
            rms = numpy.sqrt(numpy.mean(numpy.square(windowed_signal)))
            power[i] = 10 * numpy.log10(rms)
        return steps / float(self.samplerate), power

    def filter(self, lowFreq, highFreq, order):
        nyquist = 0.5 * float(self.samplerate)
        low = lowFreq / nyquist
        high = highFreq / nyquist
        b, a = butter(order, [low, high], btype='band')
        self.signal = lfilter(b, a, self.signal)

    def pitch(self, window_size, step_size, lowCut=50, highCut=600, threshold=40):
        steps = numpy.arange(0, self.signal.size - window_size + 1, step_size)
        windows = []
        pitches = []
        for i, step in zip(range(len(steps)), steps):
            windowed_signal = self.signal[step:step + window_size]
            nyquist = 0.5 * float(self.samplerate)
            low = lowCut / nyquist
            high = highCut / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = lfilter(b, a, windowed_signal)
            pitch = self.autocorr_pitch(filtered_signal, lowCut, highCut)
            frequencies = numpy.fft.rfftfreq(len(filtered_signal), 1 / self.samplerate)
            spectrum = 10 * numpy.log10(numpy.abs(numpy.fft.rfft(windowed_signal).real))
            f = numpy.argmin(numpy.abs(frequencies - pitch))
            power = spectrum[f - 2:f + 3].mean()
            if power >= threshold:
                windows.append(step + window_size / 2)
                pitches.append(pitch)
        return windows, pitches

    def autocorr_pitch(self, signal, low, high):
        autocorr = correlate(signal, signal)
        autocorr = autocorr[len(autocorr) / 2:]
        autocorr = autocorr[int(self.samplerate / high):int(self.samplerate / low)]
        return self.samplerate / (Segment.find_peak(autocorr) + int(self.samplerate / high))

    def cepstral_pitch(self, signal):
        spectrum = numpy.fft.rfft(signal).real
        cepstrum = numpy.fft.rfft(numpy.log(numpy.square(spectrum) + 0.0001)).real
        for peak in range(len(cepstrum) - 2, 1, -1):
            power = cepstrum[peak]
            if power > cepstrum[peak - 1] and power > cepstrum[peak + 1]:
                return peak, power, cepstrum

    def power_spectrum(self, window_size, step_size):
        """Return a time series of power spectra, where each spectrum corresponds to the acoustic power within a window
        of the given size and offset from the previous window by the given step. The spectra represent the power over
        frequency within the window, where the associated frequencies are specified by the first return value."""
        frequencies = numpy.fft.rfftfreq(window_size, d=1. / self.samplerate)
        spectrum = numpy.ndarray((0, frequencies.size))
        window = numpy.hanning(window_size)
        for i in range(window_size, self.signal.size, step_size):
            x = window * self.signal[i - window_size:i]
            spectrum = numpy.vstack((spectrum, abs(numpy.fft.rfft(x).real)))
        return frequencies, spectrum

    def formants(self, window_size, step_size, count):
        """Return a time series of formant frequencies identified from the power spectra. Count is the number of
        formants to identify."""
        frequencies, spectrum = self.power_spectrum(window_size, step_size)
        formants = numpy.full((spectrum.shape[0], count), float('NaN'))
        peakwidth = int(50 * (frequencies[-1] / len(frequencies)))
        for step in range(spectrum.shape[0]):
            # mean_spectrum = numpy.mean(spectrum, axis=0)
            mean_spectrum = spectrum[step]
            peaks = []
            for x, y in zip(range(len(mean_spectrum)), mean_spectrum):
                if y == max(mean_spectrum[max(0, x - peakwidth):min(len(mean_spectrum), x + peakwidth)]):
                    peaks.append((x, y))
            peaks = numpy.array(peaks)
            peaks = peaks[numpy.argsort(peaks[:, 1]), :]
            f = numpy.sort(frequencies[peaks[-count:, 0].astype(int)])
            f = numpy.pad(f, (0, count - f.size), 'constant', constant_values=float('NaN'))
            formants[step, :] = f
        return formants

    def frequencybank(self, window_size=0.05, step_size=0.05, truncate=0.6):
        """Returns a Mel-frequency filter bank for this segment."""
        x = self.signal
        if truncate:
            x = x[:int(truncate * self.sample_rate)]
        return logfbank(x, self.sample_rate, winlen=window_size, winstep=step_size)

    def mfccs(self):
        """Returns the Mel-Frequency Cepstral Coefficients for this segment."""
        return mfcc(self.signal[:int(0.6 * self.samplerate)], self.samplerate, winlen=0.05,
                    winstep=0.05, numcep=40, nfilt=80)

    def __str__(self):
        return '{0}: {1}\t{2}s - {3}s'.format(self.id, self.recording.id, self.start, self.end)


class Annotation(models.Model):
    segment = models.ForeignKey(Segment, on_delete=models.CASCADE, related_name='annotation')
    speaker = models.CharField(max_length=50)
    category = models.CharField(max_length=50, blank=True, null=True)
    transcription = models.CharField(max_length=400, blank=True, null=True)
    sensitive = models.BooleanField(default=True)
    coder = models.CharField(max_length=50)
    method = models.CharField(max_length=50, blank=True, null=True)
    annotated = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return '{0}: {1}\t{2}\t{3}\t{4}\t{5}\t({6}, {7}, {8})'.format(self.id, self.segment.id,
            self.speaker, self.category, self.transcription, self.sensitive,
            self.coder, self.method, self.annotated)


class AudioFeature(models.Model):
    @staticmethod
    def save_feature(recording, feature, t, data, labels=None):
        feature = AudioFeature(recording=recording, feature=feature)
        feature.save()
        if labels is not None:
            numpy.savez(feature.filename, t=t, data=data, labels=labels)
        else:
            numpy.savez(feature.filename, t=t, data=data)
        feature.t = t
        feature.data = data
        return feature

    recording = models.ForeignKey(Recording, on_delete=models.CASCADE, related_name='feature')
    feature = models.CharField(max_length=50, null=False)
    generated = models.DateTimeField(auto_now=True, auto_created=True)

    @property
    def filename(self):
        return os.path.join(self.recording.directory, '{}_{}.npz'.format(self.recording.id, self.feature))

    def load_data(self):
        file = numpy.load(self.filename)
        self.t = file['t']
        self.data = file['data']
        if 'labels' in file:
            self.hasLabels = True
            self.labels = file['labels']
        else:
            self.hasLabels = False

    def filter_data(self, speaker):
        speaker_segments = self.recording.segment.filter(annotation__coder='LENA', annotation__speaker=speaker)
        newT = []
        newData = []
        i = 0
        for segment in speaker_segments:
            while i < len(self.t) and self.t[i] < segment.start:
                i += 1
            while i < len(self.t) and self.t[i] <= segment.end:
                newT.append(self.t[i])
                newData.append(self.data[i,:])
                i += 1
            print('Filtered Segment {}'.format(segment.number))
        return numpy.array(newT), numpy.array(newData)

    def generate_labels(self, speakers):
        segments = self.recording.segment.filter(annotation__coder='LENA', annotation__speaker__in=speakers)
        segments = segments.order_by('start')
        labels = []
        i = 0
        for segment in segments:
            while i < len(self.t) and self.t[i] < segment.start:
                labels.append('')
                i += 1
            while i < len(self.t) and self.t[i] <= segment.end:
                labels.append(segment.annotation.filter(coder='LENA').get().speaker)
                i += 1
        self.labels = labels
        numpy.savez(self.filename, t=self.t, data=self.data, labels=self.labels)

    def __str__(self):
        return '{0} Feature: {1} {2}'.format(self.recording.id, self.feature, self.generated)
