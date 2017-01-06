from django.http import HttpResponse
from django.shortcuts import render
from speechvisserver.ivfcrvis.recording import *
from sklearn.decomposition import PCA
import json
import random
import csv
import decimal
import scipy.io
from io import BytesIO
import seaborn


def index(request):
    return render(request, 'index.html', {})


def speaker_validation(request):
    coder = request.GET.get('coder', '')
    submit = request.GET.get('submit', '')
    error = ''
    if submit:
        if not coder:
            error = 'Your name is required in the coder field.'
        else:
            annotation_id = request.GET.get('annotation_id')
            annotation = Annotation.objects.get(id=annotation_id)
            segment = annotation.segment
            sensitive = request.GET.get('sensitive', 'false')
            validation = Annotation(segment=segment, sensitive=sensitive,
                coder=coder, method='SPEAKER_VALIDATION')
            validation.speaker = request.GET.get('speaker')
            validation.save()
    records = Annotation.objects.filter(coder='LENA', speaker='CHN').exclude(segment__annotation__coder=coder)
    annotation = records[random.randrange(records.count())]
    context = {
        'coder': coder,
        'recording_id': annotation.segment.recording.id,
        'segment_number': annotation.segment.number,
        'filename': segment.static_path,
        'speaker': annotation.speaker,
        'speaker_descriptive': getDescriptiveName(annotation.speaker),
        'annotation_id': annotation.id,
        'error': error
    }
    return render(request, 'speaker_validation.html', context)


def vocal_categorization(request):
    coder = request.GET.get('coder', '')
    submit = request.GET.get('submit', '')
    error = ''
    if submit:
        if not coder:
            error = 'Your name is required in the coder field.'
        else:
            segment_id = request.GET.get('segment_id')
            segment = Segment.objects.get(id=segment_id)
            annotation = Annotation(segment=segment, coder=coder, method='VOCAL_CATEGORIZATION')
            annotation.category = request.GET.get('category')
            annotation.save()
    records = Segment.objects.filter(annotation__speaker='CHN').exclude(annotation__method='VOCAL_CATEGORIZATION')
    segment = records[random.randrange(records.count())]
    context = {
        'coder': coder,
        'segment': segment,
        'filename': segment.static_path,
        'error': error
    }
    return render(request, 'vocal_categorization.html', context)


def speaker_identification(request):
    coder = request.GET.get('coder', '')
    submit = request.GET.get('submit', '')
    error = ''
    speakers_list = ['CHN', 'CXN', 'FAN', 'MAN']
    descriptions = ['Primary Child', 'Other Child', 'Female Adult', 'Male Adult']
    recording = Recording.objects.all()[0] # IVFCR340 6 mos
    segment = None
    if submit:
        print(submit.lower())
        segment_id = request.GET.get('segment_id')
        segment = Segment.objects.get(id=segment_id)
        if submit.lower() == 'previous':
            records = Segment.objects.filter(recording=recording,
                                             annotation__speaker__in=speakers_list)
            records = records.filter(number__lt=segment.number).order_by('-number')
            segment = records[0]
        else:
            if submit.lower() == 'save':
                if not coder:
                    error = 'Your name is required in the coder field.'
                else:
                    annotation = Annotation(segment=segment, coder=coder, method='SPEAKER_IDENTIFICATION')
                    speakers = request.GET.getlist('speaker')
                    if len(speakers) == 0:
                        annotation.speaker = 'SIL'
                    elif len(speakers) == 1:
                        annotation.speaker = speakers[0]
                    else:
                        annotation.speaker = ','.join(speakers)
                    annotation.sensitive = True if request.GET.get('sensitive', False) else False
                    annotation.save()
            records = Segment.objects.filter(recording=recording,
                                             annotation__speaker__in=speakers_list)
            records = records.filter(number__gt=segment.number).order_by('number')
            segment = records[0]
    else:
        records = Segment.objects.filter(recording=recording,
                                         annotation__speaker__in=speakers_list)
        records = records.exclude(annotation__method='SPEAKER_IDENTIFICATION')
        records = records.order_by('recording__id', 'number')
        segment = records[0]
    if error:
        print('Error: {}'.format(error))
    speakers_list.append('REJ')
    descriptions.append('Noise, Television, Unknown')
    all_rows = recording.segment.filter(annotation__coder='LENA', annotation__speaker__in=speakers_list)
    percent_complete = int(100 * all_rows.filter(annotation__coder=coder, annotation__method='SPEAKER_IDENTIFICATION').count() / all_rows.count())
    context = {
        'speakers': list(zip(speakers_list, descriptions)),
        'coder': coder,
        'segment': segment,
        'segment_file': segment.static_path,
        'percent_complete': percent_complete,
        'error': error
    }
    return render(request, 'speaker_identification.html', context)


def data_manager(request):
    submit = request.GET.get('submit', '')
    error = ''
    if submit == 'save_feature':
        id = request.GET.get('feature_recording_id_option')
        feature = request.GET.get('feature').strip()
        filename = request.GET.get('save_feature_file')
        print('test')
        if not id:
            error = 'Invalid Recording Id'
        elif not feature:
            error = 'Invalid Feature'
        else:
            recording = Recording.objects.get(id=id)
            data = numpy.loadtxt(filename)
            feature = AudioFeature.save_feature(recording, feature, data)
            print(feature)
    if submit == 'import':
        id = request.GET.get('add_recording_id', '').strip()
        directory = request.GET.get('add_recording_directory', '')
        if not id:
            error = 'Invalid Recording Id'
        else:
            recording = Recording(id=id, directory=directory)
            recording.save()
    export_columns = ['number', 'speaker', 'start', 'end']
    acoustic_columns = ['peak_amplitude', 'mean_amplitude', 'pitch']
    export_id = request.GET.get('export', '')
    if Recording.objects.filter(id=export_id).exists():
        selected_columns = []
        for column in export_columns:
            if request.GET.get('export_{}'.format(column), False):
                selected_columns.append(column)
        segments = Segment.objects.filter(recording__id=export_id, annotation__coder='LENA').annotate(speaker=Max('annotation__speaker'))
        export_speakers = request.GET.getlist('export_speakers')
        if len(export_speakers) > 0:
            print('speakers: {}'.format(export_speakers))
            segments = segments.filter(speaker__in=export_speakers)
        data = segments.values_list(*selected_columns)
        export_format = request.GET.get('export_format', 'CSV')
        if export_format == 'CSV':
            return export_csv(data, selected_columns)
        else:
            arrays = values_to_numpy_arrays(data, selected_columns)
            selected_acoustic_columns = []
            for column in acoustic_columns:
                selected_acoustic_columns.append(request.GET.get('export_{}'.format(column), False))
                if selected_acoustic_columns[-1]:
                    arrays[column] = []
            for segment in segments:
                segment.read_audio()
                print('Getting acoustic features for segment {}'.format(segment.number))
                acoustic_features = segment.acoustic_features()
                for column, export in zip(acoustic_columns, selected_acoustic_columns):
                    if export:
                        arrays[column].append(acoustic_features[column])
            if export_format == 'NPZ':
                return export_npz(arrays)
            elif export_format == 'MAT':
                return export_mat(arrays)
            else:
                error = 'Invalid Export Format!'
    context = {
        'export_speakers': ['CHN', 'CXN', 'FAN', 'MAN'],
        'export_formats': ['CSV', 'NPZ', 'MAT'],
        'export_columns': export_columns,
        'acoustic_columns': acoustic_columns,
        'recordings': Recording.objects.all(),
        'error': error
    }
    return render(request, 'data_manager.html', context)


def export_csv(data, columns):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="export.csv"'
    writer = csv.writer(response)
    writer.writerow(columns)
    for row in data:
        writer.writerow(row)
    return response


def values_to_numpy_arrays(data, columns):
    arrays = {}
    for column, index in zip(columns, range(len(columns))):
        dtype = type(data[0][index])
        if dtype == decimal.Decimal:
            dtype = float
        arrays[column] = numpy.array([value[index] for value in data], dtype=dtype)
    return arrays


def export_npz(arrays):
    response = HttpResponse(content_type='application/octet-stream')
    response['Content-Disposition'] = 'attachment; filename="export.npz"'
    buffer = BytesIO()
    numpy.savez(buffer, **arrays)
    npzFile = buffer.getvalue()
    buffer.close()
    response.write(npzFile)
    return response


def export_mat(arrays):
    response = HttpResponse(content_type='application/octet-stream')
    response['Content-Disposition'] = 'attachment; filename="export.mat"'
    buffer = BytesIO()
    scipy.io.savemat(buffer, arrays)
    matFile = buffer.getvalue()
    buffer.close()
    response.write(matFile)
    return response


def data_visualizer(request):
    context = {
        "recordings": Recording.objects.all()
    }
    return render(request, 'data_visualizer.html', context)


def visualize_feature(request):
    recording_id = request.GET.get('recording_id')
    feature = request.GET.get('feature')
    method = request.GET.get('method', 'none').lower()
    sample = request.GET.get('sample', False)
    start = float(request.GET.get('start', 0))
    limit = int(request.GET.get('limit'))
    response = {}
    if recording_id and feature:
        record = AudioFeature.objects.get(recording__id=recording_id, feature=feature)
        record.load_data()
        if sample:
            index = numpy.floor(numpy.linspace(0, len(record.t), limit, endpoint=False)).astype('int');
            if method == 'none':
                response['t'] = record.t[index].tolist()
                response['x'] = record.data[index,0].tolist()
                response['y'] = record.data[index,1].tolist()
            elif method == 'pca':
                pca = PCA(n_components=2)
                v = pca.fit_transform(record.data)
                response['t'] = record.t[index].tolist()
                response['x'] = v[index,0].tolist()
                response['y'] = v[index,1].tolist()
            elif method == "pca ds":
                pca = PCA(n_components=2)
                v = pca.fit_transform(record.data)
                response['t'] = record.t[index].tolist()
                response['x'] = v[index, 0].tolist()
                response['y'] = v[index, 1].tolist()
        else:
            i = max(numpy.argmin(numpy.abs(record.t - start)) - 100, 0)
            response['t'] = record.t[i:i + limit].tolist()
            if method == 'none':
                response['x'] = record.data[i:i + limit,0].tolist()
                response['y'] = record.data[i:i + limit,1].tolist()
            elif method == 'pca':
                pca = PCA(n_components=2)
                v = pca.fit_transform(record.data)
                response['x'] = v[i:i + limit,0].tolist()
                response['y'] = v[i:i + limit,1].tolist()
            elif method == "pca ds":
                pca = PCA(n_components=2)
                v = pca.fit_transform(record.data)
                response['t'] = record.t[i:i + 10 * limit].reshape((limit, 10)).mean(axis=1).tolist()
                response['x'] = v[i:i + 10 * limit, 0].reshape((limit, 10)).mean(axis=1).tolist()
                response['y'] = v[i:i + 10 * limit, 1].reshape((limit, 10)).mean(axis=1).tolist()
    return HttpResponse(json.dumps(response), content_type='application/json')


def plot(request):
    types = ['speaker_counts', 'durations', 'intervals', 'volubility']
    type = request.GET.get('plot_type', types[0])
    speakers = ['CHN', 'FAN', 'MAN', 'CXN']
    speaker = request.GET.get('speaker', speakers[0])
    id = request.GET.get('recording_id', Recording.objects.all()[0].id)
    recording = Recording.objects.get(id=id)
    if type == types[0]:
        fig = plot_speaker_counts(recording)
    elif type == types[1]:
        fig = plot_durations(recording, speaker)
    elif type == types[2]:
        fig = plot_intervals(recording, speaker)
    elif type == types[3]:
        fig = plot_volubility(recording, speaker)
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response


def getDescriptiveName(speaker):
    descriptions = {
        'CHN': 'Child',
        'FAN': 'Female Adult',
        'MAN': 'Male Adult'
    }
    return descriptions[speaker]
