from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
from django.forms.models import model_to_dict
from speechvisserver.models import *
import random
import csv


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
            annotation_id = request.GET.get("annotation_id")
            annotation = Annotation.objects.get(id=annotation_id)
            segment = annotation.segment
            sensitive = request.GET.get('sensitive', 'false')
            validation = Annotation(segment=segment, sensitive=sensitive,
                coder=coder, method='SPEAKER_VALIDATION')
            validation.speaker = request.GET.get("speaker")
            validation.save()
    recording = Recording.objects.all()[random.randrange(Recording.objects.count())]
    numbers = [s['number'] for s in
        Segment.objects.filter(recording=recording, segment__speaker='CHN').values('number')]
    segment_number = numbers[random.randrange(len(numbers))]
    annotation = Annotation.objects.filter(segment__recording=recording, segment__number=segment_number, coder="LENA").get()
    context = {
        'coder': coder,
        'recording_id': recording.id,
        'segment_number': segment_number,
        'segment_file': '{0}/CHN/{1}.wav'.format(recording.id, segment_number),
        'speaker': annotation.speaker,
        'speaker_descriptive': getDescriptiveName(annotation.speaker),
        'annotation_id': annotation.id,
        'error': error
    }
    return render(request, 'speaker_validation.html', context)


def data_manager(request):
    submit = request.GET.get('submit', '')
    error = ''
    if submit == 'import':
        id = request.GET.get('add_recording_id', '').strip()
        directory = request.GET.get('add_recording_directory', '')
        if not id:
            error = 'Invalid Recording Id'
        else:
            recording = Recording(id=id, directory=directory)
            recording.save()
    export_columns = ['number', 'speaker', 'start', 'end']
    export_id = request.GET.get('export', '')
    if Recording.objects.filter(id=export_id).exists():
        selected_columns = []
        for column in export_columns:
            if request.GET.get('export_{}'.format(column), False):
                selected_columns.append(column)
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="export.csv"'
        writer = csv.DictWriter(response, selected_columns, extrasaction='ignore')
        writer.writeheader()
        for segment in Segment.objects.filter(recording__id=export_id):
            writer.writerow(model_to_dict(segment))
        return response
    else:
        context = {
            'export_formats': ['CSV', 'NPY', 'MAT'],
            'export_columns': export_columns,
            'recordings': Recording.objects.all(),
            'error': error
        }
        return render(request, 'data_manager.html', context)


def data_visualizer(request):
    types = ['speaker_counts', 'durations', 'intervals', 'volubility']
    speakers = ['CHN', 'FAN', 'MAN', 'CXN']
    context = {
        'plot_types': types,
        'speakers': speakers,
        'ids': Recording.objects.values('id')
    }
    return render(request, 'data_visualizer.html', context)


def plot(request):
    types = ['speaker_counts', 'durations', 'intervals', 'volubility']
    type = request.GET.get('plot_type', types[0])
    speakers = ['CHN', 'FAN', 'MAN', 'CXN']
    speaker = request.GET.get('speaker', speakers[0])
    import speechvis
    id = request.GET.get('recording_id', speechvis.ids[0])
    recording = speechvis.Recording('D:/ivfcr', id)
    if type == types[0]:
        fig = speechvis.plot_speaker_counts(recording)
    elif type == types[1]:
        fig = speechvis.plot_durations(recording)
    elif type == types[2]:
        fig = speechvis.plot_intervals(recording, speaker)
    elif type == types[3]:
        fig = speechvis.plot_volubility(recording, speaker)

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn
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
