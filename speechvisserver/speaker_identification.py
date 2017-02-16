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


# Render the speaker identification task
def speaker_identification(request):
    # Extract the coder and action parameters from the request
    coder = request.GET.get('coder', '')
    submit = request.GET.get('submit', '')
    # Setup response variable values for the task
    error = ''
    coders_list = ['Tim Shea', 'Monica Mendiola', 'Alison Cao Romero']
    speakers_list = ['CHN', 'CXN', 'FAN', 'MAN']
    descriptions = ['Primary Child', 'Other Child', 'Female Adult', 'Male Adult']
    # Setup a segment to return to the response
    segment = None
    if submit and submit.lower() in ['previous', 'save']:
        segment_id = request.GET.get('segment_id')
        segment = Segment.objects.get(id=segment_id)
        if submit.lower() == 'previous':
            segment = select_previous_segment(segment, speakers_list)
        else:
            if submit.lower() == 'save':
                if not coder:
                    error = 'Your name is required in the coder field.'
                else:
                    speakers = request.GET.getlist('speaker')
                    sensitive = request.GET.get('sensitive', False)
                    annotate_segment(segment, coder, speakers, sensitive)
            segment = select_next_segment(segment, speakers_list)
    else:
        recording_id = request.GET.get('recording_id', '')
        if recording_id != '':
            segment = select_unidentified_segment(recording_id, coder, speakers_list)
    # Render the view for the identification task with the selected segment
    speakers_list.append('REJ')
    descriptions.append('Noise, Television, Unknown')
    context = {
        'coders': coders_list,
        'recordings': Recording.objects.all(),
        'speakers': list(zip(speakers_list, descriptions)),
        'coder': coder,
        'segment': None,
        'segment_file': '',
        'percent_complete': 0,
        'error': error
    }
    if segment is not None:
        context['recording'] = segment.recording
        context['segment'] = segment
        context['percent_complete'] = get_percent_complete(segment.recording, speakers_list, coder)
        context['segment_file'] = segment.static_path
        annotations = segment.annotation.all()
        annotation = annotations.filter(method='SPEAKER_IDENTIFICATION', coder=coder)
        if annotation.count() > 0:
            context['selected_speakers'] = annotation[0].speaker
            context['selected_sensitive'] = annotation[0].sensitive
    return render(request, 'speaker_identification.html', context)


# Select the first segment from a speaker in speakers list that is not identified
def select_unidentified_segment(recording_id, coder, speakers_list):
    recording = Recording.objects.filter(id=recording_id).get()
    records = Segment.objects.filter(recording=recording, annotation__speaker__in=speakers_list)
    records = records.exclude(annotation__method='SPEAKER_IDENTIFICATION', annotation__coder=coder)
    records = records.order_by('recording__id', 'number')
    # TODO: This should handle the case where there are no remaining records!
    return records[0]


# Select the segment from any of the speakers in speakers list that follows the specified segment
def select_next_segment(segment, speakers_list):
    records = Segment.objects.filter(recording=segment.recording, annotation__speaker__in=speakers_list)
    records = records.filter(number__gt=segment.number).order_by('number')
    if records.count() == 0:
        return segment
    else:
        return records[0]


# Select the segment from any of the speakers in speakers list that precedes the specified segment
def select_previous_segment(segment, speakers_list):
    records = Segment.objects.filter(recording=segment.recording, annotation__speaker__in=speakers_list)
    records = records.filter(number__lt=segment.number).order_by('-number')
    if records.count() == 0:
        return segment
    else:
        return records[0]


# Create an annotation record with the specified speakers
def annotate_segment(segment, coder, speakers, sensitive):
    annotations = segment.annotation.all()
    annotations = annotations.filter(method='SPEAKER_IDENTIFICATION', coder=coder)
    if annotations.count() > 0:
        annotation = annotations[0]
    else:
        annotation = Annotation(segment=segment, coder=coder, method='SPEAKER_IDENTIFICATION')
    if len(speakers) == 0:
        annotation.speaker = 'SIL'
    elif len(speakers) == 1:
        annotation.speaker = speakers[0]
    else:
        annotation.speaker = ','.join(speakers)
    annotation.sensitive = True if sensitive else False
    annotation.save()


# Select rows that have speaker identification records for the specified coder
def get_percent_complete(recording, speakers_list, coder):
    all_rows = recording.segment.filter(annotation__coder='LENA', annotation__speaker__in=speakers_list)
    completed_rows = all_rows.filter(annotation__coder=coder, annotation__method='SPEAKER_IDENTIFICATION')
    percent_complete = int(100 * completed_rows.count() / all_rows.count())
    return percent_complete
