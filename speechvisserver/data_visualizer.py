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


# Render the data_visualizer page
def data_visualizer(request):
    context = {
        "recordings": Recording.objects.all()
    }
    return render(request, 'data_visualizer.html', context)


# Return a json dataset by indexing or sampling two columns of a saved
# AudioFeature. In the http request should be the recording id, audio
# feature name, x_axis column index, y_axis column index, whether to
# sample at regular intervals (sample = true) or to index from start
# through (start + limit).
def visualize_feature(request):
    # Get the parameters from the http request
    recording_id = request.GET.get('recording_id')
    feature = request.GET.get('feature')
    x_axis = int(request.GET.get('x_axis', 0))
    y_axis = int(request.GET.get('y_axis', 1))
    sample = request.GET.get('sample', False)
    start = float(request.GET.get('start', 0))
    limit = int(request.GET.get('limit'))
    # If the recording_id or feature name are missing return an empty response
    response = {}
    if recording_id and feature:
        # Get the requested AudioFeature record from the database and load the data
        record = AudioFeature.objects.get(recording__id=recording_id, feature=feature)
        record.load_data()
        # Extract the requested columns from the loaded dataset
        x_axis = min(x_axis, record.data.shape[1] - 1)
        y_axis = min(y_axis, record.data.shape[1] - 1)
        # Sample or index the dataset
        if sample:
            index = numpy.floor(numpy.linspace(0, len(record.t), limit, endpoint=False)).astype('int');
        else:
            i = max(numpy.argmin(numpy.abs(record.t - start)) - 100, 0)
            index = numpy.arange(i, i + limit)
        # Add the requested features to the response
        response['t'] = record.t[index].tolist()
        response['x'] = record.data[index, x_axis].tolist()
        response['y'] = record.data[index, y_axis].tolist()
        response['labels'] = record.labels[index].tolist()
    # Render the json response
    return HttpResponse(json.dumps(response), content_type='application/json')

