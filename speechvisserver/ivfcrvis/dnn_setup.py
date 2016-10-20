import numpy
from matplotlib import pyplot
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import RMSprop
from scipy.io import wavfile
from features import logfbank
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "speechvisserver.settings")
import django
django.setup()
from speechvisserver.models import Recording

# Load the recording audio and convert it to Mel-frequency spectrum
fbanks = numpy.zeros((0, 1, 26))
for recording in Recording.objects.all():
    print('Reading audio: {}'.format(recording.id))
    recording.read_audio()
    fbanks = numpy.concatenate((fbanks, recording.frequency_banks(winlen=0.2, winstep=0.2)))
# Normalize energy per frame
window_means = fbanks.mean(axis=2).reshape(len(fbanks), 1, 1)
inputs = fbanks.copy() - window_means
# Scale values per feature
mins = numpy.percentile(inputs, 1, axis=0)
maxes = numpy.percentile(inputs, 99, axis=0)
inputs = ((inputs - mins) / (maxes - mins)) * 2 - 1
# Generate the training output
outputs = numpy.zeros((len(inputs), 26))
for i in range(4, len(inputs)):
    outputs[i, 0:26] = inputs[i - 4, 0, :] - inputs[i, 0, :]
output_mins = numpy.percentile(outputs, 1, axis=0)
output_maxes = numpy.percentile(outputs, 99, axis=0)
outputs = (outputs - output_mins) / (output_maxes - output_mins) * 2 - 1

# Set network parameters
batch_size = 1500
epochs = 50

xLength = batch_size * int(len(fbanks) / batch_size)
x = inputs[:xLength, :, :]
y = outputs[:xLength, :]

# Create the recurrent neural network
print('Creating Model')
model = Sequential()
model.add(LSTM(100, batch_input_shape=(batch_size, 1, 26),
          return_sequences=True, stateful=False))
model.add(LSTM(100, batch_input_shape=(batch_size, 1, 26),
          return_sequences=True, stateful=False))
model.add(LSTM(100, batch_input_shape=(batch_size, 1, 26),
          return_sequences=False, stateful=False))
model.add(Dense(26))
rmsprop = RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer=rmsprop)

# Train the network
print('Training')
history = model.fit(x, y,
          batch_size=batch_size,
          verbose=1,
          nb_epoch=epochs,
          shuffle='batch')
model.reset_states()

# Visualize the network activity
print('Predicting')
# Create an encoder model to get hidden layer activity
encoder = Sequential()
encoder.add(model.layers[0])
h1 = encoder.predict(x, batch_size=batch_size)
encoder.reset_states()
encoder.add(model.layers[1])
h2 = encoder.predict(x, batch_size=batch_size)
encoder.reset_states()
encoder.add(model.layers[2])
h3 = encoder.predict(x, batch_size=batch_size)
encoder.reset_states()
h = numpy.concatenate((h1, h2, h3), axis=2)
h = numpy.reshape(h, (len(h), 300))
h = (h - h.mean(axis=0)) / h.std(axis=0)
o = model.predict(x, batch_size=batch_size)

# Plot the direct results for an arbitrary segment
print('Plotting Results')
start = 185000
end = 186000
pyplot.figure()
pyplot.subplot(411)
xScale = (maxes - mins) * ((x.copy() + 1) / 2) + mins + window_means[:xLength]
pyplot.imshow(xScale.reshape(xLength, 26).transpose()[:, start:end],
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(412)
yScale = (output_maxes - output_mins) * ((y.copy() + 1) / 2) + output_mins
pyplot.imshow(yScale.transpose()[:, start:end], vmin=-3.6, vmax=3.6,
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(413)
oScale = (output_maxes - output_mins) * ((o.copy() + 1) / 2) + output_mins
pyplot.imshow(oScale.transpose()[:, start:end], vmin=-3.6, vmax=3.6,
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(414)
xError = numpy.sqrt(numpy.mean(numpy.square((oScale[4:,0:26] + x[4:,0,0:26]) - x[:-4,0,0:26]), axis=1))
yError = numpy.sqrt(numpy.mean(numpy.square(oScale[4:,0:26]), axis=1))
pyplot.plot(range(start, end), xError[start:end] - yError[start:end])
pyplot.plot([start, end], [0, 0], '--k')
pyplot.show()

def filter_data_by_speaker(data, offset, index, starts, ends):
    filteredData = numpy.zeros((0, 300))
    filteredIndex = numpy.zeros((0, 3))
    for i, start, end in zip(index, starts, ends):
        a = int(start * 5) - offset
        b = int(end * 5) - offset
        if a >= 0 and b < len(data):
            filteredIndex = numpy.vstack((filteredIndex,
                            [i, len(filteredData), len(filteredData) + (b - a)]))
            filteredData = numpy.concatenate((filteredData, data[a:b, :]), axis=0)
    return filteredData, filteredIndex

def plot_vocal_kde(v):
    from scipy import stats
    v = v[numpy.random.rand(len(v)) < 0.05, :]
    x = v[:, 0]
    y = v[:, 1]
    xmin, xmax = -4, 8
    ymin, ymax = -3, 4
    # Peform the kernel density estimate
    xx, yy = numpy.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = numpy.vstack([xx.ravel(), yy.ravel()])
    values = numpy.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = numpy.reshape(kernel(positions).T, xx.shape)
    fig = pyplot.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Contourf plot
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    ## Or kernel density estimate plot instead of the contourf plot
    #ax.imshow(numpy.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
    # Contour plot
    cset = ax.contour(xx, yy, f, colors='k')
    # Label plot
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('Y1')
    ax.set_ylabel('Y0')

def plot_vocal_space(v, c, pts, lbl):
    v = v[numpy.random.rand(len(v)) < pts / len(v), :]
    pyplot.subplot(221)
    pyplot.scatter(v[:,0], v[:,1], c=c, marker='.', edgecolors='face', label=lbl)
    pyplot.ylabel('{0:.2f}% Variance'.format(100 * pca.explained_variance_ratio_[1]))
    pyplot.subplot(222)
    pyplot.scatter(v[:,2], v[:,1], c=c, marker='.', edgecolors='face', label=lbl)
    pyplot.xlabel('{0:.2f}% Variance'.format(100 * pca.explained_variance_ratio_[2]))
    pyplot.subplot(223)
    pyplot.scatter(v[:,0], v[:,2], c=c, marker='.', edgecolors='face', label=lbl)
    pyplot.xlabel('{0:.2f}% Variance'.format(100 * pca.explained_variance_ratio_[0]))


def plot_vocal_trajectory(v, a, b, i):
    pyplot.subplot(221)
    pyplot.plot(v[a:b, 0], v[a:b, 1], lw=1.5, label='Voc {}'.format(i))
    pyplot.subplot(222)
    pyplot.plot(v[a:b, 2], v[a:b, 1], lw=1.5, label='Voc {}'.format(i))
    pyplot.subplot(223)
    pyplot.plot(v[a:b, 0], v[a:b, 2], lw=1.5, label='Voc {}'.format(i))

# Filter the hidden layer activity by LENA speaker label
speaker = 'NON'
segments = recording.segment.filter(annotation__speaker=speaker)
values = numpy.array(segments.values_list('number', 'start', 'end'))
filteredData, filteredIndex = filter_data_by_speaker(h, 0, values[:,0], values[:,1], values[:,2])

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(filteredData)

#ica = FastICA(n_components=3)
#v = ica.fit_transform(rnns)

#tsne = TSNE(n_components=2)
#v = tsne.fit_transform(rnns)

v = pca.transform(filteredData)
plot_vocal_space(v, 'orange', 2500, speaker)

for i in range(50, 55):
    plot_vocal_trajectory(v, int(filteredIndex[i,1]), int(filteredIndex[i,2]), filteredIndex[i,0])
