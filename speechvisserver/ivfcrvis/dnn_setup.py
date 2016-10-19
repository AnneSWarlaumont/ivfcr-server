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
recording = Recording.objects.all()[0]
recording.read_audio()
fbanks = recording.frequency_banks()
mins = numpy.min(fbanks, axis=0)
maxes = numpy.max(fbanks, axis=0)

# Set network parameters
tsteps = 1
batch_size = 400
epochs = 50
lahead = 1

# Generate the training input
print('Generating Data')
xLength = batch_size * int(0.75 * len(fbanks) / batch_size)
x = fbanks[:xLength, :, :]
x = (x - mins) / (maxes - mins)
print('Input shape:', x.shape)

# Generate the training output
#expected_output = numpy.zeros((len(x), 52))
#for i in range(len(x) - 5):
#    expected_output[i, 0:26] = numpy.mean(x[i + 1:i + 5, 0, 0:26], axis=0)
#    expected_output[i, 26:52] = numpy.var(x[i + 1:i + 5, 0, 0:26], axis=0)
#expected_output = (expected_output - numpy.mean(expected_output, axis=0)) / numpy.std(expected_output, axis=0)
expected_output = numpy.zeros((len(x), 26))
for i in range(4, len(x)):
    expected_output[i, 0:26] = x[i - 4, 0, :]
    #expected_output[i, 26:52] = x[i - 4, 0, :]
    #expected_output[i, 52:78] = x[i - 8, 0, :]
expected_mins = numpy.min(expected_output, axis=0)
expected_maxes = numpy.max(expected_output, axis=0)
expected_output = (expected_output - expected_mins) / (expected_maxes - expected_mins)
print('Output shape: ', expected_output.shape)

# Create the recurrent neural network
print('Creating Model')
model = Sequential()
rnn1 = LSTM(100, batch_input_shape=(batch_size, tsteps, 26),
           return_sequences=False, stateful=True, dropout_W=0.02)
model.add(rnn1)
#rnn2 = SimpleRNN(50, return_sequences=False, stateful=True)
#model.add(rnn2)
model.add(Dense(26))
rmsprop = RMSprop(lr=0.00001)
model.compile(loss='mse', optimizer=rmsprop)

# Train the network
print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    model.fit(x,
              expected_output,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False)
    model.reset_states()

# Generate the test input and output
print('Predicting')
testOffset = len(x)
testLength = batch_size * int(0.05 * len(fbanks) / batch_size)
test_input = fbanks[testOffset:testOffset + testLength,:,:]
test_input = (test_input - mins) / (maxes - mins)
#test_expected = numpy.zeros((len(test_input), 52))
#for i in range(len(test_input) - 5):
#    test_expected[i, 0:26] = numpy.mean(test_input[i + 1:i + 5, 0, 0:26], axis=0)
#    test_expected[i, 26:52] = numpy.var(test_input[i + 1:i + 5, 0, 0:26], axis=0)
#test_expected = (test_expected - numpy.mean(test_expected, axis=0)) / numpy.std(test_expected, axis=0)
test_expected = numpy.zeros((len(test_input), 26))
for i in range(4, len(test_input)):
    test_expected[i, 0:26] = test_input[i - 4, 0, :]
    #test_expected[i, 26:52] = test_input[i + 4, 0, :]
    #test_expected[i, 52:78] = test_input[i + 8, 0, :]
test_expected = (test_expected - expected_mins) / (expected_maxes - expected_mins)

# Create an encoder model to get hidden layer activity
encoder = Sequential()
encoder.add(rnn1)
#encoder.add(rnn2)

# Generate the hidden layer and predictions for the test data
test_hidden = encoder.predict(test_input, batch_size=batch_size)
test_output = model.predict(test_input, batch_size=batch_size)

# Plot the direct results for an arbitrary segment
print('Plotting Results')
start = 0
end = len(test_input)
pyplot.figure()
pyplot.subplot(411)
pyplot.imshow(test_input.reshape(len(test_input), 26).transpose()[:, start:end],
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(412)
pyplot.imshow(test_expected.transpose()[:, start:end],
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(413)
pyplot.imshow(test_output.transpose()[:, start:end],
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(414)
error = numpy.sqrt(numpy.mean(numpy.square(test_expected[:,0:26] - test_output[:,0:26]), axis=1))
pyplot.plot(error[start:end])
pyplot.show()

def filter_data_by_speaker(data, offset, index, starts, ends):
    filteredData = numpy.zeros((0, 100))
    filteredIndex = numpy.zeros((0, 3))
    for i, start, end in zip(index, starts, ends):
        a = int(start * 40) - offset
        b = int(end * 40) - offset
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
speaker = 'CHN'
segments = recording.segment.filter(annotation__speaker=speaker)
values = numpy.array(segments.values_list('number', 'start', 'end'))
filteredData, filteredIndex = filter_data_by_speaker(test_hidden, testOffset, values[:,0], values[:,1], values[:,2])

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(filteredData)

#ica = FastICA(n_components=3)
#v = ica.fit_transform(rnns)

#tsne = TSNE(n_components=2)
#v = tsne.fit_transform(rnns)

v = pca.transform(filteredData)
plot_vocal_space(v, 'grey', 2500, speaker)

for i in range(50, 55):
    plot_vocal_trajectory(v, int(filteredIndex[i,1]), int(filteredIndex[i,2]), filteredIndex[i,0])
