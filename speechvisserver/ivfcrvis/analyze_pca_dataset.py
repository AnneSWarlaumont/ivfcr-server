from speechvisserver.models import *
import numpy
from matplotlib import pyplot
import seaborn


def load_dataset():
    features = []
    for feat in AudioFeature.objects.filter(feature='pca_fbanks'):
        feat.load_data()
        features.append(feat)
    return features


def plot_x(features):
    means = []
    stds = []
    for feat in features:
        means.append(feat.data[:,0].mean())
        stds.append(feat.data[:,0].std())
    pyplot.errorbar(range(len(features)), means, stds)
    pyplot.show()


def plot_cdf_dx(features):
    dx = numpy.array((0, 1))
    for feat in features:
        dx = numpy.concatenate((dx, numpy.abs(numpy.diff(feat.data[:,1], 1))), axis=0)
    pyplot.plot(numpy.sort(dx), numpy.linspace(1, 0, dx.size))
    pyplot.show()


def load_pca_features():
    features = []
    for f in AudioFeature.objects.filter(feature='pca_fbanks'):
        f.load_data()
        features.append(f)
    return features


def save_pca_features_as_mat(features):
    dataset = numpy.ndarray((0,6))
    for f in features:
        index = numpy.sort(numpy.random.randint(0, len(f.labels), 100000))
        recordings = numpy.reshape(numpy.repeat(f.recording.id, 100000), (100000, 1))
        children = numpy.reshape(numpy.repeat(f.recording.child.id, 100000), (100000, 1))
        labels = numpy.reshape(f.labels[index], (100000, 1))
        t = numpy.reshape(f.t[index], (100000, 1))
        fdata = numpy.hstack((recordings, children, labels, t, f.data[index,:]))
        dataset = numpy.concatenate((dataset, fdata), axis=0)
    return dataset


def plot_contingencies(features, invert=False):
    from matplotlib.colors import LogNorm
    sub = 1
    seaborn.set_style('dark')
    pyplot.figure()
    for feature in features:
        rows = numpy.zeros_like(feature.t, dtype=bool)
        last_fan = feature.t[-1]
        contingent = False
        for i in range(len(feature.labels) - 1, 0, -1):
            if feature.labels[i] == 'CHN':
                if last_fan - feature.t[i] <= 2:
                    contingent = True
            else:
                contingent = False
            if feature.labels[i] == 'FAN':
                last_fan = feature.t[i]
            rows[i] = contingent
        not_rows = numpy.logical_and(feature.labels == 'CHN', numpy.logical_not(rows)[:len(feature.labels)])
        x = feature.data[rows, 0]
        y = feature.data[rows, 1]
        not_x = feature.data[not_rows, 0]
        not_y = feature.data[not_rows, 1]
        pyplot.subplot(3, 3, sub)
        pyplot.hist2d(x, y, range=[(-4, 4), (-5, 5)], bins=50, norm=LogNorm(), cmap=pyplot.cm.jet)
        #h1, _, _ = numpy.histogram2d(x, y, range=[(-4, 4), (-5, 5)], bins=50, normed=True)
        #h2, _, _ = numpy.histogram2d(not_x, not_y, range=[(-4, 4), (-5, 5)], bins=50, normed=True)
        #pyplot.colorbar()
        #pyplot.imshow(h1 - h2, cmap=pyplot.cm.jet)
        pyplot.xlim(-4, 4)
        pyplot.ylim(-5, 5)
        sub += 1


def plot_means(features):
    seaborn.set_style('darkgrid')
    seaborn.set_palette(seaborn.color_palette('muted'))
    pyplot.figure()
    chn_x = []
    chn_y = []
    fan_x = []
    fan_y = []
    non_x = []
    non_y = []
    for feature in features:
        rows = feature.labels == 'CHN'
        chn_x.append(feature.data[rows, 0].mean())
        chn_y.append(feature.data[rows, 1].mean())
        rows = feature.labels == 'FAN'
        fan_x.append(feature.data[rows, 0].mean())
        fan_y.append(feature.data[rows, 1].mean())
        rows = feature.labels == ''
        non_x.append(feature.data[rows, 0].mean())
        non_y.append(feature.data[rows, 1].mean())
    pyplot.plot(chn_x, chn_y, 'o')
    pyplot.plot(fan_x, fan_y, '^')
    pyplot.plot(non_x, non_y, 's')


def plot_hist2d(features, speaker):
    from matplotlib.colors import LogNorm
    i = 1
    seaborn.set_style('dark')
    pyplot.figure()
    for feature in features:
        rows = feature.labels == speaker
        x = feature.data[rows, 0]
        y = feature.data[rows, 1]
        pyplot.subplot(3, 3, i)
        pyplot.hist2d(x, y, range=[(-4, 4), (-5, 5)], bins=50, norm=LogNorm(), cmap=pyplot.cm.jet)
        #pyplot.colorbar()
        pyplot.xlim(-4, 4)
        pyplot.ylim(-5, 5)
        #pyplot.clim(0, 0.5)
        i += 1


def plot_kde(features):
    from scipy import stats
    fig, ax = pyplot.subplots(3, 3)
    i = 1
    for feature in features:
        v = feature.data
        v = v[numpy.random.rand(len(v)) < 1000 / len(v), :]
        x = v[:, 0]
        y = v[:, 1]
        xmin, xmax = -3, 3
        ymin, ymax = -3, 3
        # Peform the kernel density estimate
        xx, yy = numpy.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = numpy.vstack([xx.ravel(), yy.ravel()])
        values = numpy.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        f = numpy.reshape(kernel(positions).T, xx.shape)
        pyplot.subplot(3, 3, i)
        i += 1
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
    pyplot.show()
