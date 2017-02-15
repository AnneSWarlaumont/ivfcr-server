import os
from speechvisserver.models import *
from sklearn.decomposition import PCA, FastICA


def import_recordings_from_directory(directory):
    recording_ids = [f[:-4] for f in os.listdir(directory) if f.endswith('.its')]
    for id in recording_ids:
        if Recording.objects.filter(id=id).count() == 0:
            print('Importing {}'.format(id))
            rec = Recording.import_recording(id, directory, False)
            t, fbanks = rec.frequency_banks()
            AudioFeature.save_feature(rec, 'fbanks', t, fbanks)
        else:
            print('{} Already Exists'.format(id))


def load_fbanks_dataset():
    dataset = numpy.zeros((0,26))
    for feat in AudioFeature.objects.filter(feature='fbanks'):
        feat.load_data()
        dataset = numpy.concatenate((dataset, feat.data), axis=0)
    return dataset


def save_pca_dataset():
    print('Loading Dataset')
    dataset = load_fbanks_dataset()
    print('Applying PCA')
    pca = pca_reduce(dataset)
    for feat in AudioFeature.objects.filter(feature='fbanks'):
        print('Saving {} pca_fbanks'.format(feat.recording.id))
        feat.load_data()
        v = pca.transform(feat.data)
        pca_feat = AudioFeature.save_feature(recording=feat.recording, feature='pca_fbanks',
                                             t=feat.t, data=v, labels=feat.labels)


def save_pca_hist_dataset():
    print('Loading Dataset')
    dataset = load_fbanks_dataset()
    dataset = numpy.concatenate((dataset[:-3,:], dataset[1:-2,:], dataset[2:-1,:], dataset[3:,:]), axis=1)
    print('Applying PCA')
    pca = pca_reduce(dataset)
    for feat in AudioFeature.objects.filter(feature='fbanks'):
        print('Saving {} pca_fbanks'.format(feat.recording.id))
        feat.load_data()
        data = numpy.concatenate((data[:-3,:], data[1:-2,:], data[2:-1,:], data[3:,:]), axis=1)
        v = pca.transform(feat.data)
        pca_feat = AudioFeature.save_feature(recording=feat.recording, feature='pca_fbanks',
                                             t=feat.t, data=v, labels=feat.labels)


def save_ica_dataset():
    print('Loading Dataset')
    dataset = load_fbanks_dataset()
    print('Applying ICA')
    ica = ica_reduce(dataset, components=7)
    for feat in AudioFeature.objects.filter(feature='fbanks'):
        print('Saving {} ica_fbanks'.format(feat.recording.id))
        feat.load_data()
        v = ica.transform(feat.data)
        ica_feat = AudioFeature.save_feature(recording=feat.recording, feature='ica_fbanks',
                                             t=feat.t, data=v, labels=feat.labels)


def pca_reduce(dataset, components=2, whiten=True):
    pca = PCA(n_components=components, whiten=whiten)
    pca.fit(dataset)
    return pca


def ica_reduce(dataset, components=2):
    ica = FastICA(n_components=components)
    ica.fit(dataset)
    return ica
