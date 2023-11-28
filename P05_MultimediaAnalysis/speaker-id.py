'''
A Speaker identification example
=================================

Load speaker training data and build respective models
- Each wav file in the 'train' folder is assumed to contain clean speech from exactly one speaker
- Extract MFCC features for each document (wav file) with standard parameters
- Build a GMM model for the voice with standard parameters

On a larger data set, the speaker identification performance could also be assessed using recall & precision (here, it makes no sense because the collection is so small and it works so well -> so accuracy is used)

@autor: stdm, 17.02.2017
'''

import os
import sys
import librosa
import numpy as np
from sklearn import mixture
from scipy.io import wavfile

#
# MAIN
#
def main():
    training_data = "./train"
    test_data = "./test"

    speakers = load_data(training_data)
    print("List of known speakers:")
    for speaker in speakers:
        print(speaker['name'])

    utterances = load_data(test_data)
    print("\nList of voice samples:")
    for sample in utterances:
        print(sample['file'])
    print

    score, recall = score_models(speakers, utterances)
    print("Overall accuracy of speaker identification: {a}%".format(a=score))
    print("Recall of speaker identification: {0}".format(recall))

#
# Commonly used methods to structure the code
#
def extract_features(signal, sample_rate):
    '''extract 13 MFCCs from the given signal with frame size = 512 samples, frame step = 256 samples'''
    frame_size = 512
    frame_step = 256
    mfccs = librosa.feature.mfcc(signal, sr=sample_rate, n_mfcc=13, n_fft=frame_size, hop_length=frame_step)
    return mfccs.transpose() #return the features in the format expected by scikit-learn

def build_model(features):
    '''build a 16 component diagonal covariance GMM from the given features (usually 13 MFCCs)'''
    mixture_count = 16
    gmm = mixture.GaussianMixture(n_components=mixture_count, covariance_type='diag', n_init=1)
    gmm.fit(features)
    return gmm

def get_speaker_name(file_name):
    '''extract the speaker name from the file name, assuming that the name is the file name without the extension'''
    dot_position = file_name.find('.')
    if dot_position > -1:
        return file_name[0:dot_position]
    else:
        return file_name #no dot

#
# Code to load utterances from files
#
def load_data(path, build_models=True):
    files = os.listdir(path)
    speakers = []
    for wav_file in files:
        if wav_file.endswith(".wav"):
            fs, signal = wavfile.read(os.path.join(path, wav_file))
            mfccs = extract_features(signal, fs)
            gmm = None
            if build_models:
                gmm = build_model(mfccs)
            speaker = {'file': wav_file, 'name':get_speaker_name(wav_file), 'mfccs':mfccs, 'gmm':gmm}
            speakers.append(speaker)
    return speakers

#
# Score the test samples against the pre-build speaker models
#
def score_models(speakers, utterances):
    small_number = -sys.float_info.max
    correct = 0
    for sample in utterances:
        best_score = small_number #the smallest (i.e., most neghative) float number
        for speaker in speakers: #find the most similar known speaker for the given test sample of a voice
            score_per_featurevector = speaker['gmm'].score(sample['mfccs']) #yields log-likelihoods per feature vector
            score = np.sum(score_per_featurevector) #...these can be aggregated to a score per feature-set by summing
            if score > best_score:
                best_score = score
                best_match = speaker
        if best_match['name'] == sample['name']:
            correct += 1
        print("Utterance '{q}' has been identified as speaker '{d}'".format(q=sample['name'], d=best_match['name']))
        score = 100*(float(correct)/float(len(utterances)))
        recall = score/100

    return score, recall


# start the script if executed directly
if __name__ == '__main__':
    main()
