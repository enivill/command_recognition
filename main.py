from matplotlib import use as mpl_use
from pandas import read_csv
from src.utils import config as my_config
from src.models.model import SiameseNet
from src.features.build_features import feature_extraction, plot_audio, plot_mel, plot_mfcc, plot_stft
from librosa.feature import mfcc

# mpl_use('Qt5Agg')
# mpl_use('TkAgg')

if __name__ == '__main__':

    try:
        print("--------------------------------------------------------MODEL 1----------")
        my_config.load_config("config.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 1")

    try:
        print("--------------------------------------------------------MODEL 2----------")
        my_config.load_config("config2.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 2")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config3.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 3")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config4.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 4")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config5.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 5")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config6.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 6")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config7.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 7")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config8.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 8")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config9.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 9")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config10.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 10")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config11.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 11")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config12.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 12")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config13.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 13")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config14.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 14")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config15.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 15")

    try:
        print("--------------------------------------------------------MODEL 3----------")
        my_config.load_config("config16.yaml")
        mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
        siamese_net = SiameseNet(shape=mel.shape)
        siamese_net.build()
        siamese_net.train()
        siamese_net.evaluate()
    except:
        print("ERROR DURING TRAINING 16")

    # siamese_net.restore_model(file="models/siamese_with_testing/model.h5")

    # mel = feature_extraction("/bed/00f0204f_nohash_1.wav")
    # print(mel.shape)
    # plot_mel(mel)
