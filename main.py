from matplotlib import use as mpl_use
from pandas import read_csv
from src.utils import config as my_config
from src.models.model import SiameseNet
from src.features.build_features import feature_extraction, plot_audio, plot_mel, plot_mfcc, plot_stft
from librosa.feature import mfcc

# mpl_use('Qt5Agg')
# mpl_use('TkAgg')

if __name__ == '__main__':


    print("--------------------------------------------------------MODEL 1----------")
    my_config.load_config("config.yaml")
    siamese_net = SiameseNet()
    siamese_net.build()
    siamese_net.train()
    siamese_net.evaluate()
    
    # siamese_net.restore_model(file="models/siamese_with_testing/model.h5")


    #
    #
    # mfcc = feature_extraction("/bed/00f0204f_nohash_1.wav")
    # print(mfcc.shape)
    # plot_mfcc(mfcc)
    # print("--------------------------------------------------------MODEL 2----------")
    # my_config.load_config("config2.yaml")
    # siamese_net = SiameseNet()
    # siamese_net.build()
    # siamese_net.train()
    # siamese_net.evaluate()

    # print("--------------------------------------------------------MODEL 3----------")
    # my_config.load_config("config3.yaml")
    # siamese_net = SiameseNet()
    # siamese_net.build()
    # siamese_net.train()
    # siamese_net.evaluate()














