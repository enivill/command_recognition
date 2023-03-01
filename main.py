
from matplotlib import use as mpl_use
from src.visualization.plotter import Plotter
import src.visualization.visualize as visualize
import src.features.build_features as build_features

mpl_use('Qt5Agg')

if __name__ == '__main__':
    # plot .wav file - signal
    # src.visualization.visualize.plot_data_and_save(img_type='wave', words_no=1, files_no=1)
    # create and plot stft spectrogram
    # src.visualization.visualize.plot_data_and_save(_type='stft', words_no=1, files_no=1)
    # # create and plot mel spectrogram
    # src.visualization.visualize.plot_data_and_save(_type='mel', words_no=1, files_no=1)
    # # create and plot mfcc
    # visualize.plot_data_and_save(img_type='mfcc', words_no=1, files_no=1)

    # create feature
    build_features.feature_extraction(feature_type='mfcc', data_path='data/external/speech_commands_v0.01/', files_no=100)
