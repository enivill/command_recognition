from matplotlib import use as mpl_use
from src.visualization.plotter import Plotter
import src.visualization.visualize

mpl_use('Qt5Agg')

if __name__ == '__main__':
    # plot .wav file - signal
    src.visualization.visualize.plot_data_and_save(_type='wave',  files_no=1)
    # create and plot stft spectrogram
    # src.visualization.visualize.plot_data_and_save(_type='stft', words_no=1, files_no=1)
    # # create and plot mel spectrogram
    # src.visualization.visualize.plot_data_and_save(_type='mel', words_no=1, files_no=1)
    # # create and plot mfcc
    # src.visualization.visualize.plot_data_and_save(_type='mfcc', words_no=1, files_no=1)


