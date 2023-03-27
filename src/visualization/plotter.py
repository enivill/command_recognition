import matplotlib.pyplot as plt
import librosa
import librosa.display
import librosa.feature
import numpy as np

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14


class Plotter:
    """
    The Plotter class produces matplotlib.pyplot figures that are correctly
    formatted for a certain data analysis for project
    """

    # inspiration: https://levelup.gitconnected.com/python-classes-to-standardize-plotly-figure-formatting-123fe35c8d2d

    def __init__(self, rows=1, cols=1) -> None:
        """
        initialize object attributes and create figure
        rows = number of rows of plots, 1 or more
        cols = number of cols of plots, 1 or more
        the rows and cols default is 1, but can be changed to add
        subplots
        """

        # initialize figure as subplots
        self.fig, self.axs = plt.subplots(nrows=rows,
                                          ncols=cols,
                                          figsize=(cols * 19.20 / 2, rows * 10.80 / 2),
                                          squeeze=False
                                          )
        # set the spacing between subplots
        plt.subplots_adjust(left=0.080,
                            bottom=0.057,
                            right=0.987,
                            top=0.948,
                            wspace=0.1,
                            hspace=0.278)

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    def plot_audio_waves(self,
                         audio_title: str,
                         file_name: str,
                         data: np.ndarray,
                         sr: int,
                         row: int = 1,
                         col: int = 1):
        """
        plot data on matplotlib.pyplot figure for one or more audio files
        audio_title: str = audio title
        file_name: str = audio file name
        data: np.ndarray = audio time series
        sr: int = sampling rate of data
        row: int = which row to place the plot, default 1
        col: int = which col to place the plot, default 1

        how to use:
            fig = Plotter(rows=1, cols=2)
            fig.plot_audio_waves('bed', file_name_bed, y_bed, sample_rate_bed, 0, 0)
            fig.plot_audio_wave('happy', file_name_happy, y_happy, sample_rate_happy, 0, 1)
            fig.fig.savefig('audio_signals.png')
            fig.fig.show()
        """

        librosa.display.waveshow(y=data, sr=sr, ax=self.axs[row, col])
        self.axs[row, col].set(title=f'Word: {audio_title}\nFile name: {file_name}',
                               xlabel="Time",
                               ylabel="Amplitude")

    def stft(self,
             audio_title: str,
             file_name: str,
             data: np.ndarray,
             n: int,
             wl: int,
             hl: int,
             row: int = 0,
             col: int = 0,
             feature_extr = False
             ):
        """
        :param audio_title:
        :param file_name:
        :param data:
        :param n:
        :param wl:
        :param hl:
        :param row:
        :param col:
        :return:
        """
        # fft = 128    # this corresponds to 0.16 ms with a sample rate of 8000 Hz --> 128/8000 = 0.016
        # win_len = fft
        # hl = win_len//2

        # short-time fourier transform
        # Defaults to a raised cosine window(‘hann’)
        y_fourier = librosa.stft(
            y=data,
            n_fft=n,
            hop_length=hl,
            win_length=wl,
            window='hann'
        )
        y_fourier_decibel = librosa.amplitude_to_db(np.abs(y_fourier), ref=np.max)

        print(y_fourier.shape)
        # print(y_fourier_decibel.shape)
        # spectrogram
        img = librosa.display.specshow(
            data=y_fourier_decibel,
            sr=8000,
            hop_length=hl,
            n_fft=n,
            win_length=wl,
            x_axis='frames',
            y_axis='hz',
            ax=self.axs[row, col]
        )
        if not feature_extr:
            self.axs[row, col].set(title=f'Word: {audio_title}\nFile name: {file_name}',
                                   xlabel="Počet rámcov",
                                   ylabel="Hz")
            if row == 0 and col == 0:
                self.fig.colorbar(mappable=img, ax=self.axs, format="%+2.f dB")

    def mel(self,
            audio_title: str,
            file_name: str,
            data: np.ndarray,
            n: int,
            sr: int,
            wl: int,
            hl: int,
            n_mels: int,
            row: int = 0,
            col: int = 0,
            feature_extr = False
            ):
        # n_fft = 128    # this corresponds to 0.16 ms with at a sample rate of 8000 Hz --> 128/8000 = 0.016

        mel_spectrogram = librosa.feature.melspectrogram(
            y=data,
            sr=sr,
            n_fft=n,
            hop_length=hl,
            win_length=wl,
            n_mels=n_mels
        )

        mel_spectrogram_decibel = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # print(mel_spectrogram.shape)
        # print(mel_spectrogram_decibel.shape)

        # spectrogram
        img = librosa.display.specshow(
            data=mel_spectrogram_decibel,
            sr=sr,
            hop_length=hl,
            n_fft=n,
            win_length=wl,
            x_axis='time',
            y_axis='mel',
            ax=self.axs[row, col]
        )

        if not feature_extr:
            self.axs[row, col].set(title=f'Word: {audio_title}\nFile name: {file_name}',
                                   xlabel="Time",
                                   ylabel="Hz")
            if row == 0 and col == 0:
                self.fig.colorbar(mappable=img, ax=self.axs, format="%+2.f dB")

    def mfcc(self,
             audio_title: str,
             file_name: str,
             data: np.ndarray,
             n: int,
             sr: int,
             wl: int,
             hl: int,
             n_mels: int,
             row: int = 0,
             col: int = 0,
             feature_extr = False
             ):
        # n_fft = 128    # this corresponds to 0.16 ms with at a sample rate of 8000 Hz --> 128/8000 = 0.016

#TODO first time i generated melspectrogram, then converted from ampl to db and finally made mfcc...
# but then i realized it can be done in one step, with the mfcc function with more parameters.
        # mel_spectrogram = librosa.feature.melspectrogram(
        #     y=data,
        #     sr=sr,
        #     n_fft=n,
        #     hop_length=hl,
        #     win_length=wl,
        #     n_mels=n_mels
        # )
        #
        # mel_spectrogram_decibel = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # # print(mel_spectrogram.shape)
        # # print(mel_spectrogram_decibel.shape)
        #
        # mfccs = librosa.feature.mfcc(
        #     S=mel_spectrogram_decibel,
        #     # n_mfcc=int((2 / 3) * n_mels)
        #     n_mfcc=12
        # )
        mfccs = librosa.feature.mfcc(
            y=data, sr=sr, hop_length=hl, win_length=wl, n_mels=n_mels, n_fft=n,
            # n_mfcc=int((2 / 3) * n_mels)
            n_mfcc=12
        )
        img = librosa.display.specshow(
            data=mfccs,
            sr=sr,
            hop_length=hl,
            n_fft=n,
            win_length=wl,
            x_axis='time',
            ax=self.axs[row, col]
        )

        if not feature_extr:
            self.axs[row, col].set(title=f'Word: {audio_title}\nFile name: {file_name}',
                                   xlabel="Time")
            if row == 0 and col == 0:
                self.fig.colorbar(mappable=img, ax=self.axs)
