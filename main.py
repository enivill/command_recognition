from src.utils import config as my_config
from src.models.model import SiameseNet
from src.features.build_features import plot_mel, feature_extraction_dataset
from src.data import build_siamese_pairs
from src.data import get_data
import numpy as np
from src.utils.logger import get_logger
from itertools import combinations_with_replacement

LOG = get_logger('SiameseNet')


def start_experiment():
    my_config.load_config(f"config_training.yaml")
    my_config.get_config()["layers"]["cnn"]["conv"]["kernel"] = [3, 3, 3, 3, 3, 3]
    my_config.get_config()["layers"]["cnn"]["conv"]["stride"] = [1, 1, 1, 1, 1, 1]
    my_config.get_config()["layers"]["cnn"]["batchnorm"] = [True, True, True, True, True, True]
    my_config.get_config()["layers"]["cnn"]["dropout"] = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    my_config.get_config()["layers"]["cnn"]["pool"]["size"] = [2, 2, 2, 2, 2, 2, 2]
    my_config.get_config()["layers"]["cnn"]["pool"]["stride"] = [2, 2, 2, 2, 2, 2, 2]

    for i in [4, 56]:
        if i == 4:
            my_config.get_config()["layers"]["dns"]["units"] = [1000, 500]
        if i == 5:
            my_config.get_config()["layers"]["dns"]["units"] = [500, 250]
        if i == 6:
            my_config.get_config()["layers"]["dns"]["units"] = [100, 50]
        for idx, j in enumerate(combinations_with_replacement([32, 64, 128, 256], i)):
            # if idx not in [5, 9, 28, 37]:
            #     continue
            j = list(j)
            my_config.get_config()["layers"]["cnn"]["conv"]["filters"] = j

            for k in range(2, 3):
                my_config.get_config()["train"]["log"]["name"] = f"cnn{i}_id{idx}_{k}"

                print(
                    f'--------------------------------------------------------{my_config.get_config()["train"]["log"]["name"]}-----------')
                print(
                    "------------------------------------------------------------------------------------------------")
                try:
                    print(f"--------------------------------------------------------MODEL {i} {j}----------")
                    siamese_net = SiameseNet()
                    siamese_net.build()
                    siamese_net.train()
                    siamese_net.evaluate()
                except Exception as error:
                    print(f"ERROR DURING TRAINING {i}")
                    LOG.error(error)


def do_feature_extraction():
    my_config.load_config(f"config_feature_extraction.yaml")
    feature_extraction_dataset()


def plot_mel_spectrogram():
    my_config.load_config(f"config_training.yaml")
    audio_1 = np.load("data/features/mel26_wl20_hl10_fmin300_fmax4000_sr16000/forward/0a2b400e_nohash_0.npy")

    my_config.get_config()['feature_extraction']['n_mels'] = 26
    my_config.get_config()['feature_extraction']['f_min'] = 300
    my_config.get_config()['feature_extraction']['f_max'] = 4000
    print(audio_1.shape)
    plot_mel(audio_1)


def test_restored_model():
    my_config.load_config(f"config_restore_model.yaml")
    siamese_net = SiameseNet(restore_model=True)
    siamese_net.restore()
    siamese_net.evaluate()


def make_pairs():
    my_config.load_config(f"config_make_pairs.yaml")
    build_siamese_pairs.make_pairs()


if __name__ == '__main__':
    # DOWNLOAD DATASET
    # get_data.download_dataset("http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz.",
    #                           target_path="data/external/")

    # MAKE SIAMESE PAIRS
    # make_pairs()

    # START EXPERIMENT
    start_experiment()

    # PLOT MEL SPECTROGRAM
    # plot_mel_spectrogram()

    # FEATURE EXTRACTION
    # do_feature_extraction()

    # RESTORE MODEL AND TEST WITH IT
    # test_restored_model()
