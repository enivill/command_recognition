from matplotlib import use as mpl_use

# import src.features.build_siamese_pairs
# import src.features.build_features
import src.data.build_siamese_pairs
from src.data import preprocessing
from src.helper import load_config

mpl_use('TkAgg')

if __name__ == '__main__':
    # src.features.build_features.mel("data/external/speech_commands_v0.01/bed/b2fbe484_nohash_0.wav")
    # CREATE FEATURE
    # build_features.feature_extraction(feature_type='mfcc', data_path='data/external/speech_commands_v0.01/', files_no=100)

    # TRAIN MODEL
    # base_network = train_model.initialize_base_network()
    # plot_model(base_network, show_shapes=True, show_layer_names=True, to_file='base-model.png')
    # train_model.siamese_network(base_network)
    #
    # print(f"Pair train length: {(len(pairTrain))}")
    # print(f"label 0 count: {np.count_nonzero(labelTrain == 0)}")
    # print(f"label 1 count: {np.count_nonzero(labelTrain == 1)}")

    # src.models.train_model.siamese_network()

    # outliers = src.helper.find_dataset_audio_duration_outliers(dataset_path="data/external/speech_commands_v0.01",
    #                                                            normal_value=1.0)
    # print(sorted(outliers))
    # print("LENGTH:")
    # print(len(outliers))

    # df = read_csv("reports/pairs.csv", delimiter=';')
    # train_a = df['audio_1'].to_numpy()
    # train_b = df['audio_2'].to_numpy()
    # labels = df['label'].to_numpy()
    #
    # train_a_feature = src.features.build_features.feature_extraction('', train_a[:6],
    #                                                                  "data/external/speech_commands_v0.01/")
    # print(train_a_feature)
    # print(train_a_feature.shape)  # (102176, 40, 126)
    # print(train_a.shape)  # (102176,)
    # src.models.train_model2.train_model()
    # preprocessing.test()

    # src.data.preprocessing.split_data('config.yaml')
    # src.data.build_siamese_pairs.make_pairs('config.yaml')
    pass
