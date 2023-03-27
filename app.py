import typer
import src.visualization.visualize as visualize
import src.data.build_siamese_pairs as build_siamese_pairs
from typing import List

app = typer.Typer()


@app.command()
def plot(
        type: str = typer.Option(
            'wave', "--type", "-t", prompt="Feature type: wav, stft, mfcc, mel"
        )
) -> None:
    """Plot extracted features"""
    visualize.plot_data_and_save(img_type=type, words_no=1, files_no=1)


@app.command()
def build_pairs(
        exclude: List[str] = typer.Argument(None,
                                            help="You can exclude multiple words from making pairs. "
                                                 "Usage ex.: python app.py build-pairs bird happy"),
        data_path: str = typer.Option(
            "data/external/speech_commands_v0.01/", "--dataPath", "-d", help="path to the dataset"),
        txt_file: str = typer.Option(
            "data/external/speech_commands_v0.01/train_list.txt", "--txtFile", "-t",
            help=".txt file containing the audio file paths that belongs to one set (train/val)"
        ),
        csv_path: str = typer.Option(
            "reports/pairs.csv", "--csvPath", "-c", help="path to save csv"
        )
):
    """
    Build pairs for siamese network, save as .csv file
    """
    for folder in exclude:
        print(f"{folder} excluded.")
    build_siamese_pairs.make_pairs(data_path=data_path, txt_file=txt_file, exclude=exclude, csv_path=csv_path)


@app.command()
def build_features():
    pass


@app.command()
def evaluate():
    pass


if __name__ == "__main__":
    app()
