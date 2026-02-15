import pandas as pd
from pathlib import Path
import csv
import librosa


@pd.api.extensions.register_series_accessor("mocap")
class MocapAccessor:
    """
    A custom accessor for pandas Series (i.e., DataFrame rows) that lets you
    easily convert a row from your metadata CSV into a fully-featured `Take` object.
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if "folder" not in obj.index:
            raise AttributeError("The Series must have a 'folder' attribute.")

    def to_take(self, root_path="data/processed"):
        """
        Brings the row "to life" by creating a Take object from it.

        Args:
            root_path (str or Path): The path to the root of the processed dataset.

        Returns:
            Take: An initialized Take object ready for analysis.
        """
        return Take(self._obj, root=root_path)


class Take:
    def __init__(self, row, root="/data/processed"):
        self.folder = Path(row.folder)
        self.metadata = row
        self.root = Path(root)
        self.mocap = MocapTake(self.root / self.folder / "markers.csv")

    @property
    def audio(self):
        path = self.root / self.folder / "recording.wav"
        if path.exists():
            return librosa.load(path, sr=None)
        return None, None


class MocapTake:
    def __init__(self, file):
        self.file_path = Path(file)

        self.metadata = {}

        self._load_metadata()
        self.data = pd.read_csv(
            self.file_path, header=[0, 1, 2, 3], skiprows=1, skip_blank_lines=True
        )

        self.data.set_index([self.data.columns[0], self.data.columns[1]], inplace=True)
        self.data.index.names = ["Frame", "Time"]

        self.data.columns = self.data.columns.map(
            lambda x: tuple(
                str(level).lower().replace(" ", "_").replace(":", "_") for level in x
            )
        )

    def _load_metadata(self):
        with open(self.file_path, "r") as f:
            reader = csv.reader(f)
            first_line = next(reader)

            # The row is [Key1, Val1, Key2, Val2, ...]
            # We use zip to pair them up into a dictionary
            keys = first_line[0::2]
            values = first_line[1::2]

            # Clean keys to be attribute-friendly (lowercase, no spaces)
            self.metadata = {
                k.lower().replace(" ", "_"): v
                for k, v in zip(keys, values)
                if k.strip()
            }

    def __getattr__(self, name):
        if name in self.metadata:
            return self.metadata[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    @property
    def rigid_body(self):
        return self.data.rigid_body

    @property
    def rigid_body_marker(self):
        return self.data.rigid_body_marker


if __name__ == "__main__":
    metadata = pd.read_csv("data/processed/dataset.csv")
    take = metadata.iloc[0].mocap.to_take(root_path="data/processed")
    print(take.audio)
    print(take.mocap.metadata)
    print(take.mocap.data)
