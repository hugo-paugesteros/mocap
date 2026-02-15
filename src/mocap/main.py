import pandas as pd
from pathlib import Path
import csv


class Take:
    def __init__(self, row):
        pass

    def audio():
        pass

    def mocap():
        pass


class MocapTake:
    def __init__(self, file):
        self.file_path = Path(file)

        self.metadata = {}

        self._load_metadata()
        self.data = pd.read_csv(self.file_path, header=[0, 1, 2, 3], skiprows=1)

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


# metadata = pd.read_csv("data/processed/dataset.csv")
# mocap_take = MocapTake(
#     Path("data/processed") / Path(metadata.iloc[0].folder) / "markers.csv"
# )
# print(mocap_take.data)
# take.audio
# take.mocap.markers
# take.mocap.data
