import pandas as pd
from pathlib import Path
from mappings import RENAME_MAPPING


data_path = Path("data/processed/")


def clean(row):
    """
    http://wiki.optitrack.com/index.php?title=Data_Export:_CSV
    """

    input_file = data_path / row.folder / "markers.csv"
    print(input_file)

    # 1. Capture the metadata lines
    with open(input_file, "r") as f:
        metadata_line = f.readline()
        empty_line = f.readline()

    # Read the first 7 lines to handle the complex header
    # Line 0: Format/Metadata
    # Line 1: Empty
    # Line 2: Type (Rigid Body, Rigid Body Marker, Marker)
    # Line 3: Name
    # Line 4: ID
    # Line 5: Data Type (Rotation, Position)
    # Line 6: Axis (Frame, Time, X, Y, Z)

    # 2. Read the data part first to get the columns
    # We use header=[2, 3, 4, 5, 6] to get the relevant header levels
    # We'll handle Line 0 and 1 separately if needed, but for sharing,
    # we can often start from the Type line or keep the first line.

    df = pd.read_csv(
        input_file,
        header=[2, 3, 4, 5, 6],
        skip_blank_lines=False,
    )

    # Levels:
    # 0: Type
    # 1: Name
    # 2: ID
    # 3: Data Type
    # 4: Axis

    # 3. Filter: Keep only 'Rigid Body' and 'Rigid Body Marker'
    # We also keep 'Unnamed' columns which usually contain Frame and Time
    cols_to_keep = [
        col
        for col in df.columns
        if col[0] in ["Rigid Body", "Rigid Body Marker", "Type"] or "Unnamed" in col[0]
    ]
    df = df[cols_to_keep]

    # 4. Rename the names in Level 1 and remove Level 2 (ID)
    new_tuples = []
    for col in df.columns:
        c_type, c_name, c_id, c_data, c_axis = col

        # Replace "Unnamed" strings with empty strings to match raw file style
        c_type = "" if "Unnamed" in str(c_type) else str(c_type)
        c_name = "" if "Unnamed" in str(c_name) else str(c_name)
        c_data = "" if "Unnamed" in str(c_data) else str(c_data)

        # Apply renaming mapping to the Name level
        if c_name:
            c_name = RENAME_MAPPING.get(c_name, c_name)

        # Construct new tuple without ID (Level 2)
        # We'll keep: Type, Name, Data Type, Axis
        new_tuples.append((c_type, c_name, c_data, c_axis))

    df.columns = pd.MultiIndex.from_tuples(new_tuples)

    # 5. Save to CSV
    # To keep it close to raw, we might want to manually prepend the first two lines
    # but often a clean 4-level header is better for sharing.
    print(input_file.parent / "markers.csv")
    with open(input_file.parent / "markers.csv", "w") as f:
        f.write(metadata_line)
        f.write(empty_line)
        df.to_csv(f, index=False)


if __name__ == "__main__":
    dataset_md = pd.read_csv("data/processed/dataset.csv")
    for row in dataset_md.itertuples():
        clean_csv = clean(row)
