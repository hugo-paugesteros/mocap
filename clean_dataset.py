import glob
from pathlib import Path
import shutil
import os
import pandas as pd

shutil.rmtree("Processed")

dataset = []

# --- Phase 1
files = glob.glob("Raw/Phase 1/experience_SMD_bonne/**/*.tak", recursive=True)

mappings = {
    "Stopani": "Stoppani",
    "longue_note": "Note_longe",
    "Tchaikosky": "Tchaikovsky",
    "violon_SMD": "SMD",
    "violon_A": "Klimke",
    "violon_B": "Stoppani",
    "Bach_50bpm": "Bach",
    "Glazounov_90bpm": "Glazounov",
    "Sibelius_55bpm": "Sibelius",
    "note_la_plus_longue": "Note_longue",
    "corde_a_vide_beau_son": "Corde_a_vide_beau_son",
    "gamme": "Gamme",
    "familiarisation": "Familiarisation",
}

for i, file in enumerate(files):
    parts = file.split("/")[3:]

    violin = parts[0]
    take = 1 if parts[1] != "second" else 2
    excerpt = parts[-2]
    filename = parts[-1]

    violin = mappings.get(violin, violin)
    excerpt = mappings.get(excerpt, excerpt)

    dataset.append(
        {
            "take_file": file,
            "csv_file": Path(f"Raw/Phase 1/flat/tracking/{i + 1}.csv"),
            "wav_file": Path(f"Raw/Phase 1/flat/audio/{i + 1}.wav"),
            "violin": violin,
            "excerpt": excerpt,
            "phase": 1,
        }
    )

dataset = pd.DataFrame(dataset)
dataset = dataset.sort_values(by=["take_file"])
dataset["take"] = dataset.groupby(["violin", "excerpt"]).cumcount() + 1
dataset["folder"] = ""

for row in dataset.itertuples():
    dst = Path(f"Processed/Phase 1/{row.violin}/{row.excerpt}/{row.take}/")
    dst.mkdir(exist_ok=True, parents=True)
    shutil.copy(row.take_file, dst / "take.tak")
    shutil.copy(row.wav_file, dst / "mocap_audio.wav")
    shutil.copy(row.csv_file, dst / "markers.csv")
    dataset.at[row.Index, "folder"] = dst.resolve()

dataset = dataset.drop(columns=["take_file", "csv_file", "wav_file"])
dataset.to_csv("dataset.csv")


# --- Phase 2
dataset_phase2 = []
files = Path("/home/hugo/Thèse/Data/CNSM/Mocap/Raw/Phase 2/flat_csv/").glob("*.csv")
# takes = list(
#     glob.glob(
#         "/home/hugo/Thèse/Data/CNSM/Mocap/Raw/Phase 2/SMD_exp_2_version_trié/**/*.tak",
#         recursive=True,
#     )
# )

mappings = {
    "Stopani": "Stoppani",
    "Stopanni": "Stoppani",
    "longue_note": "Note_longe",
    "Tchaikosky": "Tchaikovsky",
    "Gamme_chromatique": "Gamme",
}

for i, file in enumerate(files):
    parts = file.name.split("$")

    prise = parts[0][-1]
    violin = parts[1]
    excerpt = parts[2]
    filename = parts[3]

    violin = mappings.get(violin, violin)
    excerpt = mappings.get(excerpt, excerpt)

    output_file = Path(f"Processed/Phase 2/{violin}/{excerpt}/{prise}/file.txt")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    dataset_phase2.append(
        {
            "original_take_file": file,
            "csv_file": Path(f"Raw/Phase 1/flat/tracking/{i + 1}.csv"),
            "wav_file": Path(f"Raw/Phase 1/flat/audio/{i + 1}.wav"),
            "violin": violin,
            "excerpt": excerpt,
            "phase": 2,
        }
    )

    dst = f"{output_file.parent}/{filename[:-3]}csv"
    shutil.copyfile(file, dst)

    src = Path("Raw/Phase 2/SMD_exp_2_version_trié") / Path(
        "/".join(parts)
    ).with_suffix(".tak")
    src = src
    dst = f"{output_file.parent}/{filename[:-3]}tak"
    shutil.copyfile(src, dst)
