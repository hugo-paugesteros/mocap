from pathlib import Path
import shutil
import pandas as pd
from collections import Counter

ROOT = Path("data")
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"

if PROCESSED_DIR.exists():
    print(f"Removing old {PROCESSED_DIR} folder")
    shutil.rmtree(PROCESSED_DIR)

# --- Phase 1 cleaning
print(f"Beginning phase 1 cleaning")

mappings = {
    "Stopani": "Stoppani",
    "longue_note": "Long_note",
    "Tchaikosky": "Tchaikovsky",
    "tchai": "Tchaikovsky",
    "violon_SMD": "Test_player_violin",
    "violon_A": "Klimke",
    "violon_B": "Levaggi",
    "Bach_50bpm": "Bach",
    "Glazounov_90bpm": "Glazounov",
    "Sibelius_55bpm": "Sibelius",
    "note_la_plus_longue": "Long_note",
    "corde_a_vide_beau_son": "Open_string",
    "gamme": "Scale",
    "familiarisation": "Familiarization",
}

RAW_PHASE1_DIR = RAW_DIR / "Phase 1"

# retrieve files in the same ordering as in the "flat" folder
file_list_path = Path(RAW_PHASE1_DIR / "flat/files.txt")
tak_files = [
    RAW_PHASE1_DIR / Path(line)
    for line in file_list_path.read_text().splitlines()
    if line.strip()
]

dataset = []
take_counts = Counter()
for i, tak_file in enumerate(tak_files, 1):
    print(tak_file)
    parts = tak_file.relative_to(RAW_PHASE1_DIR / "experience_SMD_bonne").parts
    violin = parts[0]
    excerpt = parts[-2]

    violin = mappings.get(violin, violin)
    excerpt = mappings.get(excerpt, excerpt)

    take_counts[(violin, excerpt)] += 1
    take_num = take_counts[(violin, excerpt)]

    dst = PROCESSED_DIR / "Phase 1" / violin / excerpt / str(take_num)
    dst.mkdir(exist_ok=True, parents=True)

    # shutil.copy(tak_file, dst / "take.tak")
    shutil.copy(RAW_PHASE1_DIR / f"flat/audio/{i}.wav", dst / "recording.wav")
    shutil.copy(RAW_PHASE1_DIR / f"flat/tracking/{i}.csv", dst / "markers.csv")

    dataset.append(
        {
            "violin": violin,
            "excerpt": excerpt,
            "phase": 1,
            "take": take_num,
            "folder": str(dst.relative_to(PROCESSED_DIR)),
        }
    )

# pd.DataFrame(dataset).to_csv(PROCESSED_DIR / "dataset.csv")


# --- Phase 2 cleaning
print(f"Beginning phase 2 cleaning")

mappings.update(
    {
        "Stopani": "Stoppani",
        "Stopanni": "Stoppani",
        "longue_note": "Long_note",
        "Tchaikosky": "Tchaikovsky",
        "Gamme_chromatique": "Scale",
        "Corde_a_vide_beau_son": "Open_strings",
        "SMD": "Test_player_violin",
    }
)

RAW_PHASE2_DIR = RAW_DIR / "Phase 2"

# retrieve files in the same ordering as in the "flat" folder
csv_files = RAW_PHASE2_DIR.rglob("flat/tracking/*.csv")

take_counts = Counter()
for i, csv_file in enumerate(csv_files, 1):
    parts = csv_file.name.split("$")

    take = parts[0][-1]
    violin = parts[1]
    excerpt = parts[2]
    filename = parts[3]

    violin = mappings.get(violin, violin)
    excerpt = mappings.get(excerpt, excerpt)

    dst = PROCESSED_DIR / "Phase 2" / violin / excerpt / take
    dst.mkdir(exist_ok=True, parents=True)

    dataset.append(
        {
            "violin": violin,
            "excerpt": excerpt,
            "phase": 2,
            "take": take,
            "folder": str(dst.relative_to(PROCESSED_DIR)),
        }
    )

    shutil.copyfile(csv_file, dst / "markers.csv")
    # shutil.copy(
    #     RAW_PHASE2_DIR
    #     / "SMD_exp_2_version_trié"
    #     / Path("/".join(parts)).with_suffix(".tak"),
    #     dst / "take.tak",
    # )

pd.DataFrame(dataset).to_csv(PROCESSED_DIR / "dataset.csv", index=False)
