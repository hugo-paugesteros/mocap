import glob
from pathlib import Path
import shutil

files = glob.glob("Phase 1/RAW/experience_SMD_bonne/**/*.tak", recursive=True)
print(files)

mappings = {
    "Stopani": "Stopanni",
    "longue_note": "Note_longe",
    "Tchaikosky": "Tchaikovsky",
    "violon_SMD": "SMD",
    "violon_A": "Klimke",
    "violon_B": "Stopanni",
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

    output_file = Path(f"Phase 1/{violin}/{excerpt}/{take}/file.txt")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    src = f"Phase 1/RAW/flat/audio/{i+1}.wav"
    dst = f"{output_file.parent}/{filename[:-3]}wav"

    # print(dst)

    shutil.copyfile(src, dst)
