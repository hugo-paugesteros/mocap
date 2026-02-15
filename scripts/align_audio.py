import pandas as pd
from pathlib import Path
import datetime
import librosa
import numpy as np
import scipy.signal
import soundfile as sf
from mocap.main import MocapTake

phase1_recordings = [
    Path("data/raw/Phase 1/experience_SMD_bonne/Audio_file/241015_165642.wav"),
    Path("data/raw/Phase 1/experience_SMD_bonne/Audio_file/241015_175713.wav"),
]
data_path = Path("data/processed/")


def get_recording_limits(path):
    format = "%y%m%d_%H%M%S"
    start = path.stem[-13:]
    start_time = datetime.datetime.strptime(start, format)

    duration = np.round(librosa.get_duration(path=path))
    end_time = start_time + datetime.timedelta(seconds=duration)

    return (start_time, end_time)


def get_take_start_time(take):
    format = "%Y-%m-%d %I.%M.%S.%f %p"
    start_time = datetime.datetime.strptime(take.metadata["capture_start_time"], format)
    return start_time


def is_in_limits(start_time, limits):
    limit_start, limit_end = limits
    if (start_time > limit_start) & (start_time < limit_end):
        return True
    return False


scores = []


def correlate(row, recording, expected_offset=None):
    SR = 5000
    MARGIN = 10
    y_small, sr = librosa.load(
        Path("data/processed") / row.folder / "mocap_audio.wav", sr=SR
    )
    if expected_offset:
        start_search = max(0, expected_offset - MARGIN)
        duration = len(y_small) / SR + MARGIN
        y_long, sr = librosa.load(
            recording,
            sr=SR,
            offset=start_search,
            duration=duration,
        )
        search_start_sample = int(start_search * SR)
    else:
        y_long, sr = librosa.load(recording, sr=SR)
        search_start_sample = 0

    # Normalization
    y_long_norm = y_long - np.mean(y_long) / (np.std(y_long) + 1e-9)
    y_small_norm = y_small - np.mean(y_small) / (np.std(y_small) + 1e-9)

    # Cross-correlation
    # Use FFT based correlation for speed
    # mode='valid' means y_small is fully overlapped by y_long
    correlation = scipy.signal.correlate(
        y_long_norm, y_small_norm, mode="valid", method="fft"
    )

    # Find peak
    peak_index = np.argmax(correlation)

    y_long_slice = y_long_norm[peak_index : peak_index + len(y_small)]
    energy_long_slice = np.sum(y_long_slice**2)
    energy_small = np.sum(y_small_norm**2)
    max_corr_val = correlation[peak_index] / np.sqrt(energy_long_slice * energy_small)

    scores.append(max_corr_val)

    # Calculate time
    start_sample = peak_index + search_start_sample
    end_sample = start_sample + len(y_small)
    start_time = start_sample / sr
    end_time = end_sample / sr

    print(
        f"  Match found at {start_time:.2f}s - {end_time:.2f}s (Score: {max_corr_val:.2f})"
    )

    return (start_time, end_time)


if __name__ == "__main__":
    dataset_md = pd.read_csv("data/processed/dataset.csv")

    # --- Phase 1
    phase1_md = dataset_md[dataset_md.phase == 1]
    limits = [get_recording_limits(recording) for recording in phase1_recordings]

    for row in phase1_md.itertuples():
        print(row.folder)
        mocap_take = MocapTake(Path("data/processed") / row.folder / "markers.csv")
        start_time = get_take_start_time(mocap_take)

        for recording, limit in zip(phase1_recordings, limits):
            if is_in_limits(start_time, limit):
                offset = (start_time - limit[0]).total_seconds()
                start_time, end_time = correlate(row, recording, expected_offset=offset)
                y, sr = librosa.load(
                    recording,
                    sr=None,
                    offset=start_time,
                    duration=end_time - start_time,
                )
                sf.write(
                    Path("data/processed") / row.folder / "recorder_audio.wav",
                    y,
                    sr,
                )
                break
        else:
            print(f"No recording found for this take: {row}")
