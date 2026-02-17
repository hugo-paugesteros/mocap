import os
import json
import numpy as np
import librosa
import scipy.signal
import pandas as pd
from pathlib import Path
from mocap.main import MocapTake
import datetime
import soundfile as sf

# Constants
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
LONG_WAV_PATH = RAW_DIR / "Phase 2" / "ZOOM0001.WAV"
AUDIO_SR = 44100
MOCAP_SR = 120
HOP_LENGTH = AUDIO_SR // MOCAP_SR  # 200


def get_take_start_time(take):
    format = "%Y-%m-%d %I.%M.%S.%f %p"
    start_time = datetime.datetime.strptime(take.metadata["capture_start_time"], format)
    return start_time


def get_bowing_velocity(mocap_take):
    df = mocap_take.data
    try:
        pos_x = df[("rigid_body", "archet", "position", "x")].to_numpy()
        pos_y = df[("rigid_body", "archet", "position", "y")].to_numpy()
        pos_z = df[("rigid_body", "archet", "position", "z")].to_numpy()
    except KeyError:
        try:
            pos_x = df[
                ("rigid_body_marker", "bow_frogslide", "position", "x")
            ].to_numpy()
            pos_y = df[
                ("rigid_body_marker", "bow_frogslide", "position", "y")
            ].to_numpy()
            pos_z = df[
                ("rigid_body_marker", "bow_frogslide", "position", "z")
            ].to_numpy()
        except KeyError:
            return None

    # Fill NaNs
    pos_x = pd.Series(pos_x).ffill().bfill().to_numpy()
    pos_y = pd.Series(pos_y).ffill().bfill().to_numpy()
    pos_z = pd.Series(pos_z).ffill().bfill().to_numpy()

    # Compute velocity
    vx = np.diff(pos_x, prepend=pos_x[0])
    vy = np.diff(pos_y, prepend=pos_y[0])
    vz = np.diff(pos_z, prepend=pos_z[0])

    v = np.sqrt(vx**2 + vy**2 + vz**2)
    return v


def compute_feature(signal):
    # RMS or Velocity
    # Normalize
    feat = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
    # Sign of diff (binary-ish)
    feat = np.sign(np.diff(feat, prepend=feat[0]))
    return feat


def align_phase2():
    print("Collecting Phase 2 metadata...")
    dataset_md = pd.read_csv(PROCESSED_DIR / "dataset.csv")
    phase2_md = dataset_md[dataset_md.phase == 2]

    takes = []
    for row in phase2_md.itertuples():
        mocap_path = PROCESSED_DIR / row.folder / "markers.csv"
        mocap_take = MocapTake(mocap_path)
        start_time = get_take_start_time(mocap_take)
        v = get_bowing_velocity(mocap_take)
        if v is not None:
            takes.append(
                {
                    "folder": row.folder,
                    "start_time": start_time,
                    "velocity": v,
                    "duration": len(v) / MOCAP_SR,
                }
            )

    if not takes:
        print("No valid takes found.")
        return

    # Sort takes by time
    takes.sort(key=lambda x: x["start_time"])
    t0 = takes[0]["start_time"]
    for t in takes:
        t["relative_start"] = (t["start_time"] - t0).total_seconds()

    total_span_sec = takes[-1]["relative_start"] + takes[-1]["duration"]
    print(f"Total mocap span: {total_span_sec:.2f}s")

    # Create global mocap feature signal
    # We use a bit more than total span just in case
    global_mocap_feat = np.zeros(int(total_span_sec * MOCAP_SR) + 1)
    for t in takes:
        start_idx = int(t["relative_start"] * MOCAP_SR)
        feat = compute_feature(t["velocity"])
        end_idx = start_idx + len(feat)
        if end_idx > len(global_mocap_feat):
            # Extend if necessary (shouldn't happen with total_span_sec)
            padding = end_idx - len(global_mocap_feat)
            global_mocap_feat = np.pad(global_mocap_feat, (0, padding))
        global_mocap_feat[start_idx:end_idx] = feat

    # Load Audio
    print(f"Loading {LONG_WAV_PATH}...")
    # Use sr=None to keep original 48kHz if possible, or force to AUDIO_SR
    y, sr = librosa.load(LONG_WAV_PATH, sr=AUDIO_SR)
    print(f"Audio duration: {len(y) / sr:.2f}s")

    # Compute Audio feature
    print("Computing audio features...")
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH, center=True)[0]
    audio_feat = compute_feature(rms)

    # Correlation
    print("Correlating global signal...")
    correlation = scipy.signal.correlate(
        audio_feat, global_mocap_feat, mode="full", method="fft"
    )
    lags = scipy.signal.correlation_lags(
        len(audio_feat), len(global_mocap_feat), mode="full"
    )

    best_idx = np.argmax(correlation)
    best_lag = lags[best_idx]
    max_corr = correlation[best_idx]

    # Confidence: Peak vs Second Peak (excluding immediate neighborhood of best peak)
    # We exclude a 2-second window around the best peak
    neighbor_size = 2 * MOCAP_SR
    search_corr = correlation.copy()
    search_corr[
        max(0, best_idx - neighbor_size) : min(
            len(correlation), best_idx + neighbor_size
        )
    ] = -np.inf
    second_best_idx = np.argmax(search_corr)
    second_max_corr = search_corr[second_best_idx]
    confidence = max_corr / (second_max_corr + 1e-9)

    global_offset_sec = best_lag / MOCAP_SR
    print(
        f"Global offset: {global_offset_sec:.3f}s (Score: {max_corr:.1f}, Confidence: {confidence:.2f})"
    )

    # Results
    results = []
    # Refine each take individually within a small window (+/- 2 seconds)
    # to account for the second-level precision of metadata
    REFINE_WINDOW_SEC = 2.0
    REFINE_WINDOW_SAMPLES = int(REFINE_WINDOW_SEC * MOCAP_SR)

    for t in takes:
        coarse_start_in_audio = global_offset_sec + t["relative_start"]

        # Define search range in audio_feat
        # We need to find the best local lag for THIS take's feature
        t_feat = compute_feature(t["velocity"])

        # Center of search in audio_feat index
        center_idx = int(coarse_start_in_audio * MOCAP_SR)

        audio_dur_samples = len(audio_feat)
        status = "OK"
        refined_start_in_audio = coarse_start_in_audio
        take_confidence = 0.0

        if center_idx < 0 or center_idx + len(t_feat) > audio_dur_samples:
            status = "OUT_OF_BOUNDS"
        else:
            # Local correlation refinement
            start_search = max(0, center_idx - REFINE_WINDOW_SAMPLES)
            end_search = min(
                audio_dur_samples, center_idx + len(t_feat) + REFINE_WINDOW_SAMPLES
            )
            audio_segment = audio_feat[start_search:end_search]

            local_corr = scipy.signal.correlate(
                audio_segment, t_feat, mode="valid", method="fft"
            )
            local_lags = np.arange(len(local_corr))

            if len(local_corr) > 0:
                best_local_idx = np.argmax(local_corr)
                # The lag is relative to start_search
                refined_sample_idx = start_search + best_local_idx
                refined_start_in_audio = refined_sample_idx / MOCAP_SR

                # Local confidence
                local_max = local_corr[best_local_idx]
                local_search = local_corr.copy()
                # Exclude 0.5s around peak
                loc_neighbor = int(0.5 * MOCAP_SR)
                local_search[
                    max(0, best_local_idx - loc_neighbor) : min(
                        len(local_corr), best_local_idx + loc_neighbor
                    )
                ] = -np.inf
                if np.any(local_search > -np.inf):
                    local_second = np.max(local_search)
                    take_confidence = local_max / (local_second + 1e-9)
                else:
                    take_confidence = 1.0  # Only one peak possible in window

        end_in_audio = refined_start_in_audio + t["duration"]

        results.append(
            {
                "folder": t["folder"],
                "start_time_audio": float(refined_start_in_audio),
                "end_time_audio": float(end_in_audio),
                "confidence": float(take_confidence),
                "status": status,
            }
        )

        if status == "OK":
            print(
                f"  {t['folder']}: {refined_start_in_audio:.2f}s (Conf: {take_confidence:.2f})"
            )
            # Extract and save
            start_sample = int(refined_start_in_audio * sr)
            end_sample = int(end_in_audio * sr)
            y_segment = y[start_sample:end_sample]
            out_path = PROCESSED_DIR / t["folder"] / "recording.wav"
            sf.write(out_path, y_segment, sr)
        else:
            print(f"  {t['folder']}: {status}")


if __name__ == "__main__":
    align_phase2()
