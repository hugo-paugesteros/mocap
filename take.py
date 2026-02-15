import pandas as pd
import numpy as np
from dtw import *
from mappings import RENAME_MAPPING
import librosa
from scipy.interpolate import interp1d


class Take:
    aligned_xs = None

    def __init__(self, filename, start=None, end=None):
        self.filename = filename
        self._parse(self.filename)
        # self._clean_data()
        # self._cut(start, end)
        self._projection()
        self.warper_func = None
        # self.new_time = np.arange(len(self.df_time))
        self.validity = np.arange(1000, len(self.df_time))

    def _parse(self, filename):
        df = pd.read_csv(filename, header=[1, 2, 3, 5])

        # Drop the first two columns (Frame, Time) if they are just indices/metadata
        # In Take.py: df = df.iloc[:, 2:]
        df_markers = df.iloc[:, 2:]

        # Rename columns using the mapping
        df_markers.columns.names = ["Type", "Marker", "", "Data"]
        df_markers = df_markers.rename(columns=RENAME_MAPPING, level=1)

        # Drop level 2 (empty level in the multi-index)
        # In Take.py: df.columns = df.columns.droplevel(2)
        df_markers.columns = df_markers.columns.droplevel(2)

        # Read time data separately to keep it clean
        # In Take.py: df_time = pd.read_csv(filename, skiprows=6, usecols=[0, 1])
        df_time = pd.read_csv(filename, skiprows=6, usecols=[0, 1])

        # Create MultiIndex for df_time to match df_markers structure
        df_time.columns = pd.MultiIndex.from_tuples(
            [("Time", "", col) for col in df_time.columns],
            names=["Type", "Marker", "Data"],
        )

        # Concatenate Time/Frame with Marker Data
        # This makes the CSV self-contained and ready for analysis
        df_final = pd.concat([df_time, df_markers], axis=1)
        self.df = df_final
        self.df_time = df_time

        ####
        # self.df = pd.read_csv(filename, header=[0, 1, 2])
        # self.df_time = self.df["Time"].droplevel(0, axis=1)

    def _clean_data(self, threshold=200 / 60):
        markers = self.df["Rigid Body Marker"].columns.get_level_values(0).unique()
        for marker in markers:
            marker_data = self.df["Rigid Body Marker"][marker]
            if not all(col in marker_data.columns for col in ["X", "Y", "Z"]):
                continue

            displacement = np.linalg.norm(
                marker_data.diff()[["X", "Y", "Z"]].to_numpy(), axis=1
            )
            is_artifact = np.nan_to_num(displacement) > threshold
            print(is_artifact.sum())

            if np.any(is_artifact):
                self.df.loc[is_artifact, ("Rigid Body Marker", marker)] = np.nan

        self.df.interpolate(
            method="linear", axis=0, limit_direction="both", inplace=True
        )

    def _cut(self, start, end):
        valid = np.abs(self.compute_hairstring()) < 20
        start = valid.argmax()
        end = valid.size - valid[::-1].argmax()

        self.df = self.df.iloc[start:end].reset_index()
        self.df_time = self.df_time.iloc[start:end].reset_index()
        self.df_time["Frame"] = np.arange(len(self.df))

    def _projection(self):
        hairFrog = self.df["Rigid Body Marker"]["bow:HairFrog"].to_numpy()
        hairTip = self.df["Rigid Body Marker"]["bow:HairTip"].to_numpy()

        stringBridge = self.df["Rigid Body Marker"]["violin:GstringBridge"].to_numpy()
        stringNut = self.df["Rigid Body Marker"]["violin:GstringNut"].to_numpy()

        self.closest_points = self._compute_closest_points(
            (hairFrog, hairTip), (stringBridge, stringNut)
        )

    def compute_xs(self):
        hairFrog = self.df["Rigid Body Marker"]["bow:HairFrog"]
        xs = np.linalg.norm(hairFrog - self.closest_points, axis=1)
        return xs

    def compute_vs(self):
        xs = self.compute_xs()
        vs = np.diff(xs, append=0)
        return vs

    def compute_beta(self):
        stringBridge = self.df["Rigid Body Marker"]["violin:GstringBridge"]
        beta = np.linalg.norm(stringBridge - self.closest_points, axis=1)
        return beta

    def compute_hairstring(self):
        hairTip = self.df["Rigid Body Marker"]["bow:HairTip"].to_numpy()
        hairFrog = self.df["Rigid Body Marker"]["bow:HairFrog"].to_numpy()
        stringBridge = self.df["Rigid Body Marker"]["violin:GstringBridge"].to_numpy()
        stringNut = self.df["Rigid Body Marker"]["violin:GstringNut"].to_numpy()
        v_plane = np.cross(hairTip - hairFrog, stringBridge - stringNut)

        dist_hairstring = np.sum(
            (hairFrog - stringNut) * v_plane, axis=1
        ) / np.linalg.norm(v_plane, axis=-1)
        return dist_hairstring

    def compute_tilt(self):
        hairTip = self.df["Rigid Body Marker"]["bow:HairTip"].to_numpy()
        hairFrog = self.df["Rigid Body Marker"]["bow:HairFrog"].to_numpy()
        lowerStick = self.df["Rigid Body Marker"]["bow:LowerStickLeft"].to_numpy()
        stringBridge = self.df["Rigid Body Marker"]["violin:GstringBridge"].to_numpy()
        stringNut = self.df["Rigid Body Marker"]["violin:GstringNut"].to_numpy()

        y_v = stringNut - stringBridge
        y_v /= np.linalg.norm(y_v, axis=1, keepdims=True)

        x_b = hairFrog - hairTip
        x_b /= np.linalg.norm(x_b, axis=1, keepdims=True)

        y_b = np.cross(x_b, lowerStick - hairFrog)
        y_b /= np.linalg.norm(y_b, axis=1, keepdims=True)

        # Un jour je regarderai le sens géométrique de ces trucs.
        tmp = np.cross(x_b, y_v)
        tmp2 = np.sum(tmp * y_b, axis=1)

        tilt = (np.arccos(tmp2) - np.pi / 2) * 360 / (2 * np.pi)
        return tilt

    def compute_skewness(self):
        hairTip = self.df["Rigid Body Marker"]["bow:HairTip"].to_numpy()
        hairFrog = self.df["Rigid Body Marker"]["bow:HairFrog"].to_numpy()
        stringBridge = self.df["Rigid Body Marker"]["violin:GstringBridge"].to_numpy()
        stringNut = self.df["Rigid Body Marker"]["violin:GstringNut"].to_numpy()

        y_v = stringNut - stringBridge
        y_v /= np.linalg.norm(y_v, axis=1, keepdims=True)

        x_b = hairFrog - hairTip
        x_b /= np.linalg.norm(x_b, axis=1, keepdims=True)

        # Un jour je regarderai le sens géométrique de ces trucs.
        tmp = np.sum(y_v * x_b, axis=1)

        skewness = (np.pi / 2 - np.arccos(tmp)) * 360 / (2 * np.pi)
        return skewness

    def compute_inclination(self):
        hairTip = self.df["Rigid Body Marker"]["bow:HairTip"].to_numpy()
        hairFrog = self.df["Rigid Body Marker"]["bow:HairFrog"].to_numpy()
        stringBridge = self.df["Rigid Body Marker"]["violin:GstringBridge"].to_numpy()
        stringNut = self.df["Rigid Body Marker"]["violin:GstringNut"].to_numpy()
        LBDist = self.df["Rigid Body Marker"]["violin:LBDist"].to_numpy()
        RBDist = self.df["Rigid Body Marker"]["violin:RBDist"].to_numpy()

        x_v = LBDist - RBDist
        x_v /= np.linalg.norm(x_v, axis=1, keepdims=True)

        y_v = stringNut - stringBridge
        y_v /= np.linalg.norm(y_v, axis=1, keepdims=True)

        x_b = hairFrog - hairTip
        x_b /= np.linalg.norm(x_b, axis=1, keepdims=True)

        z_v = np.cross(x_v, y_v)

        # Un jour je regarderai le sens géométrique de ces trucs.
        tmp = np.sum(z_v * x_b, axis=1)
        inclination = (np.pi / 2 - np.arccos(tmp)) * 360 / (2 * np.pi)
        return inclination

    def align(self, base_take):
        base_vs = base_take.compute_vs()
        base_vs = np.nan_to_num(base_vs)

        self_vs = self.compute_vs()
        self_vs = np.nan_to_num(self_vs)

        # 2. Run DTW
        # We align Self (Query) to Base (Reference)
        alignment = dtw(
            x=self_vs,  # Query (index1)
            y=base_vs,  # Reference (index2)
            keep_internals=True,
            step_pattern="asymmetric",
            open_end=True,
            open_begin=True,
        )

        # 3. Create the Mapping Function IMMEDIATELY
        # We need a function: f(Time_Base) -> Time_Self

        # index2 is Time_Base (from y argument)
        # index1 is Time_Self (from x argument)
        index_base = alignment.index2
        index_self = alignment.index1

        # Handle Singularities:
        # Multiple 'self' points might map to one 'base' point, or vice versa.
        # We need unique points on the x-axis (Time_Base) to build an interpolator.
        unique_base_frames, unique_indices = np.unique(index_base, return_index=True)
        corresponding_self_frames = index_self[unique_indices]

        # Store the interpolator
        # This function takes a time t in Base, and gives the float time t' in Self
        self.warper_func = interp1d(
            unique_base_frames,
            corresponding_self_frames,
            kind="linear",
            fill_value="extrapolate",
        )

        self.reference_len = len(base_vs)

    def warp(self, feature_array):
        if self.warper_func is None:
            return feature_array
        # 1. Define the grid of the Reference
        ref_grid = np.arange(self.reference_len)

        # 2. Find where to look in 'Self' for each point in 'Reference'
        # (This uses the pre-computed alignment)
        target_indices = self.warper_func(ref_grid)

        # 3. Interpolate the Feature values
        # We create a temporary interpolator for the feature data we just received
        feature_interpolator = interp1d(
            np.arange(len(feature_array)),
            feature_array,
            kind="linear",
            fill_value=(np.nan, np.nan),  # Handle slight edge mismatches
            bounds_error=False,
        )

        # 4. Sample the feature at the calculated indices
        warped_feature = feature_interpolator(target_indices)

        return warped_feature

    def _compute_closest_points(self, line1, line2):
        point1A, point1B = line1
        point2A, point2B = line2

        U1 = np.array(point1B - point1A)
        # U1 /= np.linalg.norm(U1, axis=1, keepdims=True)
        U2 = np.array(point2B - point2A)
        # U2 /= np.linalg.norm(U2, axis=1, keepdims=True)

        UC = np.cross(U2, U1)
        # UC /= np.linalg.norm(UC, axis=1, keepdims=True)

        RHS = np.array(point2A - point1A)
        RHS = np.array(RHS)[:, :, np.newaxis]
        LHS = np.array([U1, -U2, UC]).T.swapaxes(0, 1)

        ts = np.linalg.solve(LHS, RHS).swapaxes(0, 1)

        Q1 = point1A + ts[0, :, :] * U1
        Q2 = point2A + ts[1, :, :] * U2

        self.Q1 = pd.DataFrame(Q1, columns=["X", "Y", "Z"])
        self.Q2 = pd.DataFrame(Q2, columns=["X", "Y", "Z"])

        for col in ["X", "Y", "Z"]:
            self.df[("Rigid Body Marker", "line:Q1", col)] = self.Q1[col]
            self.df[("Rigid Body Marker", "line:Q2", col)] = self.Q2[col]

        return point1A + ts[0, :, :] * U1

    def _rename_columns(self, df):
        df.columns.names = ["Type", "Marker", "", "Data"]
        return df.rename(columns=RENAME_MAPPING, level=1)
