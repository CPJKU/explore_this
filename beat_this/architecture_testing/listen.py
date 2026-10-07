from pathlib import Path

import IPython.display
import numpy as np
from typing import NamedTuple

import librosa
import soundfile as sf

from beat_this.model.beat_tracker import ModelOutput


class MonoAudio(NamedTuple):
    audio: np.ndarray
    sr: float

    @classmethod
    def from_path(cls, path: str | Path):
        audio, sr = sf.read(path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        return cls(audio=audio, sr=sr)

    def get_window(self, start_frame: int, window_size=1500, fps: float = 50):
        # sr is audio frames per second
        # fps is spect frames per second
        # start_frame is measure in spect frame
        # start_frame * (audio_frames / spect_frame) gets at start frame in audio
        audio_frames_per_spect_frame = self.sr / fps
        start_frame_audio = int(start_frame * audio_frames_per_spect_frame)
        window_size_audio = int(window_size * audio_frames_per_spect_frame)

        return MonoAudio(
            audio=self.audio[start_frame_audio : start_frame_audio + window_size_audio],
            sr=self.sr,
        )

    def __len__(self):
        return len(self.audio)


def play_audio_with_multi_layer_clicks(
    audio: MonoAudio,
    layers: list[tuple[np.ndarray, float, float]],
    metronome_volume=1.0,
):
    """
    layers is a list of tuples (times, click_freq, click_volume)
    """
    used_times: np.ndarray = np.array([])
    total_audio = audio.audio

    for layer_times, click_freq, click_volume in layers:
        non_overlapping_times = [
            t
            for t in layer_times
            if not np.any((used_times > t - 0.04) & (used_times < t + 0.04))
        ]

        used_times = np.concatenate((used_times, np.array(non_overlapping_times)))

        audio_layer = librosa.clicks(
            times=non_overlapping_times,
            sr=audio.sr,
            click_freq=click_freq,
            length=len(audio),
        )

        total_audio = total_audio + click_volume * audio_layer

    IPython.display.display(
        IPython.display.Audio(
            total_audio * metronome_volume,
            rate=audio.sr,
        )
    )


def play_audio_with_clicks(
    audio: MonoAudio,
    beat_times: np.ndarray,
    downbeat_times: np.ndarray | None = None,
    metronome_volume=1.0,
):
    if downbeat_times is None:
        play_audio_with_multi_layer_clicks(
            audio,
            [(beat_times, 1000, 1.0)],
            metronome_volume=metronome_volume,
        )
        return

    play_audio_with_multi_layer_clicks(
        audio,
        [
            (downbeat_times, 1500, 1.0),
            (beat_times, 1000, 0.5),
        ],
        metronome_volume=metronome_volume,
    )


def listen_to_annotations(
    spect_path: str,
    start_frame: int | None = None,
    window_size_frames: int | None = 1500,
    fps: float | None = 50,
    **kwargs,
):
    audio_path, annotation_path = get_audio_and_annotation_paths(spect_path)

    audio = MonoAudio.from_path(audio_path)
    beat_annotation = np.loadtxt(annotation_path)

    if beat_annotation.ndim == 2:
        beats = beat_annotation[:, 0]
        beat_value = beat_annotation[:, 1].astype(int)
    else:
        beats = beat_annotation
        beat_value = np.zeros_like(beats, dtype=np.int32)

    downbeats = beats[beat_value == 1]

    if start_frame is not None:
        assert fps is not None and window_size_frames is not None

        start_time = start_frame / fps
        window_size_seconds = window_size_frames / fps
        beats = beats - start_time
        beats = np.array([b for b in beats if 0 <= b <= window_size_seconds])
        downbeats = downbeats - start_time
        downbeats = np.array([b for b in downbeats if 0 <= b <= window_size_seconds])

        audio = audio.get_window(start_frame, window_size_frames, fps)

    play_audio_with_clicks(
        audio,
        beats,
        downbeats,
        **kwargs,
    )


def listen_to_model_output(
    spect_path,
    ModelOutput: ModelOutput,
    start_frame=None,
    fps=50,
):
    audio_path, _ = get_audio_and_annotation_paths(spect_path)

    audio = MonoAudio.from_path(audio_path)

    if start_frame is not None:
        audio = audio.get_window(
            start_frame,
            fps=fps,
        )

    grid_times = np.where(ModelOutput.grid_mask[0])[0] / fps
    beat_times = np.where(ModelOutput.beat_mask[0])[0] / fps
    downbeat_times = np.where(ModelOutput.downbeat_mask[0])[0] / fps

    play_audio_with_multi_layer_clicks(
        audio,
        layers=[
            (downbeat_times, 1500, 1.0),
            (beat_times, 1000, 1.0),
            (grid_times, 3000, 0.1),
        ],
    )


def listen_to_prediction(
    spect_path,
    beat_mask,
    start_frame=None,
    downbeat_mask=None,
    fps=50,
    **kwargs,
):
    audio_path, _ = get_audio_and_annotation_paths(spect_path)

    audio = MonoAudio.from_path(audio_path)

    if start_frame is not None:
        audio = audio.get_window(
            start_frame,
            fps=fps,
        )

    if downbeat_mask is None:
        downbeat_mask = []
    prediction_beats = np.where(beat_mask)[0] / fps
    prediction_downbeats = np.where(downbeat_mask)[0] / fps

    play_audio_with_clicks(
        audio,
        prediction_beats,
        prediction_downbeats,
        **kwargs,
    )


def get_audio_and_annotation_paths(
    spect_path: str, data_folder="../data"
) -> tuple[Path, Path]:
    dataset, name, _ = spect_path.split("/")

    audio_path = (
        Path(data_folder) / "audio" / "mono_tracks" / dataset / name / "track.wav"
    )

    annotation_path = (
        Path(data_folder)
        / "annotations"
        / dataset
        / "annotations"
        / "beats"
        / (name + ".beats")
    )

    return audio_path, annotation_path
