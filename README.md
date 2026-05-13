# Go2 EEG Keyboard Control Dataset

This project contains a small EEG-based movement-intention experiment built around a MuJoCo simulation of the Unitree Go2 robot. During the experiment, a participant controls the simulated robot with keyboard arrow keys while EEG data are recorded with NeuroAI. The task labels are defined by map cues in the simulation:

- `L`: left turn
- `S`: straight
- `R`: right turn

The repository includes the Windows experiment script, the processed EEG trial dataset, and theoretical tutorials for classical machine learning and neural-network examples.

## Contents

### `data/`

The `data` folder contains the processed supervised-learning dataset:

```text
data/go2_eeg_dataset.npz
```

The `.npz` file contains two arrays:

```python
eeg_feature  # EEG trial data, shape: (trials, channels, timepoints)
label        # trial labels, shape: (trials,)
```

Dataset summary:

- 180 total trials
- 8 EEG channels
- 4990 samples per trial
- 1000 Hz sampling rate
- balanced classes: `L=60`, `S=60`, `R=60`

### `tutorials/`

The `tutorials` folder contains preprocessing and decoding examples for the Go2 EEG dataset:

- `preprocess_trials_demo.ipynb`: preprocess already segmented trial-level EEG features.
- `preprocess_raw_to_epochs_demo.ipynb`: preprocess continuous raw EEG first, then cut it into epochs.
- `decode_logreg_demo.ipynb`: run a logistic-regression baseline for three-class EEG decoding.
- `decode_mlp_demo.ipynb`: run an MLP neural-network baseline for three-class EEG decoding.
- `benchmark_decoders.py`: benchmark multiple classical machine-learning and neural-network decoders.

### `windows_go2_eeg_keyboard_control.py`

This is the main Windows experiment script. It launches the Go2 MuJoCo simulation, displays map-based target cues, listens to keyboard input, and logs trial events.

Key controls:

- `Up`: move forward
- `Left`: turn left
- `Right`: turn right
- `Down`: move backward
- `Space`: start the experiment / stop movement
- `Esc` or `x`: exit

The script records trial timing events such as:

```text
trial001_start_forward
trial001_end_forward
trial002_start_turn_left
trial002_end_turn_left
```

These event timestamps can be aligned with EEG recording timestamps after the experiment.

### `run_windows_go2_eeg.bat`

This batch file starts the Windows experiment using the configured Python virtual environment:

```bat
run_windows_go2_eeg.bat
```

The default experiment configuration is:

- 20 trials per class
- 3 classes: left, straight, right
- 5 seconds per target cue
- 3 seconds between cues
- 60 trials total

## Data Construction

The processed dataset was created by aligning NeuroAI EEG recordings with Go2 trial logs using absolute timestamps. Each trial was cut from the EEG recording using its trial start and end events, then mapped to one of the three labels:

```text
turn_left  -> L
straight   -> S
turn_right -> R
```

The final merged dataset is saved as a single `go2_eeg_dataset.npz` file with only `eeg_feature` and `label`.

## Baseline

### Classical Machine Learning

| Model | Macro Accuracy | Weighted Accuracy | Macro Precision | Weighted Precision | Macro Recall | Weighted Recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LDA | 59.26% | 59.26% | 38.28% | 38.28% | 38.89% | 38.89% | 0.3785 | 0.3785 |
| Logistic Regression | 59.26% | 59.26% | 38.28% | 38.28% | 38.89% | 38.89% | 0.3785 | 0.3785 |
| SVM | 61.11% | 61.11% | 41.75% | 41.75% | 41.67% | 41.67% | 0.4063 | 0.4063 |

### Neural Network

| Model | Macro Accuracy | Weighted Accuracy | Macro Precision | Weighted Precision | Macro Recall | Weighted Recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 55.56% | 55.56% | 33.33% | 33.33% | 33.33% | 33.33% | 0.3333 | 0.3333 |
| CNN | 57.41% | 57.41% | 35.6% | 35.6% | 36.11% | 36.11% | 0.3236 | 0.3236 |
| RNN | 48.15% | 48.15% | 23.84% | 23.84% | 22.22% | 22.22% | 0.2286 | 0.2286 |
| LSTM | 55.56% | 55.56% | 32.14% | 32.14% | 33.33% | 33.33% | 0.3231 | 0.3231 |
| EEGNet-LSTM | 57.41% | 57.41% | 35.46% | 35.46% | 36.11% | 36.11% | 0.3516 | 0.3516 |
| Transformer | 55.56% | 55.56% | 32.76% | 32.76% | 33.33% | 33.33% | 0.3195 | 0.3195 |
