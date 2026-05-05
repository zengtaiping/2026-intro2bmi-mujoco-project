# 🧠 2026 Intro2BMI MuJoCo Project

A hands-on project for learning Brain–Machine Interface (BMI) through controlling a quadruped robot in MuJoCo.

![Intro2BMI MuJoCo course cover](assets/course-cover.png)

---

## 🚀 Overview

This project connects neural signal decoding with embodied robot control:


EEG → decoding → motion command → robot locomotion


The motion command can be a discrete direction such as forward/left/right, or a continuous velocity command such as `[vx, vy, yaw]`.

## Experiment Steps

These two steps show the course workflow before running the robot demo: first build an EEG decoding model offline, then use the decoded motion command for online robot control.

### Step 1: Offline EEG Decoding

Use recorded EEG data to train a decoding model. The goal is to transform neural signals into motion commands, such as discrete directions or continuous velocity commands.

![Step 1 Offline EEG Decoding](assets/Step%201%20Offline%20EEG%20Decoding.png)

### Step 2: Online Robot Control

Use real-time EEG signals and the trained decoder to generate robot commands. The robot executes these commands in MuJoCo, forming a closed-loop brain-to-robot control system.

![Step 2 Online Robot Control](assets/Step%202%20Online%20Robot%20Control.png)

## Run Simple MuJoCo Keyboard Control

```bash
conda create -n bmi_mujoco python=3.10
conda activate bmi_mujoco
```

```bash
pip install --upgrade pip
pip install mujoco numpy scipy torch pygame pyyaml opencv-python
pip install -e Go2Py-main
```

Run:

```bash
mjpython bmi_mujoco_keyboard_control/simple_mujoco_keyboard_control.py
```

Keyboard:

```text
w      forward
s      backward
a      turn left
d      turn right
q      move left
e      move right
space  stop
x      exit
```

## Details

Installation package sources:

- MuJoCo: https://github.com/google-deepmind/mujoco
- Go2Py: https://github.com/machines-in-motion/Go2Py
