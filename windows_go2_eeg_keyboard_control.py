import argparse
import csv
import io
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def load_from_bytes_cpu(b):
    return torch.load(
        io.BytesIO(b),
        map_location=torch.device("cpu"),
        weights_only=False,
    )


torch.storage._load_from_bytes = load_from_bytes_cpu
torch.set_default_device("cpu")

from Go2Py.sim.mujoco import Go2Sim


VX = 0.6
VY = 0.4
YAW = 0.9
CONTROL_DT = 0.01
STEP_DT = 0.005


KEY_LABELS = {
    "up": "forward",
    "down": "backward",
    "left": "turn_left",
    "right": "turn_right",
    "space": "stop",
    "x": "exit",
    "esc": "exit",
}

TARGET_LABELS = {
    "forward": "FORWARD",
    "turn_left": "LEFT TURN",
    "turn_right": "RIGHT TURN",
}

TARGET_COLORS = {
    "forward": np.array([0.1, 0.8, 0.25, 0.85], dtype=np.float32),
    "turn_left": np.array([1.0, 0.72, 0.1, 0.85], dtype=np.float32),
    "turn_right": np.array([0.1, 0.45, 1.0, 0.85], dtype=np.float32),
}

TARGET_LOCAL_POSES = {
    "forward": (np.array([2.5, 0.0, 0.05]), 0.0),
    "turn_left": (np.array([0.9, 0.9, 0.05]), np.pi / 2.0),
    "turn_right": (np.array([0.9, -0.9, 0.05]), -np.pi / 2.0),
}


def yaw_from_quat_wxyz(quat):
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def setup_teleop_camera(robot):
    if not getattr(robot, "render", False):
        return

    robot.viewer.cam.distance = 2.2
    robot.viewer.cam.elevation = -15
    update_teleop_camera(robot)


def update_teleop_camera(robot):
    if not getattr(robot, "render", False):
        return

    base_pos = robot.data.qpos[:3]
    base_yaw = yaw_from_quat_wxyz(robot.data.qpos[3:7])
    forward = np.array([np.cos(base_yaw), np.sin(base_yaw), 0.0])

    robot.viewer.cam.lookat[:] = base_pos + 0.5 * forward + np.array([0.0, 0.0, 0.30])
    robot.viewer.cam.azimuth = np.degrees(base_yaw)


def rotation_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            c,
            -s,
            0.0,
            s,
            c,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )


def transform_local_to_world(robot, local_pos):
    base_pos = robot.data.qpos[:3].copy()
    base_yaw = yaw_from_quat_wxyz(robot.data.qpos[3:7])
    rot = rotation_z(base_yaw).reshape(3, 3)
    world_pos = base_pos + rot @ local_pos
    world_pos[2] = local_pos[2]
    return world_pos, base_yaw


class WorldCueRenderer:
    def __init__(self):
        self.target = None
        self.anchor_pos = None
        self.anchor_yaw = 0.0

    def set_target(self, robot, target):
        self.target = target
        if target is None or not getattr(robot, "render", False):
            self.clear(robot)
            return

        self.anchor_pos = robot.data.qpos[:3].copy()
        self.anchor_yaw = yaw_from_quat_wxyz(robot.data.qpos[3:7])
        self.draw(robot)

    def clear(self, robot):
        if getattr(robot, "render", False) and robot.viewer.user_scn is not None:
            with robot.viewer.lock():
                robot.viewer.user_scn.ngeom = 0

    def draw(self, robot):
        if not getattr(robot, "render", False) or robot.viewer.user_scn is None:
            return
        if self.target is None:
            self.clear(robot)
            return

        import mujoco

        local_pos, local_yaw = TARGET_LOCAL_POSES[self.target]
        color = TARGET_COLORS[self.target]
        anchor_pos = self.anchor_pos if self.anchor_pos is not None else robot.data.qpos[:3]
        anchor_yaw = self.anchor_yaw
        rot = rotation_z(anchor_yaw).reshape(3, 3)
        target_pos = anchor_pos + rot @ local_pos
        target_pos[2] = local_pos[2]
        target_yaw = anchor_yaw + local_yaw

        with robot.viewer.lock():
            scene = robot.viewer.user_scn
            scene.ngeom = 0
            self._add_cylinder(
                scene,
                mujoco,
                pos=target_pos,
                radius=0.28,
                height=0.035,
                rgba=color,
            )
            self._add_arrow(
                scene,
                mujoco,
                pos=target_pos + np.array([0.0, 0.0, 0.20]),
                yaw=target_yaw,
                rgba=color,
            )

    def _add_cylinder(self, scene, mujoco, pos, radius, height, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array([radius, radius, height], dtype=np.float64),
            pos.astype(np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba,
        )
        scene.ngeom += 1

    def _add_arrow(self, scene, mujoco, pos, yaw, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.array([0.06, 0.06, 0.65], dtype=np.float64),
            pos.astype(np.float64),
            rotation_z(yaw),
            rgba,
        )
        scene.ngeom += 1


class KeyboardState:
    def __init__(self):
        self._pressed = set()
        self.exit_requested = False
        self._listener = None

    def __enter__(self):
        from pynput import keyboard

        self._keyboard = keyboard
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._listener is not None:
            self._listener.stop()

    def _normalize_key(self, key):
        if key == self._keyboard.Key.space:
            return "space"
        if key == self._keyboard.Key.esc:
            return "esc"
        if key == self._keyboard.Key.up:
            return "up"
        if key == self._keyboard.Key.down:
            return "down"
        if key == self._keyboard.Key.left:
            return "left"
        if key == self._keyboard.Key.right:
            return "right"
        try:
            return key.char.lower()
        except AttributeError:
            return None

    def _on_press(self, key):
        label = self._normalize_key(key)
        if label is None:
            return
        if label in {"x", "esc"}:
            self.exit_requested = True
        self._pressed.add(label)

    def _on_release(self, key):
        label = self._normalize_key(key)
        if label is not None:
            self._pressed.discard(label)

    def command(self):
        pressed = set(self._pressed)

        if "space" in pressed:
            return "stop", 0.0, 0.0, 0.0

        vx = 0.0
        vy = 0.0
        yaw = 0.0
        labels = []

        if "up" in pressed and "down" not in pressed:
            vx = VX
            labels.append("forward")
        elif "down" in pressed and "up" not in pressed:
            vx = -VX
            labels.append("backward")

        if "left" in pressed and "right" not in pressed:
            yaw = YAW
            labels.append("turn_left")
        elif "right" in pressed and "left" not in pressed:
            yaw = -YAW
            labels.append("turn_right")

        if not labels:
            labels.append("idle")

        return "+".join(labels), vx, vy, yaw

    def is_pressed(self, label):
        return label in self._pressed


class EventLogger:
    def __init__(self, path, participant, session, use_lsl):
        self.path = path
        self.participant = participant
        self.session = session
        self.use_lsl = use_lsl
        self._file = None
        self._writer = None
        self._lsl_outlet = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "wall_time_iso",
                "epoch_ms",
                "monotonic_time",
                "participant",
                "session",
                "trial",
                "target",
                "event",
                "vx",
                "vy",
                "yaw",
            ],
        )
        self._writer.writeheader()

        if self.use_lsl:
            try:
                from pylsl import StreamInfo, StreamOutlet

                info = StreamInfo(
                    "Go2KeyboardMarkers",
                    "Markers",
                    1,
                    0,
                    "string",
                    f"go2_keyboard_{self.participant}_{self.session}",
                )
                self._lsl_outlet = StreamOutlet(info)
                print("LSL marker stream: Go2KeyboardMarkers")
            except Exception as exc:
                print(f"LSL marker stream unavailable: {exc}")
                self._lsl_outlet = None

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file is not None:
            self._file.close()

    def write(self, event, vx, vy, yaw, trial="", target=""):
        row = {
            "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
            "epoch_ms": int(time.time() * 1000),
            "monotonic_time": f"{time.perf_counter():.6f}",
            "participant": self.participant,
            "session": self.session,
            "trial": trial,
            "target": target,
            "event": event,
            "vx": f"{vx:.4f}",
            "vy": f"{vy:.4f}",
            "yaw": f"{yaw:.4f}",
        }
        self._writer.writerow(row)
        self._file.flush()

        if self._lsl_outlet is not None:
            self._lsl_outlet.push_sample([event])


class CueScheduler:
    def __init__(self, targets, trials_per_target, trial_duration, iti_duration, randomize):
        self.targets = []
        if trials_per_target > 0:
            self.targets = list(targets) * trials_per_target
            if randomize:
                random.shuffle(self.targets)

        self.trial_duration = trial_duration
        self.iti_duration = iti_duration
        self.index = -1
        self.phase = "idle"
        self.phase_start = None
        self.current_target = None
        self.done = not self.targets

    def start(self, now):
        if self.done:
            return None
        self.index = 0
        self.phase = "cue"
        self.phase_start = now
        self.current_target = self.targets[self.index]
        return self.event("cue")

    def update(self, now):
        if self.done or self.phase_start is None:
            return None

        elapsed = now - self.phase_start
        if self.phase == "cue" and elapsed >= self.trial_duration:
            self.phase = "iti"
            self.phase_start = now
            return self.event("iti")

        if self.phase == "iti" and elapsed >= self.iti_duration:
            self.index += 1
            if self.index >= len(self.targets):
                self.done = True
                self.phase = "done"
                self.current_target = None
                return {"event": "cue_sequence_complete", "trial": "", "target": ""}

            self.phase = "cue"
            self.phase_start = now
            self.current_target = self.targets[self.index]
            return self.event("cue")

        return None

    def event(self, phase):
        trial = self.index + 1
        target = self.current_target or ""
        if phase == "cue":
            event = f"trial{trial:03d}_start_{target}"
        else:
            event = f"trial{trial:03d}_end_{target}"
        return {"event": event, "trial": trial, "target": target}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Windows keyboard teleoperation for Go2 MuJoCo with EEG event logging.",
    )
    parser.add_argument("--participant", default="test", help="Participant ID for logs and markers.")
    parser.add_argument("--session", default=None, help="Session ID. Defaults to current timestamp.")
    parser.add_argument("--log-dir", default="logs", help="Directory for behavioral event CSV logs.")
    parser.add_argument("--duration", type=float, default=0.0, help="Optional run duration in seconds; 0 means no limit.")
    parser.add_argument("--no-lsl", action="store_true", help="Disable LSL marker stream.")
    parser.add_argument("--no-render", action="store_true", help="Run MuJoCo without opening the viewer.")
    parser.add_argument(
        "--targets",
        default="forward,turn_left,turn_right",
        help="Comma-separated cue targets. Supported: forward,turn_left,turn_right.",
    )
    parser.add_argument("--trials-per-target", type=int, default=0, help="Number of cue trials per target; 0 disables cue scheduling.")
    parser.add_argument("--trial-duration", type=float, default=5.0, help="Cue/trial duration in seconds.")
    parser.add_argument("--iti-duration", type=float, default=3.0, help="Rest interval between cue trials in seconds.")
    parser.add_argument("--no-randomize-cues", action="store_true", help="Keep cue order blocked instead of randomized.")
    return parser.parse_args()


def make_log_path(log_dir, participant, session):
    filename = f"go2_keyboard_events_{participant}_{session}.csv"
    return Path(log_dir) / filename


def main():
    args = parse_args()
    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = make_log_path(args.log_dir, args.participant, session)
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    unsupported_targets = [target for target in targets if target not in TARGET_LABELS]
    if unsupported_targets:
        raise ValueError(f"Unsupported cue target(s): {', '.join(unsupported_targets)}")

    robot = Go2Sim(mode="highlevel", render=not args.no_render)
    robot.standUpReset()
    setup_teleop_camera(robot)

    print("Go2Sim highlevel mode initialized.")
    print(f"Total trials: {len(targets) * args.trials_per_target if args.trials_per_target > 0 else 0}")
    print("Keyboard control:")
    print("  arrow up/down: forward/backward")
    print("  arrow left/right: turn left/right")
    print("  space: stop")
    print("  x or Esc: exit")
    if args.trials_per_target > 0:
        print("Cue targets:")
        print(f"  {', '.join(TARGET_LABELS[target] for target in targets)}")
        print(f"  {args.trials_per_target} trials per target, {args.trial_duration:g}s cue, {args.iti_duration:g}s rest")
    print(f"Behavior log: {log_path}")
    print("Start NeuroAI EEG recording before or immediately after this point.")
    print("Press Space when you are ready to start the cue sequence.")

    start_time = time.monotonic()
    next_control_time = start_time
    last_event = None
    vx, vy, yaw = 0.0, 0.0, 0.0
    experiment_started = args.trials_per_target <= 0
    space_was_pressed = False
    cue_scheduler = CueScheduler(
        targets=targets,
        trials_per_target=args.trials_per_target,
        trial_duration=args.trial_duration,
        iti_duration=args.iti_duration,
        randomize=not args.no_randomize_cues,
    )
    cue_renderer = WorldCueRenderer()

    try:
        with KeyboardState() as keyboard, EventLogger(
            log_path,
            args.participant,
            session,
            use_lsl=not args.no_lsl,
        ) as logger:
            logger.write("ready_waiting", 0.0, 0.0, 0.0)
            if experiment_started:
                logger.write("experiment_start", 0.0, 0.0, 0.0)
                cue_event = cue_scheduler.start(start_time)
                if cue_event is not None:
                    logger.write(cue_event["event"], 0.0, 0.0, 0.0, cue_event["trial"], cue_event["target"])
                    cue_renderer.set_target(robot, cue_event["target"])

            while True:
                now = time.monotonic()

                if keyboard.exit_requested:
                    logger.write("exit_requested", 0.0, 0.0, 0.0)
                    break

                if not experiment_started:
                    space_pressed = keyboard.is_pressed("space")
                    if space_pressed and not space_was_pressed:
                        experiment_started = True
                        start_time = now
                        next_control_time = now
                        logger.write("experiment_start", 0.0, 0.0, 0.0)
                        cue_event = cue_scheduler.start(now)
                        if cue_event is not None:
                            logger.write(
                                cue_event["event"],
                                0.0,
                                0.0,
                                0.0,
                                cue_event["trial"],
                                cue_event["target"],
                            )
                            cue_renderer.set_target(robot, cue_event["target"])
                    space_was_pressed = space_pressed

                    robot.step(
                        0.0,
                        0.0,
                        0.0,
                        step_height=0,
                        kp=[2, 0.5, 0.5],
                        ki=[0.02, 0.01, 0.01],
                    )
                    update_teleop_camera(robot)
                    time.sleep(STEP_DT)
                    continue

                cue_event = cue_scheduler.update(now)
                if cue_event is not None:
                    logger.write(cue_event["event"], vx, vy, yaw, cue_event["trial"], cue_event["target"])
                    cue_renderer.set_target(
                        robot,
                        cue_event["target"]
                        if cue_event["target"] and "_start_" in str(cue_event["event"])
                        else None,
                    )

                if cue_scheduler.done and args.trials_per_target > 0:
                    cue_renderer.clear(robot)
                    logger.write("cue_experiment_complete", 0.0, 0.0, 0.0)
                    break

                if args.duration > 0 and now - start_time >= args.duration:
                    logger.write("duration_complete", 0.0, 0.0, 0.0)
                    break

                if now >= next_control_time:
                    event, vx, vy, yaw = keyboard.command()
                    command_signature = (event, vx, vy, yaw)
                    if command_signature != last_event:
                        logger.write(event, vx, vy, yaw)
                        last_event = command_signature

                    next_control_time += CONTROL_DT
                    if next_control_time <= now:
                        next_control_time = now + CONTROL_DT

                robot.step(
                    vx,
                    vy,
                    yaw,
                    step_height=0,
                    kp=[2, 0.5, 0.5],
                    ki=[0.02, 0.01, 0.01],
                )
                update_teleop_camera(robot)
                time.sleep(STEP_DT)

            logger.write("experiment_end", 0.0, 0.0, 0.0)

    except KeyboardInterrupt:
        print("Interrupted.")

    finally:
        robot.close()

    print("Exit.")


if __name__ == "__main__":
    main()
