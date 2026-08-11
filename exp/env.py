from __future__ import annotations

import os
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

VALID_ACTIONS = {"walk straight", "turn left", "turn right", "look up", "look down"}

@dataclass(frozen=True)
class EnvConfig:
    port: int = 8080
    timeout_wait: int = 60
    image_width: int = 640
    image_height: int = 360
    character_resource: str = "Chars/Female2"
    action_repeats: int = 1
    first_person_camera_offset: int = -8
    look_up_camera_offset: int = -2
    look_down_camera_offset: int = -1

@dataclass
class Observation:
    image_path: str
    visible_objects: list[Any]
    character_nodes: list[dict[str, Any]]
    step: int
    previous_action: str | None = None

class VirtualHomeEnv:
    """Thin VirtualHome RPC wrapper.

    This class owns the UnityCommunication client and environment operations.
    Starting or stopping the Unity process belongs in runner.py or a simulator
    manager, not here.
    """

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.comm = None
        self.camera_count = 0
        self.camera_mode = "first_person"
        self.camera_indexes: dict[str, int] = {}

    def connect(self) -> None:
        self.comm = self._new_comm()

    def reconnect(self) -> None:
        self.connect()

    def health_check(self) -> None:
        self._require_connected()
        response = self.comm.post_command({"id": str(time.time()), "action": "idle"})
        if not response.get("success"):
            raise RuntimeError("VirtualHome health check failed")

    def reset(self, sample: dict[str, Any]) -> None:
        self._require_connected()

        meta = sample["meta"]
        env_id = meta["env"]
        init_position = meta["character_init_position"]

        if not self.comm.reset(env_id):
            raise RuntimeError(f"Environment reset failed: env_id={env_id}")

        expanded = self.comm.expand_scene(sample["graph"])
        if isinstance(expanded, tuple) and not expanded[0]:
            raise RuntimeError(f"Scene expansion failed: {expanded}")

        self.comm.add_character_camera(
            position=[0, 1.5, 0],
            rotation=[-15, 0, 0],
            name="look_up_camera",
        )
        self.comm.add_character_camera(
            position=[0, 1.5, 0],
            rotation=[30, 0, 0],
            name="look_down_camera",
        )

        added = self.comm.add_character(
            character_resource=self.config.character_resource,
            position=init_position,
        )
        if isinstance(added, tuple) and not added[0]:
            raise RuntimeError(f"Character add failed: {added}")

        ok, count = self.comm.camera_count()
        if not ok:
            raise RuntimeError("Failed to read camera count")

        self.camera_count = count
        self.camera_indexes = self._character_camera_indexes(count)
        self.camera_mode = "first_person"

    def observe(
        self,
        step: int,
        image_path: str | Path,
        previous_action: str | None = None,
    ) -> Observation:
        self._require_connected()

        image_path = Path(image_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)

        camera_index = self._camera_index()
        self._save_camera_image(camera_index, image_path)

        return Observation(
            image_path=str(image_path),
            visible_objects=self._visible_objects(camera_index),
            character_nodes=self._character_nodes(),
            step=step,
            previous_action=previous_action,
        )

    def step(
        self,
        action: str,
        angle: int | None = None,
        movement_steps: int | None = None,
        turn_angle: int | None = None,
    ) -> str:
        self._require_connected()

        normalized = action.strip().lower()
        if normalized not in VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        if normalized == "look up":
            self._update_named_camera(
                "look_up_camera",
                rotation=[-int(angle or 15), 0, 0],
            )
            self.camera_mode = "look_up"
            return normalized

        if normalized == "look down":
            self._update_named_camera(
                "look_down_camera",
                rotation=[int(angle or 30), 0, 0],
            )
            self.camera_mode = "look_down"
            return normalized

        for script in self._action_to_script(
            normalized,
            movement_steps=movement_steps,
            turn_angle=turn_angle,
        ):
            rendered = self.comm.render_script(
                [script],
                recording=False,
                frame_rate=30,
                skip_animation=True,
            )
            if isinstance(rendered, tuple) and not rendered[0]:
                raise RuntimeError(f"render_script failed: {rendered}")

        self.camera_mode = "first_person"
        return normalized

    def close(self) -> None:
        self.comm = None

    def _new_comm(self):
        vh_root = os.getenv("VH_ROOT")
        if not vh_root:
            raise RuntimeError(
                "VH_ROOT is not set. Clone VirtualHome and set VH_ROOT to its repository root."
            )

        simulation_root = Path(vh_root).expanduser().resolve() / "virtualhome" / "simulation"
        comm_module = simulation_root / "unity_simulator" / "comm_unity.py"
        if not comm_module.is_file():
            raise RuntimeError(
                "Invalid VH_ROOT: expected VirtualHome communication source at "
                f"{comm_module}"
            )

        if str(simulation_root) not in sys.path:
            sys.path.insert(0, str(simulation_root))

        try:
            from unity_simulator.comm_unity import UnityCommunication
        except Exception as import_error:
            raise RuntimeError(
                f"Cannot import VirtualHome UnityCommunication from {simulation_root}: "
                f"{import_error!r}"
            ) from import_error

        return UnityCommunication(
            timeout_wait=self.config.timeout_wait,
            port=str(self.config.port),
        )

    def _require_connected(self) -> None:
        if self.comm is None:
            raise RuntimeError("VirtualHomeEnv is not connected. Call env.connect() first.")

    def _camera_index(self) -> int:
        camera_name_by_mode = {
            "first_person": "FIRST_PERSON",
            "look_up": "look_up_camera",
            "look_down": "look_down_camera",
        }
        camera_name = camera_name_by_mode.get(self.camera_mode)
        if camera_name and camera_name in self.camera_indexes:
            return self.camera_indexes[camera_name]

        if self.camera_mode == "look_up":
            offset = self.config.look_up_camera_offset
        elif self.camera_mode == "look_down":
            offset = self.config.look_down_camera_offset
        else:
            offset = self.config.first_person_camera_offset
        return self.camera_count + offset

    def _update_named_camera(
        self,
        name: str,
        position: list[float] | None = None,
        rotation: list[int] | None = None,
        field_view: int = 60,
    ) -> None:
        camera_index = self.camera_indexes.get(name)
        if camera_index is None:
            raise RuntimeError(f"Unknown camera name: {name}")
        ok, message = self.comm.update_camera(
            camera_index,
            position=position or [0, 1.5, 0],
            rotation=rotation or [0, 0, 0],
            field_view=field_view,
        )
        if not ok:
            raise RuntimeError(f"update_camera failed for {name} index={camera_index}: {message}")

    def _character_camera_indexes(self, camera_count: int) -> dict[str, int]:
        try:
            ok, payload = self.comm.character_cameras()
            if not ok:
                return {}
            names = json.loads(payload) if isinstance(payload, str) else payload
            if not isinstance(names, list):
                return {}
            start_index = camera_count - len(names)
            return {str(name): start_index + index for index, name in enumerate(names)}
        except Exception:
            return {}

    def _save_camera_image(self, camera_index: int, image_path: Path) -> None:
        ok, images = self.comm.camera_image(
            [camera_index],
            mode="normal",
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )
        if not ok or not images:
            raise RuntimeError(f"camera_image failed: camera_index={camera_index}")

        image = np.asarray(images[0]).astype("uint8")
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image[:, :, ::-1]
        Image.fromarray(image).save(image_path)

    def _visible_objects(self, camera_index: int) -> list[Any]:
        try:
            ok, objects = self.comm.get_visible_objects(camera_index)
            return objects if ok else []
        except Exception:
            return []

    def _character_nodes(self) -> list[dict[str, Any]]:
        try:
            ok, graph = self.comm.environment_graph()
            if not ok or not isinstance(graph, dict):
                return []
            nodes = graph.get("nodes", [])
            if not isinstance(nodes, list):
                return []
            return [
                node
                for node in nodes
                if isinstance(node, dict) and node.get("class_name") == "character"
            ]
        except Exception:
            return []

    def _action_to_script(
        self,
        action: str,
        movement_steps: int | None = None,
        turn_angle: int | None = None,
    ) -> list[str]:
        command_by_action = {
            "walk straight": "[WalkForward]",
            "turn left": "[TurnLeft]",
            "turn right": "[TurnRight]",
        }
        command = command_by_action[action]
        repeats = self.config.action_repeats
        if action == "walk straight" and movement_steps is not None:
            repeats = max(1, int(movement_steps))
        elif action in {"turn left", "turn right"} and turn_angle is not None:
            repeats = round(max(1, int(turn_angle)) / 30)
            repeats = max(1, repeats)
        return [f"<char0> {command}" for _ in range(repeats)]
