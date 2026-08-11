# HomeSafeBench Data

This directory contains the public training and test splits of HomeSafeBench.

```text
data/
|- train/    # 3,400 training tasks
`- test/     # 1,000 evaluation tasks
```

Each JSON file represents one home-safety inspection task in a VirtualHome
scene. A task contains its scene graph, the agent's initial position, and the
ground-truth hazards for evaluation.

## Data Format

Each task follows this schema:

```json
{
  "meta": {
    "sample_id": "...",
    "env": 3,
    "room": "kitchen",
    "N": 5,
    "character_init_position": [4.5, 0, -3]
  },
  "graph": {
    "nodes": [],
    "edges": []
  },
  "dangers": [
    {
      "id": 0,
      "danger_type": "falling",
      "name": "wallshelf",
      "objects": [
        {
          "id": 295,
          "class_name": "keyboard"
        }
      ]
    }
  ]
}
```

## `meta`

- `env`: VirtualHome environment identifier.
- `room`: Room type: `kitchen`, `bedroom`, `livingroom`, or `bathroom`.
- `N`: Number of ground-truth hazards in this task. It equals `len(dangers)`.
- `character_init_position`: Initial position of the embodied agent in the
  scene.

## `graph`

VirtualHome represents each environment as a scene graph. The graph defines
the objects in the scene and their relations:

- `nodes`: VirtualHome scene objects, including their native object IDs,
  classes, attributes, and transforms.
- `edges`: Relations between scene objects.

Each HomeSafeBench task is constructed from a VirtualHome scene graph. To
create a hazard instance, we modify the transforms of selected existing
objects in the graph, thereby changing their positions in the scene while
preserving the underlying environment and object identities. For example, an
object may be moved into a sink, near a stove, or close to the edge of an
elevated surface.

The resulting graph is used to instantiate the task scene in VirtualHome. The
agent is then placed at `meta.character_init_position` before inspection
begins.

## `dangers`

`dangers` contains the ground-truth hazards used for evaluation. Each task
contains one to five hazards.

- `id`: Identifier of the hazard within this task.
- `danger_type`: Hazard category. Possible values are `fire`, `electric`,
  `falling`, `trip`, and `children`.
- `name`: Semantic name of the annotated hazardous location or relation.
- `objects`: VirtualHome scene objects involved in the hazard.
  - `id`: Native VirtualHome object node ID. It directly corresponds to the
    `id` of the associated object in `graph.nodes`; it is not a dataset-specific
    or manually assigned identifier.
  - `class_name`: VirtualHome class name of the object.
