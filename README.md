# Franka tabletop

A manipulation benchmark for a tournament whose model is GR00T N1.7. The arm is a Franka with a 2F-85 gripper. Actions are 8-D absolute joint positions: seven arm joints in radians and a gripper command.

## Observation

| Key | Shape | Unit |
| --- | --- | --- |
| `cam_exterior_left`, `cam_exterior_right`, `cam_wrist` | `[N, 720, 1280, 3]` uint8 | image |
| `joint_pos` | `[N, 7]` | rad |
| `gripper_pos` | `[N, 1]` | 0..1 |
| `eef_pose` | `[N, 7]` | m and wxyz quaternion in the base frame |
| instruction | string per env | passed at episode start |

The reference brain resizes the left exterior and the wrist to 180x320 and converts a relative action chunk to absolute joint targets. Horizon is 8. `cam_exterior_right` is the same over-shoulder view from the other side, for a policy trained on two exterior cameras. `cam_front` looks at the table from the front and is not part of the observation sent to the brain. Mounting `/output/videos` writes three files per shard: `front_<job_id>.mp4`, `exterior_left_<job_id>.mp4`, and `wrist_<job_id>.mp4`. Each frame is 720x1280, with the shard's environments in a 4x4 grid and env 0 at the top left.

## Public and hidden

Public seeds and instruction wording ship in this package so a model can be trained. Each tournament phase replaces them with a hidden bundle: different seeds, object layouts, and instruction pools. The phase config names that bundle. Participants do not receive it.

## Submission

The brain image mounts this directory. It is not a container build. Locally, and inside the sandbox, the layout is:

```
agent.yaml
brain/
weights/     # full checkpoint, hashes listed in agent.yaml
train/       # kept for review, not executed
```

A Franka tournament does not take `weights/` in the uploaded archive. Keep the checkpoint in a private Hugging Face model repo under your own user account, then submit only the code archive and a signed seal:

```bash
npcli tournament proof --path ./my-agent --tournament-id <id>
# copy nepher-proof.json to the Hugging Face repo root, push, and note the 40-character commit SHA
npcli tournament check --path ./my-agent --submission-mode huggingface
npcli tournament submit --path ./my-agent --tournament-id <id> \
    --hf-repo alice/my-weights --hf-revision <40-hex commit> --hf-token <read-only token>
```

The token must be fine-grained and limited to read access on that one repo. Leave the repo private, and do not push changed weights after submitting. The public GR00T checkpoint may be submitted once; a later copy of those same files is rejected. A fine-tune may still include those files.

`examples/gr00t_n17_zero_shot` is a complete example. Adding the base checkpoint under `weights/` and its hashes makes it the zero-shot baseline.

`embodiment_tag` in `agent.yaml` must be the tag of that checkpoint's Franka joint-position embodiment. The tag is not hard-coded in this repository.

## Scripts

- `scripts/play_brain.py` prints a public-seed instruction for a local brain socket.
- `scripts/record_demos.py` writes a starter fine-tuning dataset from a scripted expert.
- `scripts/build_envhub_bundle.py` is Nepher-internal. It writes the hidden phase bundle. The license sidecar is a separate path and is not placed in the bundle.

Install the package with `pip install -e source/franka_tabletop` inside an Isaac Lab 3.0 environment.

## Validator host

These checks need a Linux host with a GPU, Isaac Sim 6.1, and the NVIDIA Container Toolkit. They are not part of the unit tests:

- Launch one task family with the zero-action brain over a loopback socket and a two-job manifest, then run the same two shards twice and compare trajectory hashes.
- Add the base checkpoint under `examples/gr00t_n17_zero_shot/weights`, fill its sha256 values, and score it as the phase-1 baseline.
- Build `nepher-brain-images/gr00t-n17-v0` with `build_and_push.sh`, then pull that digest and run check, smoke, and the existing sandbox image with `RUNTIME=brain`.
