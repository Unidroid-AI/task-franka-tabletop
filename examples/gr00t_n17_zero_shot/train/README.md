# Fine-tuning

Evaluation does not run this directory.

Use Isaac-GR00T's fine-tuning entry point with the dataset written by
`scripts/record_demos.py`. The modality file in that dataset is
`franka_2f85_jointpos`: absolute arm joints plus a gripper command.

Point the trainer at your copy of the base checkpoint. Keep the full
fine-tuned checkpoint, including the backbone, under `weights/`, and list
each file's sha256 in `agent.yaml`. The tournament upload does not include
`weights/`; push that directory to your private Hugging Face repo and submit
with `npcli`, as described in the repository README.
