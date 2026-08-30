"""Clone the pinned FSMDeployG1 fork (RoboMimic Deploy multi-policy G1 FSM)
used by the dance/skill demo lane. This is not redistributed in Git: it is
fetched fresh into gitignored third_party/FSMDeployG1/ at a pinned commit,
matching the third_party/mujoco_menagerie/ convention in this repo.

FSMDeployG1 has no explicit license (upstream GitHub API reports
license: null as of the pinned commit) - see docs/DECISIONS.md and
third_party/ATTRIBUTIONS.md. It is used here only as a local, offline demo
harness; nothing from it ships in SherpaOS's runtime safety/actuation path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Renforce-Dynamics/FSMDeployG1.git"
PINNED_COMMIT = "18f517b48c3eb7acce1f4c45bbb5db3900b5c2f1"
DEST = Path("third_party/FSMDeployG1")


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def apply_missing_motion_patch(dest: Path) -> None:
    """FSM.py eagerly constructs mimic policies that require motion .npz
    files this repo does not vendor (GAE_Mimic, SONIC_ROBOT_Mimic,
    SONIC_HUMAN_Mimic); unmodified it raises FileNotFoundError at import
    time and the demo never starts. Patch it to skip unavailable policies
    instead, leaving Dance/KungFu/Kick/BeyondMimic (which ship their ONNX
    weights in the repo) fully usable."""
    fsm_path = dest / "FSM" / "FSM.py"
    text = fsm_path.read_text()
    if "_try_init" in text:
        return  # already patched
    text = text.replace(
        "        self.FSMmode = FSMMode.NORMAL\n        \n        self.passive_mode",
        "        self.FSMmode = FSMMode.NORMAL\n\n"
        "        def _try_init(name, cls):\n"
        "            try:\n"
        "                return cls(state_cmd, policy_output)\n"
        "            except Exception as e:\n"
        "                print(f\"[FSM] Skipping policy '{name}' (missing asset): {e}\")\n"
        "                return None\n\n"
        "        self.passive_mode",
    )
    text = text.replace(
        "        self.gae_mimic_policy = GAE_Mimic(state_cmd, policy_output)\n"
        "        self.sonic_robot_mimic_policy = SONIC_ROBOT_Mimic(state_cmd, policy_output)\n"
        "        self.sonic_human_mimic_policy = SONIC_HUMAN_Mimic(state_cmd, policy_output)\n",
        "        self.gae_mimic_policy = _try_init(\"GAE_Mimic\", GAE_Mimic)\n"
        "        self.sonic_robot_mimic_policy = _try_init(\n"
        "            \"SONIC_ROBOT_Mimic\", SONIC_ROBOT_Mimic\n"
        "        )\n"
        "        self.sonic_human_mimic_policy = _try_init(\n"
        "            \"SONIC_HUMAN_Mimic\", SONIC_HUMAN_Mimic\n"
        "        )\n",
    )
    text = text.replace(
        "        elif((policy_name == FSMStateName.SKILL_GAE)):\n"
        "            self.cur_policy = self.gae_mimic_policy\n"
        "        elif((policy_name == FSMStateName.SKILL_SONIC_ROBOT_MIMIC)):\n"
        "            self.cur_policy = self.sonic_robot_mimic_policy\n"
        "        elif((policy_name == FSMStateName.SKILL_SONIC_HUMAN_MIMIC)):\n"
        "            self.cur_policy = self.sonic_human_mimic_policy\n",
        "        elif((policy_name == FSMStateName.SKILL_GAE)):\n"
        "            if self.gae_mimic_policy is not None:\n"
        "                self.cur_policy = self.gae_mimic_policy\n"
        "        elif((policy_name == FSMStateName.SKILL_SONIC_ROBOT_MIMIC)):\n"
        "            if self.sonic_robot_mimic_policy is not None:\n"
        "                self.cur_policy = self.sonic_robot_mimic_policy\n"
        "        elif((policy_name == FSMStateName.SKILL_SONIC_HUMAN_MIMIC)):\n"
        "            if self.sonic_human_mimic_policy is not None:\n"
        "                self.cur_policy = self.sonic_human_mimic_policy\n",
    )
    fsm_path.write_text(text)
    print(f"patched {fsm_path} to skip unavailable mimic policies")


def main() -> None:
    if DEST.exists():
        current = subprocess.run(
            ["git", "-C", str(DEST), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if current == PINNED_COMMIT:
            print(f"already present at pinned commit {PINNED_COMMIT}")
            apply_missing_motion_patch(DEST)
            return
        raise SystemExit(
            f"{DEST} exists at {current}, expected {PINNED_COMMIT}; "
            f"remove it and re-run to re-pin"
        )
    run("git", "clone", REPO_URL, str(DEST))
    run("git", "-C", str(DEST), "checkout", PINNED_COMMIT)
    apply_missing_motion_patch(DEST)
    print(f"cloned {REPO_URL} at {PINNED_COMMIT} -> {DEST}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
