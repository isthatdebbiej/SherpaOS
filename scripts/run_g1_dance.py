"""Run a single G1 dance/skill end-to-end in MuJoCo, fully automated.

Uses the pinned FSMDeployG1 fork (RoboMimic Deploy) fetched by
scripts/fetch_fsm_dance_repo.py into gitignored third_party/FSMDeployG1/.
This is a separate demo lane from the v26 walking policy
(docs/V26_HIMALAYA_PLAYGROUND.md): it drives FSMDeployG1's own bundled
FSM (PassiveMode -> FixedPose -> LocoMode -> skill ONNX -> cooldown) rather
than SherpaOS code, and SherpaOS does not intervene in it.

Sequence: let PassiveMode settle briefly -> stand up (FixedPose) -> enter
LocoMode -> trigger the chosen skill -> hold until the choreography
finishes -> return to a safe standing/passive state. No manual key timing
needed - this scripts the same sequence that worked reliably in manual
testing (start -> a+r1 -> x+r1) but with deterministic sim-time waits
instead of depending on how fast a human types.

Setup (from SherpaOS repo root):
    uv run python scripts/fetch_fsm_dance_repo.py
    cd third_party/FSMDeployG1 && uv venv --python 3.10 .venv && \\
        source .venv/bin/activate && \\
        uv pip install numpy pyyaml onnx onnxruntime mujoco torch && cd -

Usage:
    cd third_party/FSMDeployG1 && source .venv/bin/activate
    python ../../scripts/run_g1_dance.py dance
    python ../../scripts/run_g1_dance.py kungfu
    python ../../scripts/run_g1_dance.py kick        # labeled unstable upstream
    python ../../scripts/run_g1_dance.py beyondmimic # unstable, see docs

Verified results (see docs/G1_DANCE_DEMO.md for the full log):
    dance        Charleston routine.            18.0s  STABLE - ends standing
    kungfu       Martial-arts movement.          17.4s  STABLE - ends standing
    kick         "Bad mimic policy"                3.6s STABLE - ends standing
    beyondmimic  Fight/sports mimic sequence.    140.0s UNSTABLE - repeatedly falls
"""
import argparse
import os
import sys
import time
from pathlib import Path

FSM_REPO = Path(__file__).resolve().parent.parent / "third_party" / "FSMDeployG1"
if not FSM_REPO.is_dir():
    raise SystemExit(
        f"missing {FSM_REPO}; run: uv run python scripts/fetch_fsm_dance_repo.py"
    )
sys.path.insert(0, str(FSM_REPO))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from common.ctrlcomp import PolicyOutput, StateAndCmd  # noqa: E402
from common.path_config import PROJECT_ROOT  # noqa: E402
from common.utils import FSMCommand, get_gravity_orientation  # noqa: E402
from FSM.FSM import FSM  # noqa: E402

SKILLS = {
    "dance": (FSMCommand.SKILL_1, 18.0, "Charleston routine (verified stable)"),
    "kungfu": (FSMCommand.SKILL_2, 17.433, "Martial-arts movement (verified stable)"),
    "kick": (
        FSMCommand.SKILL_3, 3.633,
        "Kick (verified stable, upstream README calls it 'bad mimic policy')",
    ),
    "beyondmimic": (
        FSMCommand.SKILL_4, 140.0,
        "Fight/sports mimic (UNSTABLE - repeatedly falls, not demo-ready)",
    ),
}
# Note: KungFu2 exists as a bundled ONNX but has no FSMCommand wired to it in
# this fork's LocoMode.checkChange() (only SKILL_1..4 -> Dance/KungFu/Kick/
# BeyondMimic are reachable) - it's excluded here rather than mis-triggered.


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "skill", nargs="?", choices=list(SKILLS.keys()), help="Which skill to run"
    )
    parser.add_argument("--list", action="store_true", help="List available skills and exit")
    parser.add_argument(
        "--margin", type=float, default=2.0, help="Extra seconds to hold after motion_length"
    )
    args = parser.parse_args()

    if args.list or not args.skill:
        print("Available skills:")
        for name, (_, length, desc) in SKILLS.items():
            print(f"  {name:12s} {length:6.1f}s  {desc}")
        return

    skill_cmd, motion_length, desc = SKILLS[args.skill]
    print(f"Running skill '{args.skill}': {desc} (~{motion_length:.1f}s)")

    current_dir = str(FSM_REPO / "deploy_mujoco")
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    with open(mujoco_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = os.path.join(PROJECT_ROOT, config["xml_path"])
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    num_joints = m.nu
    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0

    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)

    sim_time = 0.0
    events = [
        (0.5, "stand", None),
        (3.5, "loco", None),
        (5.0, "skill", skill_cmd),
    ]
    finish_time = 5.0 + motion_length + args.margin
    events_done = [False] * len(events)
    returned_to_loco = False

    print("MuJoCo viewer starting - watch the window; this script drives the FSM automatically.")

    with mujoco.viewer.launch_passive(m, d, show_right_ui=False) as viewer:
        while viewer.is_running() and sim_time < finish_time + 3.0:
            step_start = time.time()

            for i, (t_trigger, action, payload) in enumerate(events):
                if not events_done[i] and sim_time >= t_trigger:
                    events_done[i] = True
                    if action == "stand":
                        state_cmd.skill_cmd = FSMCommand.POS_RESET
                        print(f"[t={sim_time:.1f}s] stand up")
                    elif action == "loco":
                        state_cmd.skill_cmd = FSMCommand.LOCO
                        print(f"[t={sim_time:.1f}s] locomotion mode")
                    elif action == "skill":
                        state_cmd.skill_cmd = payload
                        print(f"[t={sim_time:.1f}s] trigger skill: {args.skill}")

            if not returned_to_loco and sim_time >= finish_time:
                state_cmd.skill_cmd = FSMCommand.LOCO
                returned_to_loco = True
                print(f"[t={sim_time:.1f}s] skill finished, returning to LocoMode")

            tau = pd_control(policy_output_action, d.qpos[7:], kps,
                              np.zeros_like(kps), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            sim_counter += 1
            sim_time = sim_counter * simulation_dt
            if sim_counter % control_decimation == 0:
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                gravity_orientation = get_gravity_orientation(quat)

                state_cmd.q = qj.copy()
                state_cmd.dq = dqj.copy()
                state_cmd.gravity_ori = gravity_orientation.copy()
                state_cmd.base_quat = quat.copy()
                state_cmd.ang_vel = omega.copy()

                FSM_controller.run()
                policy_output_action = policy_output.actions.copy()
                kps = policy_output.kps.copy()
                kds = policy_output.kds.copy()

            if sim_counter % 500 == 0:
                print(f"[diag] t={sim_time:.1f}s pelvis_height={d.qpos[2]:.3f}m "
                      f"policy={FSM_controller.cur_policy.name_str}")

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print(f"Done. Final pelvis height: {d.qpos[2]:.3f}m "
          f"({'OK - still standing' if d.qpos[2] > 0.5 else 'FELL'})")


if __name__ == "__main__":
    main()
