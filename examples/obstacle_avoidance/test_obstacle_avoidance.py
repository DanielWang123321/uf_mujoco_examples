# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Grounded scene, loaded planning, physical pick/place, and failure-path regression."""

import copy
import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from ufactory_mujoco.path_planning import PlanningError, path_length

from common.sim import TaskError
from examples.obstacle_avoidance import manipulation as m
from examples.obstacle_avoidance import obstacle_avoidance as demo
from examples.obstacle_avoidance import planning as p

GRIPPER_BODIES = {
    "xarm_gripper_base_link",
    "left_outer_knuckle",
    "left_inner_knuckle",
    "left_finger",
    "right_outer_knuckle",
    "right_inner_knuckle",
    "right_finger",
}


@pytest.fixture(scope="module")
def loaded_start():
    """A grasp obtained through real contacts, shared only as immutable planning input."""
    with demo.ObstacleRobot(viewer=False) as robot:
        report = {}
        m.initialize(robot, p.Options(), report)
        m.pickup(robot, p.Options(), report)
        return robot.data.qpos.copy(), p.CarriedState.from_data(robot.data), report


@pytest.fixture(scope="module")
def plans(loaded_start):
    qpos, carried, _ = loaded_start
    cache = {}

    def get(seed=0):
        if seed not in cache:
            report = {}
            options = replace(p.Options(), seed=seed)
            checker = p.SceneChecker(carried=carried)
            cache[seed] = p.plan_motion(options, report, checker=checker, q_start=qpos[:6]), report
        return cache[seed]

    return get


def raw_contacts(model, data, qpos):
    """Independent full dynamics forward pass, without the application's query methods."""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return [
        (tuple(model.body(int(model.geom_bodyid[g])).name for g in c.geom), float(c.dist))
        for c in data.contact[: data.ncon]
    ]


def independent_minima(contacts, *, carried):
    environment = mechanism = 0.05
    for names, distance in contacts:
        pair = set(names)
        if pair in ({"world", "link_base"}, {"world", "link1"}):
            continue
        if carried is None and pair == {"world", "payload"} and distance >= -0.001:
            continue
        if (
            carried is not None
            and pair in ({"left_finger", "payload"}, {"right_finger", "payload"})
            and distance >= -0.002
        ):
            continue
        if all(name in GRIPPER_BODIES for name in names):
            mechanism = min(mechanism, distance)
        else:
            environment = min(environment, distance)
    return environment, mechanism


def independent_check(path, carried):
    model = mujoco.MjModel.from_xml_path(str(p.SCENE))
    model.geom_margin[(model.geom_contype != 0) | (model.geom_conaffinity != 0)] = 0.05
    data = mujoco.MjData(model)
    adr = int(model.jnt_qposadr[model.joint("payload_joint").id])
    minimum, internal = 0.05, 0.05
    for first, last in zip(path[:-1], path[1:], strict=True):
        count = max(1, int(np.ceil(np.linalg.norm(last - first) / 0.01)))
        for u in np.linspace(0, 1, count + 1):
            q = first + u * (last - first)
            assert np.all(q >= model.jnt_range[:6, 0] - 1e-9)
            assert np.all(q <= model.jnt_range[:6, 1] + 1e-9)
            data.qpos[:] = model.key("home").qpos
            data.qpos[:6] = q
            data.qpos[6:12] = carried.gripper_qpos
            mujoco.mj_kinematics(model, data)
            tool = data.site("attachment_site")
            rotation = tool.xmat.reshape(3, 3)
            data.qpos[adr : adr + 3] = tool.xpos + rotation @ carried.position
            mujoco.mju_mat2Quat(data.qpos[adr + 3 : adr + 7], (rotation @ carried.rotation).ravel())
            env, grip = independent_minima(
                raw_contacts(model, data, data.qpos.copy()), carried=carried
            )
            minimum, internal = min(minimum, env), min(internal, grip)
    return minimum, internal


@pytest.fixture(scope="module")
def physical_reports(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pick_pillar")
    cache = {}

    def get(seed):
        if seed not in cache:
            path = folder / f"seed{seed}.json"
            code = demo.main(["--headless", "--seed", str(seed), "--report", str(path)])
            cache[seed] = code, json.loads(path.read_text())
        return cache[seed]

    return get


def test_only_pillar_is_grounded_and_payload_has_free_dynamics():
    model = mujoco.MjModel.from_xml_path(str(p.SCENE))
    pillar = model.body("pillar")
    assert pillar.pos == pytest.approx([0.35, 0, 0.225])
    assert model.geom("pillar_geom").size[:2] == pytest.approx([0.045, 0.225])
    assert pillar.pos[2] - model.geom("pillar_geom").size[1] == pytest.approx(0)
    assert pillar.jntnum[0] == 0  # Explicitly anchored to the table.
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "overhead") == -1
    payload = model.body("payload")
    assert payload.mass[0] == pytest.approx(0.08)
    assert model.geom("payload_geom").size == pytest.approx([0.025] * 3)
    assert payload.pos[2] - model.geom("payload_geom").size[2] == pytest.approx(0)
    assert model.joint("payload_joint").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert all(kind != mujoco.mjtEq.mjEQ_WELD for kind in model.eq_type)
    assert model.nmocap == 0
    assert model.opt.timestep == 0.001
    assert np.all(model.geom_margin == 0)  # Inflated margins remain private.


def test_collision_only_queries_match_full_mujoco_forward(loaded_start):
    _, carried, _ = loaded_start
    checker = p.SceneChecker(carried=carried)
    independent_data = mujoco.MjData(checker.model)
    rng = np.random.default_rng(11)
    for _ in range(50):
        q = rng.uniform(checker.q_lo, checker.q_hi)
        checker._set_q(q)
        expected = {
            (int(c.geom1), int(c.geom2), round(float(c.dist), 8))
            for c in checker.data.contact[: checker.data.ncon]
        }
        independent_data.qpos[:] = checker.data.qpos
        mujoco.mj_forward(checker.model, independent_data)
        actual = {
            (int(c.geom1), int(c.geom2), round(float(c.dist), 8))
            for c in independent_data.contact[: independent_data.ncon]
        }
        assert expected == actual


def test_baseline_collides_while_endpoints_are_safe(plans, loaded_start):
    plan, report = plans()
    assert report["baseline_collides"] and report["baseline_min_clearance_m"] < 0
    assert independent_check(plan.path[[0, -1]], loaded_start[1])[0] < 0
    for q in plan.path[[0, -1]]:
        assert not plan.checker.in_collision(q)


def test_pillar_mount_exclusion_and_arm_self_collision():
    spec = mujoco.MjSpec.from_file(str(p.SCENE))
    spec.body("pillar").pos = [0, 0, 0.18]
    checker = p.SceneChecker(spec)
    assert checker.in_collision(checker.home[:6])
    assert any(
        {checker.model.body(a).name, checker.model.body(b).name} == {"pillar", "link1"}
        and distance < 0
        for a, b, distance in checker._contacts(checker.home[:6])
    )
    checker = p.SceneChecker()
    folded = np.array([2.7933, -1.8918, -1.2496, 2.0179, 3.14159, -1.4719])
    assert checker.in_collision(folded)
    assert any(
        {checker.model.body(a).name, checker.model.body(b).name} == {"link3", "right_finger"}
        and distance < 0
        for a, b, distance in checker._contacts(folded)
    )


def test_payload_collision_detected_when_robot_itself_clears_pillar(loaded_start):
    empty = p.SceneChecker()
    q = p.endpoint_candidates(empty, [350, 0, 470, 180, 0, 0], seed=0, name="goal")[0]
    assert not empty.in_collision(q)
    loaded = p.SceneChecker(carried=loaded_start[1])
    assert loaded.in_collision(q)
    assert any(
        {loaded.model.body(a).name, loaded.model.body(b).name} == {"payload", "pillar"}
        and distance < 0
        for a, b, distance in loaded._contacts(q)
    )


def test_runtime_uses_actual_payload_pose_and_does_not_attach_it(loaded_start):
    qpos, carried, _ = loaded_start
    checker = p.SceneChecker(carried=carried)
    measured = qpos.copy()
    adr = checker.payload_adr
    measured[adr : adr + 3] = [0.35, 0, 0.30]
    measured[adr + 3 : adr + 7] = [1, 0, 0, 0]
    assert checker.actual_distance(measured) < 0
    assert checker.data.qpos[adr : adr + 7] == pytest.approx(measured[adr : adr + 7])
    assert checker.min_distance(qpos[:6]) >= 0.01


def test_intended_contacts_do_not_hide_deep_floor_penetration(loaded_start):
    qpos, carried, _ = loaded_start
    checker = p.SceneChecker(carried=carried)
    checker.allow_table = checker.allow_fingers = True
    q = qpos.copy()
    q[checker.payload_adr : checker.payload_adr + 3] = [0.35, -0.2, 0.01]
    assert checker.actual_distance(q) < 0


def test_extrinsic_xyz_pose_conversion():
    pose = [123, -456, 789, 32, -27, 48]
    pos, quat = p.pose_to_target(pose)
    roll, pitch, yaw = np.deg2rad(pose[3:])
    rx = [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    ry = [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    rz = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    assert pos == pytest.approx([0.123, -0.456, 0.789])
    assert Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix() == pytest.approx(
        np.array(rz) @ ry @ rx
    )


@pytest.mark.parametrize("pose", [[0] * 5, [0] * 7, [np.nan] * 6, [np.inf] * 6])
def test_invalid_poses_are_rejected(pose):
    with pytest.raises(ValueError, match="six finite"):
        p.pose_to_target(pose)


@pytest.mark.parametrize("pose", [[2000, 0, 2000, 180, 0, 0], [350, 0, 300, 180, 0, 0]])
def test_unreachable_or_colliding_ik_fails(pose, loaded_start):
    with pytest.raises(p.MotionError) as failure:
        p.endpoint_candidates(p.SceneChecker(carried=loaded_start[1]), pose, seed=0, name="goal")
    assert failure.value.stage == "goal_ik"


def test_multistart_ik_caps_attempts_and_deduplicates(monkeypatch):
    checker = p.SceneChecker()
    q = p.endpoint_candidates(checker, p.START_POSE, seed=0, name="start")[0]
    calls = []

    def duplicated(model, data, site, pos, quat, initial, **kwargs):
        calls.append(initial.copy())
        full = checker.home.copy()
        full[:6] = q
        return full, True

    monkeypatch.setattr(p, "solve_ik", duplicated)
    assert len(p.endpoint_candidates(checker, p.START_POSE, seed=3, name="start")) == 1
    assert len(calls) == 8
    assert calls[0] == pytest.approx(checker.home)
    assert not np.array_equal(calls[1], calls[0])


@pytest.mark.parametrize("seed", range(10))
def test_loaded_seed_stability_and_independent_edge_validation(seed, plans, loaded_start):
    plan, report = plans(seed)
    minimum, internal = independent_check(plan.path, loaded_start[1])
    assert minimum >= 0.01 and internal >= 0.005
    assert report["planned_min_clearance_m"] >= 0.01
    assert report["planned_gripper_internal_min_clearance_m"] >= 0.005
    assert np.array_equal(plan.path[[0, -1]], plan.raw_path[[0, -1]])
    assert path_length(plan.path) <= path_length(plan.raw_path) + 1e-9
    assert np.max(np.abs(plan.trajectory.qd)) <= 0.65 + 1e-6
    assert np.max(np.abs(plan.trajectory.qdd)) <= 2 + 1e-6


def test_loaded_plan_is_reproducible(plans, loaded_start):
    old, _ = plans(1)
    new = p.plan_motion(
        replace(p.Options(), seed=1),
        {},
        checker=p.SceneChecker(carried=loaded_start[1]),
        q_start=loaded_start[0][:6],
    )
    assert np.array_equal(new.path, old.path)
    assert np.array_equal(new.trajectory.q, old.trajectory.q)


def test_retained_goal_ik_solutions_pass_independent_fk(plans):
    plan, report = plans()
    model = mujoco.MjModel.from_xml_path(str(p.SCENE))
    data = mujoco.MjData(model)
    target_pos, target_quat = p.pose_to_target(p.GOAL_POSE)
    target_rotation = Rotation.from_quat(target_quat[[1, 2, 3, 0]])
    assert 1 <= len(report["goal_candidates_rad"]) <= 4
    for q in report["goal_candidates_rad"]:
        data.qpos[:] = model.key("home").qpos
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        assert np.linalg.norm(data.site("attachment_site").xpos - target_pos) <= 0.0002
        rotation = Rotation.from_matrix(data.site("attachment_site").xmat.reshape(3, 3))
        assert np.rad2deg((target_rotation * rotation.inv()).magnitude()) <= 0.1
        assert np.all(np.asarray(q) >= plan.checker.q_lo)
        assert np.all(np.asarray(q) <= plan.checker.q_hi)


def test_entire_timed_trajectory_is_independently_safe(plans, loaded_start):
    plan, _ = plans()
    minimum, internal = independent_check(plan.trajectory.q, loaded_start[1])
    assert minimum >= 0.01 and internal >= 0.005
    assert np.array_equal(plan.trajectory.q[[0, -1]], plan.path[[0, -1]])


def test_invalid_spline_falls_back_to_verified_quintics(monkeypatch, plans):
    plan, _ = plans()
    original = p.plan_joint_waypoints

    def bad_spline(*args, **kwargs):
        tr = original(*args, **kwargs)
        if kwargs["method"] == "spline":
            tr.q[len(tr.q) // 2, 0] = plan.checker.q_hi[0] + 0.1
        return tr

    monkeypatch.setattr(p, "plan_joint_waypoints", bad_spline)
    tr, method, clearance, rejection = p.time_parameterize(plan.path, plan.checker, p.Options())
    assert method == "quintic_polyline" and "joint limits" in rejection
    assert clearance >= 0.01
    assert np.array_equal(tr.q[[0, -1]], plan.path[[0, -1]])


def test_two_invalid_interpolations_prevent_local_physics(monkeypatch, plans):
    plan, _ = plans()

    def invalid(*args, **kwargs):
        result = copy.deepcopy(plan.trajectory)
        result.q[len(result.q) // 2, 0] = np.nan
        return result

    monkeypatch.setattr(p, "plan_joint_waypoints", invalid)
    with demo.ObstacleRobot(viewer=False) as robot:
        robot.runtime_checker = p.SceneChecker()
        robot.data.qpos[:] = robot.runtime_checker.home
        robot.set_arm_qpos(plan.path[0])
        mujoco.mj_forward(robot.model, robot.data)
        before = robot.data.qpos.copy()
        pos, quat = robot.get_tcp_pose()
        with pytest.raises(p.MotionError):
            m.cartesian_move(robot, pos, quat, stage="test", holding=False, allow_table=True)
        assert robot.data.time == 0
        assert np.array_equal(robot.data.qpos, before)


@pytest.mark.parametrize("defect", ["velocity", "acceleration", "endpoint", "timestep"])
def test_trajectory_defects_fail_preflight(defect, plans):
    plan, _ = plans()
    tr = copy.deepcopy(plan.trajectory)
    if defect == "velocity":
        tr.qd[len(tr.q) // 2, 0] = 0.66
    elif defect == "acceleration":
        tr.qdd[len(tr.q) // 2, 0] = 2.01
    elif defect == "endpoint":
        tr.q[-1, 0] += 0.1
    else:
        tr.times[-1] += 0.001
    with pytest.raises(p.MotionError) as failure:
        p.check_trajectory(tr, plan.path, plan.checker, p.Options())
    assert failure.value.stage == "trajectory_validation"


def test_search_exhaustion_and_pair_budget(monkeypatch, plans):
    plan, _ = plans()
    starts = [plan.path[0] + i * 0.01 for i in range(4)]
    goals = [plan.path[-1] + i * 0.01 for i in range(4)]
    monkeypatch.setattr(
        p,
        "endpoint_candidates",
        lambda _c, _pose, *, seed, name: starts if name == "start" else goals,
    )
    attempted = []

    def exhausted(a, b, **kwargs):
        attempted.append(np.linalg.norm(b - a))
        raise PlanningError("budget exhausted")

    monkeypatch.setattr(p, "plan_collision_free", exhausted)
    with pytest.raises(p.MotionError, match="预算内未找到路径"):
        p.plan_motion(p.Options(), {})
    assert attempted == pytest.approx(
        sorted(np.linalg.norm(b - a) for a in starts for b in goals)[:4]
    )


def test_unobstructed_and_identical_endpoints(loaded_start):
    qpos, carried, _ = loaded_start
    spec = mujoco.MjSpec.from_file(str(p.SCENE))
    spec.geom("pillar_geom").contype = spec.geom("pillar_geom").conaffinity = 0
    checker = p.SceneChecker(spec, carried=carried)
    report = {}
    direct = p.plan_motion(p.Options(), report, scene=spec, checker=checker, q_start=qpos[:6])
    assert len(direct.path) == 2 and not report["baseline_collides"]
    identical = p.plan_motion(replace(p.Options(), goal_pose=p.START_POSE), {})
    assert path_length(identical.path) == pytest.approx(0)


@pytest.mark.parametrize("seed", [0, 1])
def test_physical_grasp_carry_release_and_rest(seed, physical_reports):
    code, report = physical_reports(seed)
    assert code == 0 and report["success"] and report["failure_stage"] is None
    assert report["baseline_collides"]
    assert report["lift_mm"] > 100
    assert 47 <= report["grasp_gap_mm"] <= 53
    assert all(force > 0 for force in report["grasp_forces_n"])
    assert report["planned_min_clearance_m"] >= 0.01
    assert report["actual_min_clearance_m"] >= 0.005
    assert report["actual_gripper_internal_min_clearance_m"] >= 0.005
    assert report["final_joint_error_rad"] <= 0.01
    assert report["final_tcp_position_error_mm"] <= 2
    assert report["final_tcp_orientation_error_deg"] <= 1
    assert report["max_grasp_shift_mm"] <= 5
    assert report["max_grasp_rotation_deg"] <= 5
    assert report["max_contact_loss_s"] <= 0.1
    assert report["placement_error_mm"] <= 5
    assert report["placement_drift_mm"] <= 1
    assert report["placement_tilt_deg"] <= 5
    assert report["placement_stable_s"] >= 1
    assert report["final_payload_position_m"][2] == pytest.approx(0.025, abs=0.001)
    stages = [item["stage"] for item in report["local_phases"]]
    assert stages.index("lift") < stages.index("place_descent") < stages.index("retreat")


def test_missing_bilateral_contact_refuses_lift_and_writes_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(demo.ObstacleRobot, "bilateral_finger_contact", lambda *_: False)
    path = tmp_path / "failed_grasp.json"
    assert demo.main(["--headless", "--report", str(path)]) == 1
    report = json.loads(path.read_text())
    assert report["failure_stage"] == "grasp"
    assert report["final_payload_position_m"][2] < 0.04
    assert "lift_mm" not in report


def test_slip_and_contact_loss_monitors(loaded_start, monkeypatch):
    qpos, carried, _ = loaded_start
    checker = p.SceneChecker(carried=carried)
    checker.actual_distance(qpos.copy())
    monitor = m.PayloadMonitor(carried)
    moved = qpos.copy()
    moved[checker.payload_adr] += 0.01
    checker.actual_distance(moved)
    with pytest.raises(p.MotionError, match="slipped"):
        monitor.observe(checker.model, checker.data, 0.001)
    checker.actual_distance(qpos.copy())
    monitor = m.PayloadMonitor(carried)
    monkeypatch.setattr(m, "bodies_in_contact", lambda *_args, **_kwargs: False)
    with pytest.raises(p.MotionError, match="bilateral contact lost"):
        for _ in range(101):
            monitor.observe(checker.model, checker.data, 0.001)


def test_runtime_collision_and_ungrasped_replay_abort(plans, loaded_start):
    plan, _ = plans()
    with demo.ObstacleRobot(viewer=False) as robot:
        with pytest.raises(p.MotionError, match="verified physical grasp"):
            demo.replay(robot, plan, p.Options(), {})
        robot.runtime_checker = p.SceneChecker()
        robot.data.qpos[:] = loaded_start[0]
        robot.data.qpos[
            robot.runtime_checker.payload_adr : robot.runtime_checker.payload_adr + 3
        ] = [0.35, 0, 0.3]
        robot.gripper_expected = "holding"
        with pytest.raises(p.MotionError, match="actual clearance"):
            robot.observe()


def test_viewer_close_and_visual_overlays(plans):
    plan, _ = plans()
    with demo.ObstacleRobot(viewer=False) as robot:
        scene = mujoco.MjvScene(robot.model, maxgeom=1000)
        demo.draw_overlays(
            scene, robot.fk_positions(plan.baseline), robot.fk_positions(plan.path), p.Options()
        )
        assert {g.label for g in scene.geoms[: scene.ngeom]} >= {"start", "goal"}

        class ClosingViewer:
            calls = 0

            def is_running(self):
                self.calls += 1
                return self.calls < 2

            def sync(self):
                pass

        robot._viewer = ClosingViewer()
        try:
            with pytest.raises(TaskError, match="Viewer window was closed"):
                robot._step()
        finally:
            robot._viewer = None


def test_cli_precedence_and_sim_only(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"headless": True, "seed": 5, "task": {"goal_pose": p.START_POSE}})
    )
    parser = demo.build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(config),
            "--seed",
            "7",
            "--no-headless",
            "--goal-pose",
            *map(str, p.GOAL_POSE),
        ]
    )
    opts, headless, _ = demo._load_options(args, {})
    assert opts.seed == 7 and not headless and opts.goal_pose == list(p.GOAL_POSE)
    for mode in ("real", "sdk-sim", "dry-run"):
        with pytest.raises(SystemExit) as failure:
            parser.parse_args(["--mode", mode])
        assert failure.value.code == 2


def test_invalid_input_writes_standard_json(tmp_path):
    path = tmp_path / "invalid.json"
    assert (
        demo.main(
            ["--headless", "--report", str(path), "--start-pose", "nan", "0", "0", "180", "0", "0"]
        )
        == 1
    )
    report = json.loads(path.read_text())
    assert not report["success"] and report["failure_stage"] == "configuration"
    assert report["inputs"]["start_pose"][0] is None


@pytest.mark.parametrize(
    "values",
    [
        {"seed": -1},
        {"max_iterations": 0},
        {"edge_resolution_rad": 0.03},
        {"max_velocity_rad_s": np.nan},
        {"max_velocity_rad_s": "invalid"},
        {"execution_clearance_m": 0.02},
        {"timestep_s": 0.002},
    ],
)
def test_invalid_options(values):
    with pytest.raises(ValueError):
        replace(p.Options(), **values).validate()
