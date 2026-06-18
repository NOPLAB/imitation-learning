"""Shared rclpy node for the imitation-learning experiments.

This base class holds everything the supervised ``learner`` and the
``daglearner`` have in common:

* subscriptions to the laser scan, wheel odometry, AMCL pose and the expert
  command stream (``/cmd_vel_temp`` -- the Nav2 controller output, remapped so
  we can intercept it),
* publishers for the robot command (``/cmd_vel``) and the reset trigger,
* a ``mode_change`` service used by the teleop helper to switch between
  IDLE / COLLECT / TRAIN / EXECUTE / CLEARDATA,
* a Nav2 ``NavigateToPose`` action client that drives the expert planner, and
* a fixed-rate timer that ticks a small state machine.

Unlike the original ROS 1 code (which used blocking ``while`` loops with
``rate.sleep()``), each mode advances one step per timer tick so the rclpy
executor stays responsive.
"""

import json
import os

import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose

from il_interfaces.srv import Mode
from imitation_learning import model_il
from imitation_learning.model_il import ImitationNet

# Mode constants (kept identical to the original project)
IDLE = 0
COLLECT = 1
TRAIN = 2
EXECUTE = 3
CLEARDATA = 4


class ImitationLearnerNode(Node):
    """Common machinery; subclasses implement :meth:`select_command`."""

    #: ``"supervised"`` or ``"dagger"`` -- set by the subclass.
    mode_name = "supervised"

    def __init__(self, node_name: str):
        super().__init__(node_name)

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        pkg_default = os.path.expanduser("~/il_data")
        self.declare_parameter("data_dir", os.path.join(pkg_default, "data"))
        self.declare_parameter("model_dir", os.path.join(pkg_default, "models"))
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("device", "auto")
        # Goal pose (map frame) -- the room the expert drives towards.
        self.declare_parameter("goal_x", 6.0)
        self.declare_parameter("goal_y", -1.0)
        self.declare_parameter("goal_qz", 0.7071)
        self.declare_parameter("goal_qw", -0.7071)
        self.declare_parameter("goal_tol_x", 0.5)
        self.declare_parameter("goal_tol_y", 0.3)

        data_root = self.get_parameter("data_dir").value
        model_root = self.get_parameter("model_dir").value
        self.data_dir = os.path.join(data_root, self.mode_name) + os.sep
        self.model_dir = os.path.join(model_root, self.mode_name) + os.sep
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.device = self._resolve_device(self.get_parameter("device").value)
        self.get_logger().info(f"Using device: {self.device}")

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self.mode = IDLE
        self.expert_cmd = None        # Twist from /cmd_vel_temp
        self.laser = None             # LaserScan
        self.pos = None               # PoseWithCovarianceStamped (map frame, AMCL)
        self.true_pos = None          # Odometry
        self.robot_cmd = Twist()
        self.num_traj = 0
        self.D = self._empty_dataset()

        # Episode / action bookkeeping
        self._goal_handle = None
        self._goal_pending = False

        self.model = ImitationNet(control_dim=2, device=self.device).to(self.device)

        # ------------------------------------------------------------------ #
        # ROS interfaces
        # ------------------------------------------------------------------ #
        self.create_subscription(LaserScan, "scan", self._laser_cb, 10)
        self.create_subscription(Odometry, "odom", self._odom_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, "amcl_pose", self._amcl_cb, 10)
        self.create_subscription(Twist, "/cmd_vel_temp", self._expert_cb, 10)

        self.pub_reset = self.create_publisher(Bool, "reset_robot_pos", 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_service(Mode, "mode_change", self._handle_mode_change)

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        rate = float(self.get_parameter("control_rate").value)
        self.create_timer(1.0 / rate, self._on_timer)

        self.get_logger().info(
            f"{node_name} ready (mode='{self.mode_name}'). "
            "Use the teleop_mode_key node to switch modes."
        )

    # ====================================================================== #
    # Callbacks
    # ====================================================================== #
    def _expert_cb(self, msg):
        self.expert_cmd = msg

    def _laser_cb(self, msg):
        self.laser = msg

    def _amcl_cb(self, msg):
        self.pos = msg

    def _odom_cb(self, msg):
        self.true_pos = msg

    def _handle_mode_change(self, request, response):
        reqmode = int(request.reqmode)
        names = {IDLE: "IDLE", COLLECT: "COLLECT", TRAIN: "TRAIN",
                 EXECUTE: "EXECUTE", CLEARDATA: "CLEARDATA"}
        if reqmode in names:
            self.get_logger().info(f"Mode change requested: {names[reqmode]}")
            if reqmode == CLEARDATA:
                self._clear_data_files()
                self.mode = IDLE
            else:
                # Switching mode aborts any in-progress episode.
                self._reset_episode_state()
                self.mode = reqmode
        else:
            self.get_logger().warn(f"Unknown mode {reqmode}; keeping current mode.")
        response.setmode = self.mode
        return response

    # ====================================================================== #
    # State machine
    # ====================================================================== #
    def _on_timer(self):
        if self.mode == COLLECT:
            self._collect_step()
        elif self.mode == TRAIN:
            self._train_step()
        elif self.mode == EXECUTE:
            self._execute_step()
        # IDLE: do nothing.

    def _collect_step(self):
        # Kick off the expert planner once per episode.
        if self._goal_handle is None and not self._goal_pending:
            self._send_goal()
            return
        if not self._have_observation():
            return

        if self._goal_reached():
            self._stop_robot()
            self.on_episode_complete()          # subclass hook (beta decay)
            self._cancel_goal()
            self._finalize_episode()
            self.get_logger().info("Demonstration complete. Resetting robot.")
            self._reset_robot()
            self._reset_episode_state()
            # Stay in COLLECT: the next tick starts a fresh episode.
            return

        cmd = self.select_command()             # subclass hook
        if cmd is not None:
            self.pub_cmd.publish(cmd)
            self._record_step()

    def _train_step(self):
        self.get_logger().info("Training started ...")
        model_path = os.path.join(self.model_dir, "model.pt")
        if os.path.exists(model_path):
            self.model.load(model_path)
        self.model = self.model.to(self.device)
        self.model = model_il.train(
            self.model,
            data_dir=self.data_dir,
            model_dir=self.model_dir,
            mode=self.mode_name,
            device=self.device,
            logger=self.get_logger(),
        )
        self.model.eval()
        self.get_logger().info("Training complete. Resetting robot, returning to IDLE.")
        self._reset_robot()
        self.mode = IDLE

    def _execute_step(self):
        if not self._have_observation(require_expert=False):
            return
        # Load the freshest weights at the start of an execution run.
        if self._goal_handle is None and not self._goal_pending:
            model_path = os.path.join(self.model_dir, "model.pt")
            if os.path.exists(model_path):
                self.model.load(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._goal_pending = True  # reuse flag to mark "execution started"
            self.get_logger().info("Starting policy execution.")

        vel = model_il.predict(self.model, self._odom_vector(), self._laser_matrix(), self.device)
        self.robot_cmd.linear.x = float(vel[0, 0])
        self.robot_cmd.angular.z = float(vel[0, 1])
        self.pub_cmd.publish(self.robot_cmd)

        if self._goal_reached():
            self._stop_robot()
            self.get_logger().info("Goal reached. Resetting robot, returning to IDLE.")
            self._reset_robot()
            self._reset_episode_state()
            self.mode = IDLE

    # ====================================================================== #
    # Hooks for subclasses
    # ====================================================================== #
    def select_command(self) -> Twist:
        """Return the Twist to publish this tick (and update self.robot_cmd)."""
        raise NotImplementedError

    def on_episode_complete(self):
        """Called once when an episode reaches the goal (e.g. beta decay)."""
        pass

    # ====================================================================== #
    # Helpers
    # ====================================================================== #
    def _have_observation(self, require_expert: bool = True) -> bool:
        if self.pos is None or self.laser is None:
            return False
        if require_expert and self.expert_cmd is None:
            return False
        return True

    def _goal_reached(self) -> bool:
        gx = self.get_parameter("goal_x").value
        gy = self.get_parameter("goal_y").value
        tx = self.get_parameter("goal_tol_x").value
        ty = self.get_parameter("goal_tol_y").value
        px = self.pos.pose.pose.position.x
        py = self.pos.pose.pose.position.y
        return abs(px - gx) < tx and abs(py - gy) < ty

    def _odom_vector(self):
        p = self.pos.pose.pose
        return np.array([[p.position.x, p.position.y, p.position.z,
                          p.orientation.x, p.orientation.y,
                          p.orientation.z, p.orientation.w]])

    def _laser_matrix(self):
        return np.array([list(self.laser.ranges)], dtype=np.float32)

    def _record_step(self):
        p = self.pos.pose.pose
        self.D["robot_pos"].append([
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
        ])
        self.D["laser_scan"].append([float(r) for r in self.laser.ranges])
        self.D["robot_vel"].append([self.expert_cmd.linear.x, self.expert_cmd.angular.z])

    def _finalize_episode(self):
        if len(self.D["robot_vel"]) == 0:
            return
        file_name = os.path.join(self.data_dir, f"trajectory_{self.num_traj}.json")
        while os.path.exists(file_name):
            self.num_traj += 1
            file_name = os.path.join(self.data_dir, f"trajectory_{self.num_traj}.json")
        with open(file_name, "w") as fout:
            json.dump(self.D, fout)
        self.get_logger().info(f"Saved {len(self.D['robot_vel'])} samples to {file_name}")
        self.num_traj += 1
        self.D = self._empty_dataset()

    def _send_goal(self):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Nav2 'navigate_to_pose' server not available yet.")
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(self.get_parameter("goal_x").value)
        goal.pose.pose.position.y = float(self.get_parameter("goal_y").value)
        goal.pose.pose.orientation.z = float(self.get_parameter("goal_qz").value)
        goal.pose.pose.orientation.w = float(self.get_parameter("goal_qw").value)
        self._goal_pending = True
        self.get_logger().info("Sending expert (Nav2) goal.")
        self.nav_client.send_goal_async(goal).add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Nav2 rejected the goal.")
            self._goal_pending = False
            return
        self._goal_handle = handle

    def _cancel_goal(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

    def _reset_episode_state(self):
        self._goal_handle = None
        self._goal_pending = False

    def _stop_robot(self):
        stop = Twist()
        self.pub_cmd.publish(stop)

    def _reset_robot(self):
        msg = Bool(data=True)
        self.pub_reset.publish(msg)
        self.pub_reset.publish(msg)  # repeat to be sure it lands

    def _clear_data_files(self):
        removed = 0
        for f in os.listdir(self.data_dir):
            if f.startswith("trajectory_") and f.endswith(".json"):
                os.remove(os.path.join(self.data_dir, f))
                removed += 1
        self.num_traj = 0
        self.D = self._empty_dataset()
        self.get_logger().info(f"Cleared {removed} trajectory file(s).")

    @staticmethod
    def _empty_dataset():
        return {"robot_pos": [], "laser_scan": [], "robot_vel": []}

    @staticmethod
    def _resolve_device(requested):
        import torch
        if requested == "cpu":
            return "cpu"
        if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
