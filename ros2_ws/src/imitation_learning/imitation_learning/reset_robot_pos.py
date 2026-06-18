"""Reset the robot to the start pose in both Gazebo and the localization stack.

ROS 2 port of the original ``reset_robot_pos`` script.  Listens on the
``reset_robot_pos`` topic; whenever ``True`` is received it:

* stops the robot,
* teleports the model in Gazebo via the ``/set_entity_state`` service
  (``gazebo_msgs/srv/SetEntityState``, provided by the ``gazebo_ros_state``
  plugin), and
* re-initialises AMCL by publishing to ``/initialpose``.

All poses and names are parameters so the node is not tied to one world.
"""

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from std_msgs.msg import Bool

import rclpy
from rclpy.node import Node


class ResetRobotPos(Node):
    def __init__(self):
        super().__init__("reset_robot_pos")
        self.declare_parameter("entity_name", "turtlebot3_waffle")
        self.declare_parameter("set_state_service", "/set_entity_state")
        # Start pose (room A) in the map / world frame.
        self.declare_parameter("start_x", -6.503)
        self.declare_parameter("start_y", -2.422)
        self.declare_parameter("start_qz", -0.77)
        self.declare_parameter("start_qw", -0.63)

        self.pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.pub_initialpose = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10)
        self.create_subscription(Bool, "reset_robot_pos", self._reset_cb, 10)

        service = self.get_parameter("set_state_service").value
        self.set_state_client = self.create_client(SetEntityState, service)

        # Reset once on startup so the robot begins at room A.
        self.create_timer(0.8, self._initial_reset_once)
        self._did_initial = False
        self.get_logger().info("reset_robot_pos ready.")

    def _initial_reset_once(self):
        if not self._did_initial:
            self._did_initial = True
            self.reset_robot_position()

    def _reset_cb(self, msg):
        if msg.data:
            self.reset_robot_position()

    def reset_robot_position(self):
        # Stop the robot.
        self.pub_cmd.publish(Twist())

        sx = self.get_parameter("start_x").value
        sy = self.get_parameter("start_y").value
        qz = self.get_parameter("start_qz").value
        qw = self.get_parameter("start_qw").value

        # Teleport in Gazebo.
        if self.set_state_client.wait_for_service(timeout_sec=2.0):
            state = EntityState()
            state.name = self.get_parameter("entity_name").value
            state.pose.position.x = float(sx)
            state.pose.position.y = float(sy)
            state.pose.position.z = 0.0
            state.pose.orientation.z = float(qz)
            state.pose.orientation.w = float(qw)
            state.reference_frame = "world"
            req = SetEntityState.Request()
            req.state = state
            self.set_state_client.call_async(req)
        else:
            self.get_logger().warn(
                "set_entity_state service unavailable; skipping Gazebo teleport.")

        # Re-initialise AMCL.
        init = PoseWithCovarianceStamped()
        init.header.frame_id = "map"
        init.header.stamp = self.get_clock().now().to_msg()
        init.pose.pose.position.x = float(sx)
        init.pose.pose.position.y = float(sy)
        init.pose.pose.orientation.z = float(qz)
        init.pose.pose.orientation.w = float(qw)
        init.pose.covariance[0] = 0.25
        init.pose.covariance[7] = 0.25
        init.pose.covariance[35] = 0.07
        self.pub_initialpose.publish(init)
        self.get_logger().info("Reset to start pose done.")


def main(args=None):
    rclpy.init(args=args)
    node = ResetRobotPos()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
