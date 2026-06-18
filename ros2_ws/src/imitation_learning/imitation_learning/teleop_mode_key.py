"""Keyboard helper to switch the learner's mode via the ``mode_change`` service.

ROS 2 port of the mode-switching part of the original ``turtlebot3_teleop_key``.

Key bindings:
    0 : IDLE
    1 : COLLECT
    2 : TRAIN
    3 : EXECUTE
    4 : CLEARDATA
    CTRL-C : quit
"""

import sys
import termios
import tty

from il_interfaces.srv import Mode

import rclpy
from rclpy.node import Node

MSG = """
Imitation-learning mode control
-------------------------------
  0 : IDLE
  1 : COLLECT   (gather expert demonstrations)
  2 : TRAIN     (train the policy on collected data)
  3 : EXECUTE   (run the learned policy to the goal)
  4 : CLEARDATA (delete all recorded trajectories)

  CTRL-C to quit
"""

MODE_NAMES = {0: "IDLE", 1: "COLLECT", 2: "TRAIN", 3: "EXECUTE", 4: "CLEARDATA"}


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class TeleopModeKey(Node):
    def __init__(self):
        super().__init__("teleop_mode_key")
        self.cli = self.create_client(Mode, "mode_change")

    def change_mode(self, m):
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("mode_change service not available.")
            return
        req = Mode.Request()
        req.reqmode = int(m)
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None:
            new_mode = future.result().setmode
            self.get_logger().info(f"New mode: {MODE_NAMES.get(new_mode, new_mode)}")
        else:
            self.get_logger().warn("mode_change service call failed.")


def main(args=None):
    rclpy.init(args=args)
    node = TeleopModeKey()
    settings = termios.tcgetattr(sys.stdin)
    print(MSG)
    try:
        while True:
            key = get_key(settings)
            if key in ("0", "1", "2", "3", "4"):
                node.change_mode(int(key))
            elif key == "\x03":  # CTRL-C
                break
    except Exception as e:  # noqa: BLE001
        node.get_logger().error(str(e))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
