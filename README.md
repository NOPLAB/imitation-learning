# Imitation Learning for Turtlebot3 (ROS 2 Humble)

[jeffrey-fong/ImitationLearning-Turtlebot3](https://github.com/jeffrey-fong/ImitationLearning-Turtlebot3)
（ROS 1 / catkin）を **ROS 2 Humble** 向けに全面的に書き直したものです。
Gazebo シミュレーション上で Turtlebot3 を「ある部屋から別の部屋へ」自律走行させるための
模倣学習（Imitation Learning）を実装しています。**Nav2 ナビゲーションスタックをエキスパート方策**
として使い、小さな PyTorch ニューラルネットに走行方策を学習させます。

学習アルゴリズムは 2 種類：

- **DAgger（Dataset Aggregation）** — 学習済み方策とエキスパートを確率的に混合しながらデータを集め、
  反復的に学習する。分布シフトに強い。
- **教師あり学習（Supervised / Behavior Cloning）** — エキスパートのデモを集めてからまとめて学習する。

---

## ROS 1 → ROS 2 でのおもな変更点

| ROS 1（原版） | ROS 2 Humble（本リポジトリ） |
|---|---|
| `rospy` | `rclpy` |
| `catkin_make` | `colcon build` |
| `move_base` + `actionlib`（エキスパート） | **Nav2** `NavigateToPose` アクション |
| `gazebo_msgs/SetModelState` | `gazebo_msgs/srv/SetEntityState` |
| `tf` で map→odom をブロードキャスト | `/initialpose` で AMCL を初期化 |
| `rospkg.RosPack` / `os.chdir` | `ament_index` + パラメータ（`data_dir` / `model_dir`） |
| ブロッキング `while` + `rospy.Rate` | 20 Hz タイマーで駆動するステートマシン |
| カスタム `srv`（`imitation_learning/Mode`） | 独立パッケージ `il_interfaces/Mode` |
| Python 2 系混在 | Python 3 / 型ヒント |

---

## パッケージ構成

```
ros2_ws/src/
├── il_interfaces/                 # Mode.srv（モード切替サービス）
│   ├── srv/Mode.srv
│   ├── CMakeLists.txt
│   └── package.xml
└── imitation_learning/            # ament_python パッケージ
    ├── imitation_learning/
    │   ├── neural_network.py      # PoseToControl（FC ネット）
    │   ├── model_il.py            # ImitationNet + train / predict
    │   ├── dataset.py             # 軌跡 JSON を読む Dataset
    │   ├── base_node.py           # 学習ノード共通基底（ステートマシン）
    │   ├── learner.py             # 教師あり学習ノード
    │   ├── daglearner.py          # DAgger ノード
    │   ├── policyexec.py          # 学習済み方策の単体実行
    │   ├── reset_robot_pos.py     # ロボットを開始位置へリセット
    │   └── teleop_mode_key.py     # キーボードでモード切替
    ├── launch/
    │   ├── simulation.launch.py   # Gazebo(house) + Nav2（cmd_vel リマップ）
    │   └── il_nodes.launch.py     # reset + learner/dagger
    ├── config/nav2_params.yaml    # Nav2 パラメータ（waffle）
    └── maps/house_map.{yaml,pgm}  # プレースホルダ地図（要差し替え）
```

### ネットワーク入出力

- **入力（19 次元）**: ロボット姿勢 7 次元（位置 xyz + 四元数 xyzw）+ レーザースキャン 12 次元
  （360 ビームを 30 本ごとにサブサンプリング）。
- **出力（2 次元）**: `linear.x`, `angular.z`。
- 損失: 予測速度とエキスパート速度の二乗誤差和。

### エキスパート速度の横取り（重要）

Nav2 のコントローラ出力（`/cmd_vel`）を **`/cmd_vel_temp` にリマップ**します
（`simulation.launch.py` 内の `SetRemap`）。学習ノードは `/cmd_vel_temp` を購読してエキスパート指令を受け取り、
実際にロボットへ送る指令を判断して `/cmd_vel` に publish します。

---

## モード（`mode_change` サービス / `teleop_mode_key`）

| キー | モード | 動作 |
|---|---|---|
| `0` | IDLE | 待機 |
| `1` | COLLECT | エキスパートのデモを収集（DAgger は方策と混合） |
| `2` | TRAIN | 収集済みデータで学習し `model.pt` を保存 |
| `3` | EXECUTE | 学習済み方策でゴールまで走行 |
| `4` | CLEARDATA | 収集済み軌跡を全削除 |

---

## Docker で動かす（推奨）

GUI（Gazebo / RViz）をホストの X サーバへ転送して実行します。

```bash
# 1. X サーバへのアクセスを許可
xhost +local:root

# 2. イメージをビルド
docker compose build
#   もしくは: docker build -t imitation-learning-tb3:humble .

# 3. コンテナに入る
docker compose run --rm il
#   もしくは:
#   docker run -it --rm --net=host \
#     -e DISPLAY=$DISPLAY -e TURTLEBOT3_MODEL=waffle \
#     -v /tmp/.X11-unix:/tmp/.X11-unix imitation-learning-tb3:humble bash
```

ベースイメージは `osrf/ros:humble-desktop-full`。Nav2 / Turtlebot3 / Gazebo と
CPU 版 PyTorch を導入し、`colcon build` 済みです。GPU を使う場合は CUDA 版 `torch` を入れ、
ノード起動時に `device:=cuda` を指定してください。

---

## 実行手順（コンテナ内 / ネイティブ共通）

各端末で workspace を source します（コンテナでは `.bashrc` で自動）：

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash     # ネイティブなら <ws>/install/setup.bash
export TURTLEBOT3_MODEL=waffle
```

**端末 1 — シミュレーション + Nav2（エキスパート）**

```bash
ros2 launch imitation_learning simulation.launch.py
```

**端末 2 — リセットノード + 学習ノード**

```bash
# DAgger（既定）
ros2 launch imitation_learning il_nodes.launch.py algo:=dagger
# 教師あり学習
ros2 launch imitation_learning il_nodes.launch.py algo:=supervised
```

**端末 3 — モード切替（キーボード）**

```bash
ros2 run imitation_learning teleop_mode_key
```

`1`（COLLECT）でデモを何回か収集 → `2`（TRAIN）で学習 → `3`（EXECUTE）で評価、を繰り返します。
DAgger では COLLECT → TRAIN を反復するほど `beta` が減衰し、学習方策の割合が増えます。

### 学習済み方策の単体実行

Nav2 を使わず、保存済みモデルだけで走行：

```bash
ros2 run imitation_learning policyexec \
  --ros-args -p model_dir:=$HOME/il_data/models -p model_file:=best_model.pt
```

---

## おもなパラメータ

| パラメータ | 既定値 | 説明 |
|---|---|---|
| `data_dir` | `~/il_data/data` | 軌跡 JSON の保存先（`<dir>/<algo>/`） |
| `model_dir` | `~/il_data/models` | モデルの保存先（`<dir>/<algo>/`） |
| `device` | `auto` | `auto` / `cpu` / `cuda` |
| `goal_x`, `goal_y`, `goal_qz`, `goal_qw` | `6.0, -1.0, 0.7071, -0.7071` | ゴール姿勢（map 座標） |
| `goal_tol_x`, `goal_tol_y` | `0.5, 0.3` | ゴール到達判定の許容誤差 |
| `beta`, `beta_decay` | `0.9, 0.95` | DAgger の混合率と減衰率 |
| `entity_name` | `turtlebot3_waffle` | Gazebo 上のモデル名（リセット用） |
| `start_x`, `start_y`, `start_qz`, `start_qw` | `-6.503, -2.422, -0.77, -0.63` | 開始姿勢（reset_robot_pos） |

例：

```bash
ros2 run imitation_learning daglearner --ros-args \
  -p goal_x:=6.0 -p goal_y:=-1.0 -p device:=cpu
```

---

## 地図について（要差し替え）

`maps/house_map.pgm` は **空のプレースホルダ**です（外周だけ壁）。
意味のあるナビゲーションのためには、使用するワールドの実地図に差し替えてください。
[slam_toolbox](https://github.com/SteveMacenski/slam_toolbox) などで作成できます：

```bash
# 端末A: Gazebo を起動（teleop で走り回る）
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
ros2 run turtlebot3_teleop teleop_keyboard
# 端末B: SLAM
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true
# 端末C: 保存
ros2 run nav2_map_server map_saver_cli -f \
  ros2_ws/src/imitation_learning/maps/house_map
```

作成した地図に合わせて `goal_*` / `start_*` 座標も調整してください。

---

## ネイティブビルド（Docker を使わない場合）

ROS 2 Humble・Nav2・Turtlebot3・Gazebo・PyTorch が導入済みであることが前提です。

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

## ライセンス

MIT
