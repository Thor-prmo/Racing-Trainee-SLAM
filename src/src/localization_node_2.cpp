#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/float32.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace
{

constexpr double kDefaultWheelRadius = 0.2525;
constexpr double kMinDt = 1.0e-4;
constexpr double kMaxDt = 1.0;

constexpr double kInitCovariance = 0.001;

constexpr double kProcessXy = 0.002;
constexpr double kProcessYaw = 0.005;
constexpr double kProcessVelocity = 0.02;
constexpr double kProcessYawRate = 0.02;

constexpr double kWheelVariance = 0.0025;
constexpr double kImuYawRateVariance = 0.001;
constexpr double kChiSquare95Scale2d = 2.44774683068;

double angle_wrap(double value)
{
  while (value > M_PI) {
    value -= 2.0 * M_PI;
  }
  while (value < -M_PI) {
    value += 2.0 * M_PI;
  }
  return value;
}

geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw * 0.5);
  q.w = std::cos(yaw * 0.5);
  return q;
}

geometry_msgs::msg::Point make_point(double x, double y, double z = 0.0)
{
  geometry_msgs::msg::Point p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

Eigen::Matrix2d rotation_2d(double yaw)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);

  Eigen::Matrix2d rotation;
  rotation << c, -s,
              s,  c;
  return rotation;
}

rclcpp::QoS bag_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
}

std::vector<std::string> split_csv_line(const std::string & line)
{
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;

  while (std::getline(stream, field, ',')) {
    fields.push_back(field);
  }

  return fields;
}

bool contains_token(const std::string & value, const std::string & token)
{
  return value.find(token) != std::string::npos;
}

}  // namespace

class LocalizationNode2 : public rclcpp::Node
{
public:
  LocalizationNode2()
  : Node("localization_node_2"),
    state_(State::Zero()),
    covariance_(StateMatrix::Identity() * kInitCovariance),
    prev_imu_time_(0, 0, RCL_ROS_TIME),
    first_imu_(true)
  {
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link_kf");
    observation_topic_ = declare_parameter<std::string>("observation_topic", "/ground_truth_cones");
    map_csv_param_ = declare_parameter<std::string>("map_csv", "small_track.csv");
    normalize_to_start_ = declare_parameter<bool>("normalize_to_start", true);
    observations_are_local_ = declare_parameter<bool>("observations_are_local", false);
    use_color_association_ = declare_parameter<bool>("use_color_association", true);
    publish_landmark_uncertainty_ = declare_parameter<bool>("publish_landmark_uncertainty", true);
    association_gate_ =declare_parameter<double>("association_gate", 1.5);
    cone_measurement_stddev_ =declare_parameter<double>("cone_measurement_stddev", 0.01);
    covariance_scale_ = declare_parameter<double>("covariance_scale", kChiSquare95Scale2d);
    min_covariance_diameter_ = declare_parameter<double>("min_covariance_diameter", 0.05);
    max_covariance_diameter_ = declare_parameter<double>("max_covariance_diameter", 3.0);
    wheel_speed_scale_ = declare_parameter<double>(
      "wheel_speed_scale", kDefaultWheelRadius * M_PI / 30.0);
    const double publish_rate_hz = declare_parameter<double>("publish_rate_hz", 50.0);

    load_track_csv(resolve_map_path(map_csv_param_));

    pub_map_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/localization_2/map_markers", 10);
    pub_landmark_uncertainty_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/localization_2/landmark_uncertainty_markers", 10);
    pub_observations_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/localization_2/observation_markers", 10);
    pub_associations_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/localization_2/association_markers", 10);
    pub_state_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/localization_2/state_markers", 10);
    pub_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "/localization_2/pose", 10);
    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>(
      "/localization_2/odometry", 10);
    pub_path_ = create_publisher<nav_msgs::msg::Path>(
      "/localization_2/path", 10);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    sub_wheel_ = create_subscription<std_msgs::msg::Float32>(
      "/wheel_speed_avg", bag_qos(),
      [this](const std_msgs::msg::Float32::SharedPtr msg) {
        wheel_update(static_cast<double>(msg->data) * wheel_speed_scale_);
      });

    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu", bag_qos(),
      std::bind(&LocalizationNode2::imu_cb, this, std::placeholders::_1));

    sub_observations_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      observation_topic_, bag_qos(),
      std::bind(&LocalizationNode2::observation_cb, this, std::placeholders::_1));

    const auto publish_period = std::chrono::duration<double>(
      1.0 / std::max(1.0, publish_rate_hz));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(publish_period),
      std::bind(&LocalizationNode2::publish_all, this));

    RCLCPP_INFO(get_logger(), "=========================================");
    RCLCPP_INFO(get_logger(), " LocalizationNode2 - KF measurement update");
    RCLCPP_INFO(get_logger(), "  CSV map: %s", map_csv_path_.c_str());
    RCLCPP_INFO(get_logger(), "  Loaded %zu map cones from small_track.csv", map_cones_.size());
    RCLCPP_INFO(get_logger(), "  Observation topic: %s", observation_topic_.c_str());
    RCLCPP_INFO(
      get_logger(), "  Observation frame: %s",
      observations_are_local_ ? "vehicle-local" : "map/global");
    RCLCPP_INFO(get_logger(), "  RViz map: /localization_2/map_markers");
    RCLCPP_INFO(
      get_logger(), "  RViz landmark uncertainty: /localization_2/landmark_uncertainty_markers");
    RCLCPP_INFO(get_logger(), "  RViz pose/path: /localization_2/state_markers, /localization_2/path");
    RCLCPP_INFO(get_logger(), "=========================================");
  }

private:
  using State = Eigen::Matrix<double, 5, 1>;
  using StateMatrix = Eigen::Matrix<double, 5, 5>;

  enum StateIndex : int
  {
    IDX_X = 0,
    IDX_Y = 1,
    IDX_YAW = 2,
    IDX_V = 3,
    IDX_YAW_RATE = 4
  };

  struct Cone
  {
    std::string tag;
    Eigen::Vector2d position;
    Eigen::Matrix2d covariance;
  };

  struct Observation
  {
    Eigen::Vector2d measurement;
    Eigen::Vector2d map_frame_position;
    std::string tag;
  };

  struct Association
  {
    Observation observation;
    std::size_t cone_index;
    double distance;
  };

  State state_;
  StateMatrix covariance_;
  rclcpp::Time prev_imu_time_;
  bool first_imu_;

  std::string map_frame_;
  std::string base_frame_;
  std::string observation_topic_;
  std::string map_csv_param_;
  std::string map_csv_path_;
  bool normalize_to_start_;
  bool observations_are_local_;
  bool use_color_association_;
  bool publish_landmark_uncertainty_;
  double association_gate_;
  double cone_measurement_stddev_;
  double covariance_scale_;
  double min_covariance_diameter_;
  double max_covariance_diameter_;
  double wheel_speed_scale_;

  std::vector<Cone> map_cones_;
  std::vector<Observation> last_observations_;
  std::vector<Association> last_associations_;
  nav_msgs::msg::Path path_msg_;

  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_wheel_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr sub_observations_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_map_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_landmark_uncertainty_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_observations_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_associations_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_state_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::string resolve_map_path(const std::string & requested_path)
  {
    const std::filesystem::path requested(requested_path);
    if (std::filesystem::exists(requested)) {
      return requested.string();
    }

    try {
      const std::filesystem::path share_dir =
        ament_index_cpp::get_package_share_directory("slam_localization");
      const auto installed_path = share_dir / requested_path;
      if (std::filesystem::exists(installed_path)) {
        return installed_path.string();
      }

      const auto installed_default = share_dir / "small_track.csv";
      if (std::filesystem::exists(installed_default)) {
        return installed_default.string();
      }
    } catch (const std::exception &) {
      // Fall through to local paths while running from the source tree.
    }

    const auto cwd_path = std::filesystem::current_path() / requested_path;
    if (std::filesystem::exists(cwd_path)) {
      return cwd_path.string();
    }

    const auto cwd_default = std::filesystem::current_path() / "small_track.csv";
    if (std::filesystem::exists(cwd_default)) {
      return cwd_default.string();
    }

    throw std::runtime_error("Could not find map CSV: " + requested_path);
  }

  void load_track_csv(const std::string & csv_path)
  {
    map_csv_path_ = csv_path;

    std::ifstream file(csv_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open map CSV: " + csv_path);
    }

    struct RawRow
    {
      std::string tag;
      double x;
      double y;
      double direction;
      double x_variance;
      double y_variance;
      double xy_covariance;
    };

    std::string line;
    std::getline(file, line);

    std::vector<RawRow> rows;
    RawRow start{"car_start", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool found_start = false;

    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }

      const auto fields = split_csv_line(line);
      if (fields.size() < 7) {
        RCLCPP_WARN(get_logger(), "Skipping malformed CSV row: %s", line.c_str());
        continue;
      }

      RawRow row{
        fields[0],
        std::stod(fields[1]),
        std::stod(fields[2]),
        std::stod(fields[3]),
        std::stod(fields[4]),
        std::stod(fields[5]),
        std::stod(fields[6])};

      if (row.tag == "car_start") {
        start = row;
        found_start = true;
      } else {
        rows.push_back(row);
      }
    }

    if (!found_start) {
      RCLCPP_WARN(get_logger(), "No car_start row found in CSV; using origin as start pose.");
    }

    const Eigen::Vector2d start_position(start.x, start.y);
    const Eigen::Matrix2d world_to_start = rotation_2d(-start.direction);

    map_cones_.clear();
    map_cones_.reserve(rows.size());

    for (const auto & row : rows) {
      Eigen::Vector2d position(row.x, row.y);
      Eigen::Matrix2d covariance;
      covariance << row.x_variance, row.xy_covariance,
                    row.xy_covariance, row.y_variance;

      if (normalize_to_start_) {
        position = world_to_start * (position - start_position);
        covariance = world_to_start * covariance * world_to_start.transpose();
      }

      map_cones_.push_back(Cone{row.tag, position, covariance});
    }

    if (normalize_to_start_) {
      state_(IDX_X) = 0.0;
      state_(IDX_Y) = 0.0;
      state_(IDX_YAW) = 0.0;
    } else {
      state_(IDX_X) = start.x;
      state_(IDX_Y) = start.y;
      state_(IDX_YAW) = start.direction;
    }

    path_msg_.header.frame_id = map_frame_;
  }

  bool compatible_tag(const std::string & observation_tag, const Cone & cone) const
  {
    if (!use_color_association_ || observation_tag.empty()) {
      return true;
    }

    if (observation_tag == "orange") {
      return contains_token(cone.tag, "orange");
    }

    return observation_tag == cone.tag;
  }

  std::string marker_to_tag(const visualization_msgs::msg::Marker & marker) const
  {
    const std::string ns = marker.ns;
    if (contains_token(ns, "blue")) {
      return "blue";
    }
    if (contains_token(ns, "yellow")) {
      return "yellow";
    }
    if (contains_token(ns, "orange")) {
      return "orange";
    }

    const auto & color = marker.color;
    if (color.a <= 0.0f) {
      return "";
    }

    if (color.b > 0.5f && color.r < 0.5f) {
      return "blue";
    }
    if (color.r > 0.6f && color.g > 0.45f && color.b < 0.4f) {
      return "yellow";
    }
    if (color.r > 0.6f && color.g > 0.2f && color.g < 0.7f && color.b < 0.3f) {
      return "orange";
    }

    return "";
  }

  Eigen::Vector2d marker_point_to_xy(
    const visualization_msgs::msg::Marker & marker,
    const geometry_msgs::msg::Point & point) const
  {
    return Eigen::Vector2d(
      marker.pose.position.x + point.x,
      marker.pose.position.y + point.y);
  }

  bool marker_is_observation_candidate(const visualization_msgs::msg::Marker & marker) const
  {
    if (marker.action == visualization_msgs::msg::Marker::DELETE ||
      marker.action == visualization_msgs::msg::Marker::DELETEALL)
    {
      return false;
    }

    switch (marker.type) {
      case visualization_msgs::msg::Marker::SPHERE:
      case visualization_msgs::msg::Marker::CUBE:
      case visualization_msgs::msg::Marker::CYLINDER:
      case visualization_msgs::msg::Marker::MESH_RESOURCE:
      case visualization_msgs::msg::Marker::SPHERE_LIST:
      case visualization_msgs::msg::Marker::CUBE_LIST:
      case visualization_msgs::msg::Marker::POINTS:
        return true;
      default:
        return false;
    }
  }

  std::vector<Observation> extract_observations(
    const visualization_msgs::msg::MarkerArray & marker_array) const
  {
    std::vector<Observation> observations;

    for (const auto & marker : marker_array.markers) {
      if (!marker_is_observation_candidate(marker)) {
        continue;
      }

      const std::string tag = marker_to_tag(marker);

      if (!marker.points.empty()) {
        for (const auto & point : marker.points) {
          const Eigen::Vector2d measurement = marker_point_to_xy(marker, point);
          observations.push_back(make_observation(measurement, tag));
        }
      } else {
        const Eigen::Vector2d measurement(marker.pose.position.x, marker.pose.position.y);
        observations.push_back(make_observation(measurement, tag));
      }
    }

    return observations;
  }

  Observation make_observation(const Eigen::Vector2d & measurement, const std::string & tag) const
  {
    Observation observation;
    observation.measurement = measurement;
    observation.tag = tag;

    if (observations_are_local_) {
      observation.map_frame_position =
        Eigen::Vector2d(state_(IDX_X), state_(IDX_Y)) +
        rotation_2d(state_(IDX_YAW)) * measurement;
    } else {
      observation.map_frame_position = measurement;
    }

    return observation;
  }

  bool nearest_cone(
    const Observation & observation,
    std::size_t & best_index,
    double & best_distance) const
  {
    best_index = 0;
    best_distance = std::numeric_limits<double>::infinity();

    auto search = [&](bool enforce_tag) {
      bool found = false;
      for (std::size_t i = 0; i < map_cones_.size(); ++i) {
        if (enforce_tag && !compatible_tag(observation.tag, map_cones_[i])) {
          continue;
        }

        const double distance =
          (observation.map_frame_position - map_cones_[i].position).norm();
        if (distance < best_distance) {
          best_distance = distance;
          best_index = i;
          found = true;
        }
      }
      return found;
    };

    if (!search(true)) {
      search(false);
    }

    return std::isfinite(best_distance) && best_distance <= association_gate_;
  }

  void observation_cb(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
  {
    last_observations_ = extract_observations(*msg);
    last_associations_.clear();

    for (const auto & observation : last_observations_) {
      std::size_t cone_index = 0;
      double distance = 0.0;
      if (nearest_cone(observation, cone_index, distance)) {
        last_associations_.push_back(Association{observation, cone_index, distance});
      }
    }

    if (last_associations_.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "No cone observations passed the %.2f m association gate.", association_gate_);
      return;
    }

    if (observations_are_local_) {
      for (const auto & association : last_associations_) {
        local_cone_update(association);
      }
    } else {
      global_cone_update(last_associations_);
    }

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "cone KF update: observations=%zu associated=%zu x=%.3f y=%.3f yaw=%.2f deg",
      last_observations_.size(), last_associations_.size(),
      state_(IDX_X), state_(IDX_Y), state_(IDX_YAW) * 180.0 / M_PI);
  }

  void local_cone_update(const Association & association)
  {
    const auto & cone = map_cones_[association.cone_index];
    const double yaw = state_(IDX_YAW);
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    const double dx = cone.position.x() - state_(IDX_X);
    const double dy = cone.position.y() - state_(IDX_Y);

    Eigen::Vector2d predicted;
    predicted.x() = c * dx + s * dy;
    predicted.y() = -s * dx + c * dy;

    Eigen::Vector2d innovation = association.observation.measurement - predicted;

    Eigen::Matrix<double, 2, 5> h = Eigen::Matrix<double, 2, 5>::Zero();
    h(0, IDX_X) = -c;
    h(0, IDX_Y) = -s;
    h(0, IDX_YAW) = predicted.y();
    h(1, IDX_X) = s;
    h(1, IDX_Y) = -c;
    h(1, IDX_YAW) = -predicted.x();

    Eigen::Matrix2d r = Eigen::Matrix2d::Identity() *
      (cone_measurement_stddev_ * cone_measurement_stddev_);
    r += cone.covariance;

    kalman_update_2d(innovation, h, r);
  }

  void global_cone_update(const std::vector<Association> & associations)
  {
    Eigen::Vector2d residual = Eigen::Vector2d::Zero();
    Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();

    for (const auto & association : associations) {
      const auto & cone = map_cones_[association.cone_index];
      residual += cone.position - association.observation.map_frame_position;
      covariance += cone.covariance;
    }

    const double count = static_cast<double>(associations.size());
    residual /= count;

    Eigen::Matrix<double, 2, 5> h = Eigen::Matrix<double, 2, 5>::Zero();
    h(0, IDX_X) = 1.0;
    h(1, IDX_Y) = 1.0;

    Eigen::Matrix2d r = Eigen::Matrix2d::Identity() *
      (cone_measurement_stddev_ * cone_measurement_stddev_);
    r += covariance / std::max(1.0, count);

    kalman_update_2d(residual, h, r);
  }

  void wheel_update(double measured_v)
  {
    Eigen::Matrix<double, 1, 5> h = Eigen::Matrix<double, 1, 5>::Zero();
    h(0, IDX_V) = 1.0;

    const double innovation = measured_v - state_(IDX_V);
    scalar_kalman_update(innovation, h, kWheelVariance);
  }

  void imu_yaw_rate_update(double measured_yaw_rate)
  {
    Eigen::Matrix<double, 1, 5> h = Eigen::Matrix<double, 1, 5>::Zero();
    h(0, IDX_YAW_RATE) = 1.0;

    const double innovation = measured_yaw_rate - state_(IDX_YAW_RATE);
    scalar_kalman_update(innovation, h, kImuYawRateVariance);
  }

  void scalar_kalman_update(
    double innovation,
    const Eigen::Matrix<double, 1, 5> & h,
    double measurement_variance)
  {
    const double s = (h * covariance_ * h.transpose())(0, 0) + measurement_variance;
    if (s <= 0.0) {
      return;
    }

    const Eigen::Matrix<double, 5, 1> k = covariance_ * h.transpose() / s;
    const StateMatrix identity = StateMatrix::Identity();

    state_ += k * innovation;
    covariance_ =
      (identity - k * h) * covariance_ * (identity - k * h).transpose() +
      k * measurement_variance * k.transpose();
    state_(IDX_YAW) = angle_wrap(state_(IDX_YAW));
  }

  void kalman_update_2d(
    const Eigen::Vector2d & innovation,
    const Eigen::Matrix<double, 2, 5> & h,
    const Eigen::Matrix2d & r)
  {
    const Eigen::Matrix2d s = h * covariance_ * h.transpose() + r;
    const Eigen::LDLT<Eigen::Matrix2d> ldlt(s);
    if (ldlt.info() != Eigen::Success) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Skipping singular KF update.");
      return;
    }

    const Eigen::Matrix<double, 5, 2> k = ldlt.solve(h * covariance_).transpose();
    const StateMatrix identity = StateMatrix::Identity();

    state_ += k * innovation;
    covariance_ =
      (identity - k * h) * covariance_ * (identity - k * h).transpose() +
      k * r * k.transpose();
    state_(IDX_YAW) = angle_wrap(state_(IDX_YAW));
  }

  void predict(double dt)
  {
    if (dt < kMinDt || dt > kMaxDt) {
      return;
    }
    if (dt > kMaxDt) {
    RCLCPP_WARN(
        get_logger(),
        "Skipping prediction dt=%f",
        dt);
}

    const double yaw = state_(IDX_YAW);
    const double velocity = state_(IDX_V);
    const double yaw_rate = state_(IDX_YAW_RATE);

    State predicted = state_;
    predicted(IDX_X) += velocity * std::cos(yaw) * dt;
    predicted(IDX_Y) += velocity * std::sin(yaw) * dt;
    predicted(IDX_YAW) = angle_wrap(yaw + yaw_rate * dt);

    StateMatrix f = StateMatrix::Identity();
    f(IDX_X, IDX_YAW) = -velocity * std::sin(yaw) * dt;
    f(IDX_X, IDX_V) = std::cos(yaw) * dt;
    f(IDX_Y, IDX_YAW) = velocity * std::cos(yaw) * dt;
    f(IDX_Y, IDX_V) = std::sin(yaw) * dt;
    f(IDX_YAW, IDX_YAW_RATE) = dt;

    StateMatrix q = StateMatrix::Zero();
    q(IDX_X, IDX_X) = kProcessXy * dt;
    q(IDX_Y, IDX_Y) = kProcessXy * dt;
    q(IDX_YAW, IDX_YAW) = kProcessYaw * dt;
    q(IDX_V, IDX_V) = kProcessVelocity * dt;
    q(IDX_YAW_RATE, IDX_YAW_RATE) = kProcessYawRate * dt;

    state_ = predicted;
    covariance_ = f * covariance_ * f.transpose() + q;
    state_(IDX_YAW) = angle_wrap(state_(IDX_YAW));
  }

  void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    
    rclcpp::Time stamp(msg->header.stamp);
    if (stamp.nanoseconds() == 0) {
      stamp = now();
    }

    const double measured_yaw_rate = msg->angular_velocity.z;

    if (first_imu_) {
      prev_imu_time_ = stamp;
      first_imu_ = false;
      imu_yaw_rate_update(measured_yaw_rate);
      return;
    }

    const double dt = (stamp - prev_imu_time_).seconds();
    prev_imu_time_ = stamp;

    predict(dt);
    imu_yaw_rate_update(measured_yaw_rate);

    RCLCPP_DEBUG(
      get_logger(), "prediction: x=%.3f y=%.3f yaw=%.3f v=%.3f yaw_rate=%.3f",
      state_(IDX_X), state_(IDX_Y), state_(IDX_YAW), state_(IDX_V), state_(IDX_YAW_RATE));
  }

  void publish_all()
  {
    const rclcpp::Time stamp = now();

    publish_map(stamp);
    publish_landmark_uncertainty(stamp);
    publish_observations(stamp);
    publish_associations(stamp);
    publish_state(stamp);
    publish_pose_odom_path_tf(stamp);
  }

  std_msgs::msg::ColorRGBA color_for_tag(const std::string & tag, float alpha = 1.0f) const
  {
    std_msgs::msg::ColorRGBA color;
    color.a = alpha;

    if (tag == "blue") {
      color.r = 0.1f;
      color.g = 0.25f;
      color.b = 1.0f;
    } else if (tag == "yellow") {
      color.r = 1.0f;
      color.g = 0.85f;
      color.b = 0.05f;
    } else if (contains_token(tag, "orange")) {
      color.r = 1.0f;
      color.g = 0.35f;
      color.b = 0.0f;
    } else {
      color.r = 0.8f;
      color.g = 0.8f;
      color.b = 0.8f;
    }

    return color;
  }

  visualization_msgs::msg::Marker make_basic_marker(
    const std::string & ns,
    int id,
    int type,
    const rclcpp::Time & stamp) const
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = map_frame_;
    marker.header.stamp = stamp;
    marker.ns = ns;
    marker.id = id;
    marker.type = type;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.lifetime = rclcpp::Duration::from_seconds(0.0);
    return marker;
  }

  visualization_msgs::msg::Marker make_covariance_ellipse_marker(
    const std::string & ns,
    int id,
    const Eigen::Vector2d & center,
    const Eigen::Matrix2d & covariance,
    double z,
    const std_msgs::msg::ColorRGBA & color,
    const rclcpp::Time & stamp) const
  {
    auto marker = make_basic_marker(ns, id, visualization_msgs::msg::Marker::CYLINDER, stamp);

    const Eigen::Matrix2d symmetric_covariance =
      0.5 * (covariance + covariance.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(symmetric_covariance);

    double major_variance = std::max(0.0, symmetric_covariance(0, 0));
    double minor_variance = std::max(0.0, symmetric_covariance(1, 1));
    double yaw = 0.0;

    if (solver.info() == Eigen::Success) {
      const auto eigenvalues = solver.eigenvalues();
      const auto eigenvectors = solver.eigenvectors();
      const int major_index = eigenvalues(1) >= eigenvalues(0) ? 1 : 0;
      const int minor_index = major_index == 1 ? 0 : 1;
      const Eigen::Vector2d major_axis = eigenvectors.col(major_index);

      major_variance = std::max(0.0, eigenvalues(major_index));
      minor_variance = std::max(0.0, eigenvalues(minor_index));
      yaw = std::atan2(major_axis.y(), major_axis.x());
    }

    const double scale = std::max(0.0, covariance_scale_);
    marker.pose.position.x = center.x();
    marker.pose.position.y = center.y();
    marker.pose.position.z = z;
    marker.pose.orientation = yaw_to_quaternion(yaw);
    marker.scale.x = std::clamp(
      2.0 * scale * std::sqrt(major_variance),
      min_covariance_diameter_, max_covariance_diameter_);
    marker.scale.y = std::clamp(
      2.0 * scale * std::sqrt(minor_variance),
      min_covariance_diameter_, max_covariance_diameter_);
    marker.scale.z = 0.035;
    marker.color = color;

    return marker;
  }

  void publish_map(const rclcpp::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray array;
    array.markers.reserve(map_cones_.size());

    for (std::size_t i = 0; i < map_cones_.size(); ++i) {
      const auto & cone = map_cones_[i];
      auto marker = make_basic_marker("csv_map_" + cone.tag, static_cast<int>(i),
        visualization_msgs::msg::Marker::SPHERE, stamp);
      marker.pose.position.x = cone.position.x();
      marker.pose.position.y = cone.position.y();
      marker.pose.position.z = 0.15;
      marker.scale.x = contains_token(cone.tag, "orange") ? 0.55 : 0.35;
      marker.scale.y = marker.scale.x;
      marker.scale.z = 0.35;
      marker.color = color_for_tag(cone.tag, 0.95f);
      array.markers.push_back(marker);
    }

    pub_map_->publish(array);
  }

  void publish_landmark_uncertainty(const rclcpp::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray array;

    if (!publish_landmark_uncertainty_) {
      auto delete_all = make_basic_marker("landmark_covariance_95", 0,
        visualization_msgs::msg::Marker::CYLINDER, stamp);
      delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
      array.markers.push_back(delete_all);
      pub_landmark_uncertainty_->publish(array);
      return;
    }

    array.markers.reserve(map_cones_.size());

    for (std::size_t i = 0; i < map_cones_.size(); ++i) {
      const auto & cone = map_cones_[i];
      auto color = color_for_tag(cone.tag, 0.20f);
      auto ellipse = make_covariance_ellipse_marker(
        "landmark_covariance_95",
        static_cast<int>(i),
        cone.position,
        cone.covariance,
        0.02,
        color,
        stamp);
      array.markers.push_back(ellipse);
    }

    pub_landmark_uncertainty_->publish(array);
  }

  void publish_observations(const rclcpp::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray array;

    auto marker = make_basic_marker("associated_observations", 0,
      visualization_msgs::msg::Marker::SPHERE_LIST, stamp);
    marker.scale.x = 0.22;
    marker.scale.y = 0.22;
    marker.scale.z = 0.22;
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0f;

    for (const auto & association : last_associations_) {
      marker.points.push_back(make_point(
        association.observation.map_frame_position.x(),
        association.observation.map_frame_position.y(),
        0.55));
    }

    array.markers.push_back(marker);
    pub_observations_->publish(array);
  }

  void publish_associations(const rclcpp::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray array;

    auto marker = make_basic_marker("cone_associations", 0,
      visualization_msgs::msg::Marker::LINE_LIST, stamp);
    marker.scale.x = 0.04;
    marker.color.r = 0.0f;
    marker.color.g = 0.9f;
    marker.color.b = 0.2f;
    marker.color.a = 0.85f;

    for (const auto & association : last_associations_) {
      const auto & cone = map_cones_[association.cone_index];
      marker.points.push_back(make_point(
        association.observation.map_frame_position.x(),
        association.observation.map_frame_position.y(),
        0.45));
      marker.points.push_back(make_point(cone.position.x(), cone.position.y(), 0.45));
    }

    array.markers.push_back(marker);
    pub_associations_->publish(array);
  }

  void publish_state(const rclcpp::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray array;

    auto arrow = make_basic_marker("kf_pose", 0, visualization_msgs::msg::Marker::ARROW, stamp);
    arrow.pose.position.x = state_(IDX_X);
    arrow.pose.position.y = state_(IDX_Y);
    arrow.pose.position.z = 0.75;
    arrow.pose.orientation = yaw_to_quaternion(state_(IDX_YAW));
    arrow.scale.x = 1.6;
    arrow.scale.y = 0.28;
    arrow.scale.z = 0.35;
    arrow.color.r = 0.0f;
    arrow.color.g = 0.95f;
    arrow.color.b = 0.2f;
    arrow.color.a = 1.0f;
    array.markers.push_back(arrow);

    Eigen::Matrix2d pose_covariance;
    pose_covariance << covariance_(IDX_X, IDX_X), covariance_(IDX_X, IDX_Y),
                       covariance_(IDX_Y, IDX_X), covariance_(IDX_Y, IDX_Y);

    std_msgs::msg::ColorRGBA covariance_color;
    covariance_color.r = 0.0f;
    covariance_color.g = 0.7f;
    covariance_color.b = 0.4f;
    covariance_color.a = 0.28f;

    auto covariance = make_covariance_ellipse_marker(
      "pose_covariance_95",
      1,
      Eigen::Vector2d(state_(IDX_X), state_(IDX_Y)),
      pose_covariance,
      0.04,
      covariance_color,
      stamp);
    array.markers.push_back(covariance);

    pub_state_->publish(array);
  }

  void publish_pose_odom_path_tf(const rclcpp::Time & stamp)
  {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = map_frame_;
    pose.header.stamp = stamp;
    pose.pose.position.x = state_(IDX_X);
    pose.pose.position.y = state_(IDX_Y);
    pose.pose.position.z = 0.0;
    pose.pose.orientation = yaw_to_quaternion(state_(IDX_YAW));
    pub_pose_->publish(pose);

    nav_msgs::msg::Odometry odom;
    odom.header = pose.header;
    odom.child_frame_id = base_frame_;
    odom.pose.pose = pose.pose;
    odom.twist.twist.linear.x = state_(IDX_V);
    odom.twist.twist.angular.z = state_(IDX_YAW_RATE);

    for (std::size_t row = 0; row < 5; ++row) {
      for (std::size_t col = 0; col < 5; ++col) {
        const std::size_t odom_row = row == IDX_YAW ? 5 : row;
        const std::size_t odom_col = col == IDX_YAW ? 5 : col;
        if (odom_row < 6 && odom_col < 6) {
          odom.pose.covariance[odom_row * 6 + odom_col] = covariance_(row, col);
        }
      }
    }

    pub_odom_->publish(odom);

    path_msg_.header.frame_id = map_frame_;
    path_msg_.header.stamp = stamp;
    path_msg_.poses.push_back(pose);
    if (path_msg_.poses.size() > 2000) {
      path_msg_.poses.erase(path_msg_.poses.begin());
    }
    pub_path_->publish(path_msg_);

    geometry_msgs::msg::TransformStamped transform;
    transform.header = pose.header;
    transform.child_frame_id = base_frame_;
    transform.transform.translation.x = pose.pose.position.x;
    transform.transform.translation.y = pose.pose.position.y;
    transform.transform.translation.z = 0.0;
    transform.transform.rotation = pose.pose.orientation;
    tf_broadcaster_->sendTransform(transform);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalizationNode2>());
  rclcpp::shutdown();
  return 0;
}