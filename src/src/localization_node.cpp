#include <cmath>
#include <array>
#include <memory>
#include <string>

#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "visualization_msgs/msg/marker.hpp"

// ── Vehicle parameters ────────────────────────────────────────────────────────
static constexpr double L_R          = 0.765;
static constexpr double L_F          = 0.765;
static constexpr double WHEELBASE    = L_F + L_R;   // 1.530 m
static constexpr double WHEEL_RADIUS = 0.2525;      // m

static constexpr double INIT_COVARIANCE = 1.0;
static constexpr double MIN_DT = 1.0e-4;
static constexpr double MAX_DT = 1.0;

static constexpr double Q_XY        = 0.05;  // m^2 / s
static constexpr double Q_YAW       = 0.03;  // rad^2 / s
static constexpr double Q_V         = 0.30;  // (m/s)^2 / s
static constexpr double Q_YAW_RATE  = 0.20;  // (rad/s)^2 / s

static constexpr double R_WHEEL_V   = 0.15;  // (m/s)^2
static constexpr double R_IMU_YAW   = 0.05;  // (rad/s)^2

static inline double angle_wrap(double a)
{
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

static geometry_msgs::msg::Quaternion yaw_to_quat(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.x = 0.0;  q.y = 0.0;
  q.z = std::sin(yaw * 0.5);
  q.w = std::cos(yaw * 0.5);
  return q;
}

// Build arrow marker. z_offset separates overlapping markers visually.
static visualization_msgs::msg::Marker make_arrow(
    const std::string & frame_id,
    const rclcpp::Time & stamp,
    int id,
    double x, double y, double z_offset, double yaw,
    float r, float g, float b,
    const std::string & ns)
{
  visualization_msgs::msg::Marker m;
  m.header.frame_id  = frame_id;
  m.header.stamp     = stamp;
  m.ns               = ns;
  m.id               = id;
  m.type             = visualization_msgs::msg::Marker::ARROW;
  m.action           = visualization_msgs::msg::Marker::ADD;

  m.pose.position.x  = x;
  m.pose.position.y  = y;
  m.pose.position.z  = z_offset;
  m.pose.orientation = yaw_to_quat(yaw);

  m.scale.x = 2.0;   // length
  m.scale.y = 0.4;   // width
  m.scale.z = 0.5;   // head

  m.color.r = r;  m.color.g = g;  m.color.b = b;  m.color.a = 1.0f;

  // Long lifetime: 0 = forever (until replaced)
  m.lifetime = rclcpp::Duration::from_seconds(0);
  return m;
}

// QoS matching bag: reliability=RELIABLE, durability=VOLATILE
static rclcpp::QoS bag_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(10))
           .reliable()
           .durability_volatile();
}

// ─────────────────────────────────────────────────────────────────────────────
class LocalizationNode : public rclcpp::Node
{
public:
  LocalizationNode()
  : Node("localization_node"),
    state_(State::Zero()),
    covariance_(StateMatrix::Identity() * INIT_COVARIANCE),
    prev_imu_time_(0, 0, RCL_ROS_TIME),
    first_imu_(true)
  {
    // ── Subscriptions ─────────────────────────────────────────────────────

    // /wheel_speed_avg → linear speed (rad/s, converted inside)
    sub_wheel_ = create_subscription<std_msgs::msg::Float32>(
      "/wheel_speed_avg", bag_qos(),
      [this](const std_msgs::msg::Float32::SharedPtr msg) {
        const double wheel_speed_rads = static_cast<double>(msg->data);
        const double measured_v = wheel_speed_rads * WHEEL_RADIUS * M_PI / 30.0;
        wheel_update(measured_v);
        RCLCPP_DEBUG(get_logger(), "wheel_speed_avg: %.4f rad/s -> %.4f m/s",
          wheel_speed_rads, measured_v);
      });

    // /imu → yaw rate + triggers motion update
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu", bag_qos(),
      std::bind(&LocalizationNode::imu_cb, this, std::placeholders::_1));

    // ── Publishers ────────────────────────────────────────────────────────
    pub_predicted_ = create_publisher<visualization_msgs::msg::Marker>(
      "/localization/predicted_pose_marker", 10);

    pub_gt_pose_ = create_publisher<visualization_msgs::msg::Marker>(
      "/localization/gt_pose_marker", 10);

    // ── Timer: publish markers at 10 Hz regardless of IMU rate ────────────
    // This prevents RViz from losing markers between message bursts.
    timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&LocalizationNode::publish_markers, this));

    RCLCPP_INFO(get_logger(), "=========================================");
    RCLCPP_INFO(get_logger(), " LocalizationNode  –  Bicycle Model");
    RCLCPP_INFO(get_logger(), "  l_r=%.3f m  l_f=%.3f m  L=%.3f m", L_R, L_F, WHEELBASE);
    RCLCPP_INFO(get_logger(), "  wheel_radius=%.4f m", WHEEL_RADIUS);
    RCLCPP_INFO(get_logger(), "  frame_id = \"map\"");
    RCLCPP_INFO(get_logger(), "  Filter    = EKF [x, y, yaw, v, yaw_rate]");
    RCLCPP_INFO(get_logger(), "  Predicted → /localization/predicted_pose_marker  [RED]");
    RCLCPP_INFO(get_logger(), "  GT ref   → /localization/gt_pose_marker           [BLUE]");
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

  // ── State ─────────────────────────────────────────────────────────────────
  State state_;
  StateMatrix covariance_;
  rclcpp::Time prev_imu_time_;
  bool first_imu_;

  // ── Handles ───────────────────────────────────────────────────────────────
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr  sub_wheel_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr   sub_imu_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_predicted_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_gt_pose_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ── Timer callback: publish current pose markers ──────────────────────────
  void publish_markers()
  {
    const rclcpp::Time now = this->now();

    // RED arrow → bicycle-model predicted pose  (z = 0.5 so it floats above)
    pub_predicted_->publish(make_arrow(
      "map", now, 0,
      state_(IDX_X), state_(IDX_Y), /*z=*/0.5, state_(IDX_YAW),
      /*r*/1.0f, /*g*/0.0f, /*b*/0.0f,
      "predicted_pose"));

    // BLUE arrow → ground-truth origin reference  (z = 0.0)
    pub_gt_pose_->publish(make_arrow(
      "map", now, 1,
      0.0, 0.0, /*z=*/0.0, 0.0,
      /*r*/0.0f, /*g*/0.4f, /*b*/1.0f,
      "gt_pose"));
  }

  void wheel_update(double measured_v)
  {
    Eigen::Matrix<double, 1, 5> h = Eigen::Matrix<double, 1, 5>::Zero();
    h(0, IDX_V) = 1.0;
    scalar_update(measured_v, R_WHEEL_V, h);
  }

  void imu_update(double measured_yaw_rate)
  {
    Eigen::Matrix<double, 1, 5> h = Eigen::Matrix<double, 1, 5>::Zero();
    h(0, IDX_YAW_RATE) = 1.0;
    scalar_update(measured_yaw_rate, R_IMU_YAW, h);
  }

  void scalar_update(
    double measurement,
    double measurement_variance,
    const Eigen::Matrix<double, 1, 5> & h)
  {
    const double innovation = measurement - (h * state_)(0, 0);
    const double s = (h * covariance_ * h.transpose())(0, 0) + measurement_variance;

    if (s <= 0.0) {
      return;
    }

    const Eigen::Matrix<double, 5, 1> k = covariance_ * h.transpose() / s;
    state_ += k * innovation;

    const StateMatrix identity = StateMatrix::Identity();
    covariance_ = (identity - k * h) * covariance_;
    state_(IDX_YAW) = angle_wrap(state_(IDX_YAW));
  }

  void predict(double dt)
  {
    if (dt < MIN_DT || dt > MAX_DT) {
      return;
    }

    const double yaw = state_(IDX_YAW);
    const double v = state_(IDX_V);
    const double yaw_rate = state_(IDX_YAW_RATE);

    State predicted = state_;
    predicted(IDX_X) += v * std::cos(yaw) * dt;
    predicted(IDX_Y) += v * std::sin(yaw) * dt;
    predicted(IDX_YAW) = angle_wrap(yaw + yaw_rate * dt);

    StateMatrix f = StateMatrix::Identity();
    f(IDX_X, IDX_YAW) = -v * std::sin(yaw) * dt;
    f(IDX_X, IDX_V) = std::cos(yaw) * dt;
    f(IDX_Y, IDX_YAW) = v * std::cos(yaw) * dt;
    f(IDX_Y, IDX_V) = std::sin(yaw) * dt;
    f(IDX_YAW, IDX_YAW_RATE) = dt;

    StateMatrix q = StateMatrix::Zero();
    q(IDX_X, IDX_X) = Q_XY * dt;
    q(IDX_Y, IDX_Y) = Q_XY * dt;
    q(IDX_YAW, IDX_YAW) = Q_YAW * dt;
    q(IDX_V, IDX_V) = Q_V * dt;
    q(IDX_YAW_RATE, IDX_YAW_RATE) = Q_YAW_RATE * dt;

    state_ = predicted;
    covariance_ = f * covariance_ * f.transpose() + q;
    state_(IDX_YAW) = angle_wrap(state_(IDX_YAW));
  }

  // ── IMU callback: EKF prediction + yaw-rate correction ────────────────────
  void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    const rclcpp::Time stamp(msg->header.stamp);
    const double measured_yaw_rate = msg->angular_velocity.z;
  
    if (first_imu_) {
      prev_imu_time_ = stamp;
      first_imu_     = false;
      imu_update(measured_yaw_rate);
      RCLCPP_INFO(get_logger(), "First IMU message received — starting integration.");
      return;
    }

    const double dt = (stamp - prev_imu_time_).seconds();
    prev_imu_time_  = stamp;

    if (dt < MIN_DT || dt > MAX_DT) {
      imu_update(measured_yaw_rate);
      return;
    }

    predict(dt);
    imu_update(measured_yaw_rate);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "ekf: x=%7.3f  y=%7.3f  yaw=%6.2f deg  |  v=%.3f m/s  yaw_rate=%.4f rad/s  dt=%.4fs",
      state_(IDX_X), state_(IDX_Y), state_(IDX_YAW) * 180.0 / M_PI,
      state_(IDX_V), state_(IDX_YAW_RATE), dt);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalizationNode>());
  rclcpp::shutdown();
  return 0;
}
