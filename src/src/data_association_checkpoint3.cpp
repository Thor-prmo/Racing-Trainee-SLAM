#include "slam_localization/data_association_checkpoint3.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace slam_localization
{
namespace checkpoint3
{
namespace
{

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kDefaultIcGate95 = 5.991464547107979;

double squared_distance(const Eigen::Vector2d & a, const Eigen::Vector2d & b)
{
  return (a - b).squaredNorm();
}

Eigen::Matrix<double, 2, 3> pose_jacobian(
  const Pose2D & pose,
  const Eigen::Vector2d & landmark,
  double min_range)
{
  const double dx = landmark.x() - pose.x;
  const double dy = landmark.y() - pose.y;
  const double q = std::max(dx * dx + dy * dy, min_range * min_range);
  const double range = std::sqrt(q);

  Eigen::Matrix<double, 2, 3> h;
  h << -dx / range, -dy / range, 0.0,
        dy / q,     -dx / q,    -1.0;
  return h;
}

double association_score(
  const Eigen::Vector2d & innovation,
  const Eigen::Matrix2d & covariance)
{
  const Eigen::Matrix2d symmetric_covariance =
    0.5 * (covariance + covariance.transpose());
  const Eigen::LDLT<Eigen::Matrix2d> ldlt(symmetric_covariance);

  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return std::numeric_limits<double>::infinity();
  }

  const double mahalanobis = innovation.dot(ldlt.solve(innovation));
  const double determinant = std::max(1.0e-12, symmetric_covariance.determinant());

  return mahalanobis + std::log(determinant);
}

double mahalanobis_distance_squared(
  const Eigen::Vector2d & innovation,
  const Eigen::Matrix2d & covariance)
{
  const Eigen::Matrix2d symmetric_covariance =
    0.5 * (covariance + covariance.transpose());
  const Eigen::LDLT<Eigen::Matrix2d> ldlt(symmetric_covariance);

  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
    return std::numeric_limits<double>::infinity();
  }

  return innovation.dot(ldlt.solve(innovation));
}

}  // namespace

AssociationOptions::AssociationOptions()
: nearest_neighbor_gate(std::numeric_limits<double>::infinity()),
  individual_compatibility_gate(kDefaultIcGate95),
  min_predicted_range(1.0e-6),
  use_pose_covariance(false),
  measurement_covariance(Eigen::Matrix2d::Identity() * 0.04),
  pose_covariance(Eigen::Matrix3d::Zero())
{
}

double wrap_angle(double angle)
{
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

Eigen::Vector2d reading_to_world(
  const Pose2D & pose,
  const RangeBearing & reading)
{
  const double global_bearing = pose.yaw + reading.bearing;
  return Eigen::Vector2d(
    pose.x + reading.range * std::cos(global_bearing),
    pose.y + reading.range * std::sin(global_bearing));
}

RangeBearing predict_reading(
  const Pose2D & pose,
  const Eigen::Vector2d & landmark)
{
  const double dx = landmark.x() - pose.x;
  const double dy = landmark.y() - pose.y;

  RangeBearing prediction;
  prediction.range = std::sqrt(dx * dx + dy * dy);
  prediction.bearing = wrap_angle(std::atan2(dy, dx) - pose.yaw);
  return prediction;
}

std::vector<int> associate_nearest_neighbor(
  const Pose2D & pose,
  const std::vector<RangeBearing> & readings,
  const std::vector<Eigen::Vector2d> & landmarks,
  const AssociationOptions & options)
{
  std::vector<int> associations(readings.size(), -1);

  if (landmarks.empty()) {
    return associations;
  }

  const double gate_squared =
    std::isfinite(options.nearest_neighbor_gate) ?
    options.nearest_neighbor_gate * options.nearest_neighbor_gate :
    std::numeric_limits<double>::infinity();

  for (std::size_t reading_index = 0; reading_index < readings.size(); ++reading_index) {
    if (readings[reading_index].range < 0.0) {
      continue;
    }

    const Eigen::Vector2d query = reading_to_world(pose, readings[reading_index]);
    double best_distance_squared = gate_squared;
    int best_index = -1;

    for (std::size_t landmark_index = 0; landmark_index < landmarks.size(); ++landmark_index) {
      const double distance_squared = squared_distance(query, landmarks[landmark_index]);
      if (distance_squared <= best_distance_squared) {
        best_distance_squared = distance_squared;
        best_index = static_cast<int>(landmark_index);
      }
    }

    associations[reading_index] = best_index;
  }

  return associations;
}

std::vector<int> associate_ml_ic(
  const Pose2D & pose,
  const std::vector<RangeBearing> & readings,
  const std::vector<Eigen::Vector2d> & landmarks,
  const AssociationOptions & options)
{
  std::vector<int> associations(readings.size(), -1);

  if (landmarks.empty()) {
    return associations;
  }

  for (std::size_t reading_index = 0; reading_index < readings.size(); ++reading_index) {
    const auto & reading = readings[reading_index];
    if (reading.range < 0.0) {
      continue;
    }

    double best_score = std::numeric_limits<double>::infinity();
    int best_index = -1;

    for (std::size_t landmark_index = 0; landmark_index < landmarks.size(); ++landmark_index) {
      const RangeBearing prediction = predict_reading(pose, landmarks[landmark_index]);
      if (prediction.range < options.min_predicted_range) {
        continue;
      }

      Eigen::Vector2d innovation;
      innovation << reading.range - prediction.range,
        wrap_angle(reading.bearing - prediction.bearing);

      Eigen::Matrix2d innovation_covariance = options.measurement_covariance;
      if (options.use_pose_covariance) {
        const auto h = pose_jacobian(
          pose, landmarks[landmark_index], options.min_predicted_range);
        innovation_covariance += h * options.pose_covariance * h.transpose();
      }

      const double distance_squared =
        mahalanobis_distance_squared(innovation, innovation_covariance);
      if (distance_squared > options.individual_compatibility_gate) {
        continue;
      }

      const double score = association_score(innovation, innovation_covariance);
      if (score < best_score) {
        best_score = score;
        best_index = static_cast<int>(landmark_index);
      }
    }

    associations[reading_index] = best_index;
  }

  return associations;
}

KdTree2D::KdTree2D(const std::vector<Eigen::Vector2d> & landmarks)
: landmarks_(landmarks)
{
  std::vector<int> indices(landmarks_.size());
  std::iota(indices.begin(), indices.end(), 0);

  nodes_.reserve(landmarks_.size());
  build_recursive(indices, 0, indices.size(), 0);
}

bool KdTree2D::nearest(
  const Eigen::Vector2d & query,
  double max_distance,
  int & index,
  double & distance) const
{
  index = -1;
  distance = std::numeric_limits<double>::infinity();

  if (nodes_.empty() || max_distance < 0.0) {
    return false;
  }

  double best_distance_squared =
    std::isfinite(max_distance) ?
    max_distance * max_distance :
    std::numeric_limits<double>::infinity();

  nearest_recursive(0, query, best_distance_squared, index);

  if (index < 0) {
    return false;
  }

  distance = std::sqrt(best_distance_squared);
  return true;
}

int KdTree2D::build_recursive(
  std::vector<int> & indices,
  std::size_t begin,
  std::size_t end,
  int depth)
{
  if (begin >= end) {
    return -1;
  }

  const int axis = depth % 2;
  const std::size_t mid = begin + (end - begin) / 2;

  std::nth_element(
    indices.begin() + static_cast<std::ptrdiff_t>(begin),
    indices.begin() + static_cast<std::ptrdiff_t>(mid),
    indices.begin() + static_cast<std::ptrdiff_t>(end),
    [this, axis](int lhs, int rhs) {
      return landmarks_[lhs](axis) < landmarks_[rhs](axis);
    });

  const int node_id = static_cast<int>(nodes_.size());
  nodes_.push_back(Node{landmarks_[indices[mid]], indices[mid], axis, -1, -1});

  nodes_[node_id].left = build_recursive(indices, begin, mid, depth + 1);
  nodes_[node_id].right = build_recursive(indices, mid + 1, end, depth + 1);

  return node_id;
}

void KdTree2D::nearest_recursive(
  int node_id,
  const Eigen::Vector2d & query,
  double & best_distance_squared,
  int & best_index) const
{
  if (node_id < 0) {
    return;
  }

  const Node & node = nodes_[static_cast<std::size_t>(node_id)];
  const double distance_squared = squared_distance(query, node.point);
  if (distance_squared <= best_distance_squared) {
    best_distance_squared = distance_squared;
    best_index = node.landmark_index;
  }

  const double axis_delta = query(node.axis) - node.point(node.axis);
  const int near_node = axis_delta < 0.0 ? node.left : node.right;
  const int far_node = axis_delta < 0.0 ? node.right : node.left;

  nearest_recursive(near_node, query, best_distance_squared, best_index);

  if (axis_delta * axis_delta <= best_distance_squared) {
    nearest_recursive(far_node, query, best_distance_squared, best_index);
  }
}

std::vector<int> associate_kd_tree_nearest_neighbor(
  const Pose2D & pose,
  const std::vector<RangeBearing> & readings,
  const std::vector<Eigen::Vector2d> & landmarks,
  const AssociationOptions & options)
{
  std::vector<int> associations(readings.size(), -1);

  if (landmarks.empty()) {
    return associations;
  }

  const KdTree2D tree(landmarks);

  for (std::size_t reading_index = 0; reading_index < readings.size(); ++reading_index) {
    if (readings[reading_index].range < 0.0) {
      continue;
    }

    const Eigen::Vector2d query = reading_to_world(pose, readings[reading_index]);
    int landmark_index = -1;
    double distance = std::numeric_limits<double>::infinity();

    if (tree.nearest(query, options.nearest_neighbor_gate, landmark_index, distance)) {
      associations[reading_index] = landmark_index;
    }
  }

  return associations;
}

}  // namespace checkpoint3
}  // namespace slam_localization
