#ifndef SLAM_LOCALIZATION_DATA_ASSOCIATION_CHECKPOINT3_HPP_
#define SLAM_LOCALIZATION_DATA_ASSOCIATION_CHECKPOINT3_HPP_

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

namespace slam_localization
{
namespace checkpoint3
{

struct Pose2D
{
  double x;
  double y;
  double yaw;
};

struct RangeBearing
{
  double range;
  double bearing;
};

struct AssociationOptions
{
  AssociationOptions();

  double nearest_neighbor_gate;
  double individual_compatibility_gate;
  double min_predicted_range;
  bool use_pose_covariance;
  Eigen::Matrix2d measurement_covariance;
  Eigen::Matrix3d pose_covariance;
};

double wrap_angle(double angle);

Eigen::Vector2d reading_to_world(
  const Pose2D & pose,
  const RangeBearing & reading);

RangeBearing predict_reading(
  const Pose2D & pose,
  const Eigen::Vector2d & landmark);

std::vector<int> associate_nearest_neighbor(
  const Pose2D & pose,
  const std::vector<RangeBearing> & readings,
  const std::vector<Eigen::Vector2d> & landmarks,
  const AssociationOptions & options = AssociationOptions());

std::vector<int> associate_ml_ic(
  const Pose2D & pose,
  const std::vector<RangeBearing> & readings,
  const std::vector<Eigen::Vector2d> & landmarks,
  const AssociationOptions & options = AssociationOptions());

class KdTree2D
{
public:
  explicit KdTree2D(const std::vector<Eigen::Vector2d> & landmarks);

  bool nearest(
    const Eigen::Vector2d & query,
    double max_distance,
    int & index,
    double & distance) const;

private:
  struct Node
  {
    Eigen::Vector2d point;
    int landmark_index;
    int axis;
    int left;
    int right;
  };

  int build_recursive(
    std::vector<int> & indices,
    std::size_t begin,
    std::size_t end,
    int depth);

  void nearest_recursive(
    int node_id,
    const Eigen::Vector2d & query,
    double & best_distance_squared,
    int & best_index) const;

  std::vector<Eigen::Vector2d> landmarks_;
  std::vector<Node> nodes_;
};

std::vector<int> associate_kd_tree_nearest_neighbor(
  const Pose2D & pose,
  const std::vector<RangeBearing> & readings,
  const std::vector<Eigen::Vector2d> & landmarks,
  const AssociationOptions & options = AssociationOptions());

}  // namespace checkpoint3
}  // namespace slam_localization

#endif  // SLAM_LOCALIZATION_DATA_ASSOCIATION_CHECKPOINT3_HPP_
