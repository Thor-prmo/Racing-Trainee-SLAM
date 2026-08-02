#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "slam_localization/data_association_checkpoint3.hpp"

namespace da = slam_localization::checkpoint3;

namespace
{

void print_associations(const std::string & label, const std::vector<int> & associations)
{
  std::cout << label << ": [";
  for (std::size_t i = 0; i < associations.size(); ++i) {
    std::cout << associations[i];
    if (i + 1 < associations.size()) {
      std::cout << ", ";
    }
  }
  std::cout << "]\n";
}

}  // namespace

int main()
{
  const da::Pose2D pose{0.0, 0.0, 0.0};

  const std::vector<Eigen::Vector2d> landmarks{
    Eigen::Vector2d(5.0, 0.0),
    Eigen::Vector2d(0.0, 5.0),
    Eigen::Vector2d(8.0, -2.0)};

  const std::vector<da::RangeBearing> readings{
    da::RangeBearing{5.05, 0.02},
    da::RangeBearing{4.95, 1.56},
    da::RangeBearing{3.0, -0.75}};

  da::AssociationOptions options;
  options.nearest_neighbor_gate = 1.5;
  options.individual_compatibility_gate = 5.991464547107979;
  options.measurement_covariance << 0.25, 0.0,
    0.0, 0.01;

  print_associations(
    "nearest_neighbor",
    da::associate_nearest_neighbor(pose, readings, landmarks, options));
  print_associations(
    "ml_ic",
    da::associate_ml_ic(pose, readings, landmarks, options));
  print_associations(
    "kd_tree_nearest_neighbor",
    da::associate_kd_tree_nearest_neighbor(pose, readings, landmarks, options));

  return 0;
}
