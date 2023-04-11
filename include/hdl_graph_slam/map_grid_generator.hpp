// SPDX-License-Identifier: BSD-2-Clause

#ifndef MAP_GRID_GENERATOR_HPP
#define MAP_GRID_GENERATRO_HPP

#include <vector>
#include <numeric>

#include <hdl_graph_slam/keyframe.hpp>
#include <nav_msgs/OccupancyGrid.h>
#include <map_msgs/OccupancyGridUpdate.h>

#include <tf/tf.h>
#include <tf/transform_listener.h>

namespace hdl_graph_slam {

class MapGridGenerator {
public:
  using PointT = pcl::PointXYZI;

  MapGridGenerator();
  ~MapGridGenerator();

  nav_msgs::OccupancyGrid::Ptr generate(const std::vector<KeyFrameSnapshot::Ptr>& keyframes, double resolution = 0.05) const;

private:
  inline float ProbabilityToLogOdds(float prob) const {
    return std::log(prob / (1.f - prob));
  }

  inline float LogOddsToProbability(float odd) const {
    return 1.f / (1.f + std::exp(-odd));
  }

  float inverse_sensor_model(const Eigen::Vector2i& c, const Eigen::Vector2i& range, const double& resolution) const {

    if(std::sqrt((double)c.squaredNorm()) < (std::sqrt((double)range.squaredNorm()) - 0.5 * resolution)) {
      return _log_odd_free;
    }
    if(std::sqrt((double)c.squaredNorm()) > (std::sqrt((double)range.squaredNorm()) + 0.5 * resolution)) {
      return _log_odd_occ;
    }

    return _log_odd_prior;
  }

  std::vector<Eigen::Vector2i> bresenham_line(Eigen::Vector2i p1, Eigen::Vector2i p2) const;

  tf::TransformListener tf_listener;

  const float _log_odd_free;
  const float _log_odd_occ;
  const float _log_odd_prior;
};

}  // namespace hdl_graph_slam

#endif