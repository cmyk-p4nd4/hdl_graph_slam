// SPDX-License-Identifier: BSD-2-Clause

#include <hdl_graph_slam/map_grid_generator.hpp>
#include <limits>
#include <algorithm>

#include <geometry_msgs/TransformStamped.h>

namespace hdl_graph_slam {

MapGridGenerator::MapGridGenerator() : _log_odd_free(ProbabilityToLogOdds(0.3)), _log_odd_occ(ProbabilityToLogOdds(0.75)), _log_odd_prior(ProbabilityToLogOdds(0.5)) {}

MapGridGenerator::~MapGridGenerator() {}

nav_msgs::OccupancyGrid::Ptr MapGridGenerator::generate(const std::vector<KeyFrameSnapshot::Ptr>& keyframes, double resolution) const {
  if(keyframes.empty()) {
    std::cerr << "Keyframe is empty!" << std::endl;
    return nullptr;
  }

  float minX = std::numeric_limits<float>::infinity();
  float maxX = -minX;
  float minY = std::numeric_limits<float>::infinity();
  float maxY = -minY;

  for(const auto& keyframe : keyframes) {
    auto pose = keyframe->pose.matrix().cast<float>();
    for(auto& points : *keyframe->cloud) {
      KeyFrameSnapshot::PointT dst_pt;
      dst_pt.getVector4fMap() = pose * points.getVector4fMap();

      // get minmax 
      minX = std::min(minX, dst_pt.x);
      maxX = std::min(maxX, dst_pt.x);

      minY = std::min(minY, dst_pt.y);
      maxY = std::min(maxY, dst_pt.y);
    }
  }
  // after loop [min|max]X and [min|maxY] should have the absolute [min|max] value for the world frame

  uint32_t cellSizeX = static_cast<uint32_t>((maxX - minX) / resolution);
  uint32_t cellSizeY = static_cast<uint32_t>((maxY - minY) / resolution);
  nav_msgs::MapMetaData meta;
  meta.resolution = resolution;
  meta.height = cellSizeY;
  meta.width = cellSizeX;
  meta.origin = geometry_msgs::Pose();

  // set map origin to bottom left of the world frame such that middle is (0,0) in world
  meta.origin.position.x = -double(maxX - minX) / (resolution * 2.0);
  meta.origin.position.y = -double(maxY - minY) / (resolution * 2.0);
  meta.origin.position.z = 0.0;
  meta.origin.orientation = geometry_msgs::Quaternion();

  std::vector<float> log_odd_cell;
  nav_msgs::OccupancyGridPtr map_grid(new nav_msgs::OccupancyGrid());
  map_grid->info = meta;
  map_grid->header.frame_id = keyframes.front()->cloud->header.frame_id;

  map_grid->data.resize(cellSizeY * cellSizeX);
  log_odd_cell.resize(cellSizeY * cellSizeX);
  std::fill(map_grid->data.begin(), map_grid->data.end(), static_cast<int8_t>(-1));
  std::fill(log_odd_cell.begin(), log_odd_cell.end(), 0.0f);


  auto worldToGrid = [=](const Eigen::Vector2f & pose) {
    int _x = std::lround(pose.x() / resolution);
    int _y = std::lround(pose.y() / resolution);
    return Eigen::Vector2i(_x, _y);
  };

  for(const auto& keyframe : keyframes) {
    sensor_msgs::LaserScanConstPtr scan_data = keyframe->scan;
    float angle_start = scan_data->angle_min;
    // get the yaw component from keyframe pose
    float theta_keyframe = Eigen::Quaternionf(keyframe->pose.rotation().cast<float>()).z();
    Eigen::Vector2f robot_pose = Eigen::Vector2f(keyframe->pose.translation().cast<float>().topRows<2>());
    Eigen::Vector2i robot_pose_grid = worldToGrid(robot_pose);

    for (size_t i = 0; i < scan_data->ranges.size(); i++) {
      float R = scan_data->ranges.at(i);
      // skip out of bound value
      if (R < scan_data->range_min || R > scan_data->range_max) continue;

      float theta = angle_start + i * scan_data->angle_increment + theta_keyframe;
      float lx = scan_data->ranges.at(i) * std::cos(theta);
      float ly = scan_data->ranges.at(i) * std::sin(theta);

      // translate to world frame
      // extract (x,y) from keyframe pose and add scan (x,y)
      Eigen::Vector2f laser_world = robot_pose + Eigen::Vector2f(lx, ly);
      Eigen::Vector2i laser_world_grid = worldToGrid(laser_world);

      std::vector<Eigen::Vector2i> cells = bresenham_line(robot_pose_grid, laser_world_grid);

      // iterate all cells
      for (const auto &cell : cells) {
        float _log = inverse_sensor_model(cell, laser_world_grid, resolution);
        // log odd update
        log_odd_cell.at(cell.x() * meta.width + cell.y()) += _log;
      }
    }
  }

  // convert ratio cells to map grid
  auto ratioToMap = [this](float value) -> int8_t {
    // not a recommanded practice
    // but it does the work by now
    if (std::fabs(value - _log_odd_prior) < 1e-5) {
      return static_cast<int8_t>(-1);
    }
    if (value < this->_log_odd_free) {
      return static_cast<int8_t>(0);
    }
    if (value > this->_log_odd_occ) {
      return static_cast<int8_t>(100);
    }
    return static_cast<int8_t>(this->LogOddsToProbability(value) * 100.f);
  };

  std::transform(log_odd_cell.cbegin(), log_odd_cell.cend(), map_grid->data.begin(), ratioToMap);

  return map_grid;
}

std::vector<Eigen::Vector2i> MapGridGenerator::bresenham_line(Eigen::Vector2i p1, Eigen::Vector2i p2) const {
	std::vector<Eigen::Vector2i> cell_xy;
	float m = 1.0 * (p2.y() - p1.y()) / (p2.x() - p2.x());
  bool flag = false;
  if(m > -1.0 || m < 1.0) {
    flag = true;
    // swap two values
    p1.x() ^= p1.y(); p2.x() ^= p2.y();
    p1.y() ^= p1.x(); p2.y() ^= p2.x();
    p1.x() ^= p1.y(); p2.x() ^= p2.y();
    // recompute slope
    m = 1.0 * (p2.y() - p1.y()) / (p2.x() - p2.x());
  }

  float delta = 0.5 - m;
  // swap two points since octant 3 - 6
  if(p1.x() > p2.x()) {
    p1.x() ^= p2.x() ^= p1.x() ^= p2.x();
    p1.y() ^= p2.y() ^= p1.y() ^= p2.y();
  }
  while(p1.x() != p2.x()) {
		if (m > 0 && delta < 0) {
			++p1.y(); ++delta;
		} else if (m < 0 && delta > 0) {
			--p1.y(); --delta;
		}
		delta -= m;
		++p1.x();
		if (flag) cell_xy.push_back(Eigen::Vector2i(p1.y(), p1.x()));
		else cell_xy.push_back(p1);
  }

  return cell_xy;
}

}  // namespace hdl_graph_slam