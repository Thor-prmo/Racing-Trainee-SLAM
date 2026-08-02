cmake_minimum_required(VERSION 3.8)
project(slam_localization)

# ── Compiler flags ─────────────────────────────────────────────────────────
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Find dependencies ──────────────────────────────────────────────────────
find_package(ament_cmake        REQUIRED)
find_package(ament_index_cpp    REQUIRED)
find_package(Eigen3             REQUIRED)
find_package(rclcpp             REQUIRED)
find_package(std_msgs           REQUIRED)
find_package(sensor_msgs        REQUIRED)
find_package(nav_msgs           REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs      REQUIRED)
find_package(tf2                REQUIRED)
find_package(tf2_geometry_msgs  REQUIRED)
find_package(tf2_ros            REQUIRED)

# ── Checkpoint 3: standalone data-association functions ─────────────────────
add_library(data_association_checkpoint3
  src/data_association_checkpoint3.cpp
)

target_include_directories(data_association_checkpoint3 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

target_compile_features(data_association_checkpoint3 PUBLIC cxx_std_17)

add_executable(data_association_checkpoint3_demo
  src/data_association_checkpoint3_demo.cpp
)

target_link_libraries(data_association_checkpoint3_demo
  data_association_checkpoint3
)

# ── Build the localization node ────────────────────────────────────────────
add_executable(localization_node
  src/localization_node.cpp
)

target_include_directories(localization_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

target_compile_features(localization_node PUBLIC cxx_std_17)

ament_target_dependencies(localization_node
  rclcpp
  std_msgs
  sensor_msgs
  visualization_msgs
  geometry_msgs
  tf2
  tf2_geometry_msgs
)

# ── Build the KF localization node with cone measurement updates ─────────────
add_executable(localization_node_2
  src/localization_node_2.cpp
)

target_include_directories(localization_node_2 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

target_compile_features(localization_node_2 PUBLIC cxx_std_17)

ament_target_dependencies(localization_node_2
  ament_index_cpp
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  geometry_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

# ── Install ────────────────────────────────────────────────────────────────
install(TARGETS
  localization_node
  localization_node_2
  data_association_checkpoint3_demo
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS data_association_checkpoint3
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES small_track.csv
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(data_association_checkpoint3)

# ── Linting (optional, comment out if ament_lint not installed) ────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
