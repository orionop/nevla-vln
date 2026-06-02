cmake_minimum_required(VERSION 3.8)
project(tare_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# SysNav semantic-map / perception interfaces (vendored, BSD-3). Msgs only.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectionResult.msg"
  "msg/NavigationQuery.msg"
  "msg/ObjectNode.msg"
  "msg/ObjectNodeList.msg"
  "msg/ObjectType.msg"
  "msg/RoomEarlyStop1.msg"
  "msg/RoomNode.msg"
  "msg/RoomNodeList.msg"
  "msg/RoomType.msg"
  "msg/TargetObject.msg"
  "msg/TargetObjectInstruction.msg"
  "msg/TargetObjectWithSpatial.msg"
  "msg/ViewpointRep.msg"
  "msg/VlmAnswer.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
