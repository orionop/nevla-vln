FROM zhangjicmu/ubuntu24_ros:ai_module

# Copy our AI-module sources into the existing workspace.
# dummy_vlm is kept for reference/fallback; vln_orchestrator is our entry.
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
COPY src/vln_orchestrator /home/docker/ai_module/src/vln_orchestrator

# Build both packages against the already-built workspace dependencies.
# vln_orchestrator is pure-Python (rclpy + std/geometry/visualization msgs);
# its VLM dependencies (openai, pydantic, cv2) are imported lazily and only
# needed once the live VLM path is wired, so they are not required to build.
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm vln_orchestrator"

# Ensure the updated install is sourced in every new shell
RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc
