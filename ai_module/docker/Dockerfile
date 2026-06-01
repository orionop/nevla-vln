FROM zhangjicmu/ubuntu24_ros:ai_module

# Copy our AI-module sources into the existing workspace.
# dummy_vlm is kept for reference/fallback; vln_orchestrator is our entry.
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
COPY src/vln_orchestrator /home/docker/ai_module/src/vln_orchestrator

# VLM client dependencies for the live Gemini/Qwen path (lazy-imported by
# reasoning/vlm_client.py; only needed when an API key is present). The image
# runs as user 'docker', so install to the user site; --break-system-packages
# satisfies PEP 668 on Ubuntu 24. cv2 (image verification) is already in the
# perception image and not needed for text decomposition.
RUN python3 -m pip install --no-cache-dir --user --break-system-packages \
      "openai>=1.40" "pydantic>=2"

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
