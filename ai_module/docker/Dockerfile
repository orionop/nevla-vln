FROM zhangjicmu/ubuntu24_ros:ai_module

# ---------------------------------------------------------------------------
# Python dependencies (heavy). Kept BEFORE the source COPY so editing our code
# does NOT re-trigger the multi-GB download (Docker layer cache). Installed
# system-wide as root with --break-system-packages (PEP 668, Ubuntu 24).
#   - openai/pydantic : VLM client (Gemini path)
#   - torch/torchvision/ultralytics/supervision : perception A1 (detection_node).
#     torch pulls the CUDA build from PyPI; verify torch.cuda.is_available() on
#     the box. (SAM2 + open3d + scikit-learn + line_profiler come in A2.)
# ---------------------------------------------------------------------------
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
# The base image's numpy is debian-managed (no pip RECORD), so pip cannot
# uninstall it when a dep wants a different version -> "Cannot uninstall numpy".
# Install a pip-managed numpy FIRST (--ignore-installed shadows the debian one),
# pinned to 1.26.x so the debian-built cv2/cv_bridge keep their numpy-1.x ABI.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
      --ignore-installed "numpy==1.26.4"
# Then the rest, holding numpy at 1.26.4 so torch/ultralytics don't try to move it.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
      "numpy==1.26.4" \
      "openai>=1.40" "pydantic>=2" \
      torch torchvision ultralytics supervision
USER docker

# ---------------------------------------------------------------------------
# Sources (change often -> after the dep layer).
#   dummy_vlm        : reference/fallback
#   vln_orchestrator : our entry node
#   tare_planner     : msgs-only (ObjectNodeList, ...) shared by perception + us
#   semantic_mapping : SysNav perception (detection_node + semantic_mapping_node),
#                      vendored + adapted (.pt detector instead of TensorRT engine)
# ---------------------------------------------------------------------------
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
COPY src/vln_orchestrator /home/docker/ai_module/src/vln_orchestrator
COPY src/tare_planner /home/docker/ai_module/src/tare_planner
COPY vendor/sysnav/semantic_mapping /home/docker/ai_module/src/semantic_mapping

# Use our mined challenge vocabulary as the detector's default object_file
# (detection_node reads CONFIG_DIR/config/objects.yaml).
COPY src/vln_orchestrator/config/challenge_classes.yaml \
     /home/docker/ai_module/src/semantic_mapping/semantic_mapping/config/objects.yaml

# ---------------------------------------------------------------------------
# Build. --symlink-install so semantic_mapping's CONFIG_DIR (= __file__.parent)
# resolves config/objects.yaml + config/botsort.yaml from the source tree at
# runtime (plain install would not place config/ next to the module).
# ---------------------------------------------------------------------------
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select tare_planner dummy_vlm vln_orchestrator semantic_mapping"

# Ensure the updated install is sourced in every new shell
RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc
