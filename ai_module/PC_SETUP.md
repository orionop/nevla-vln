# PC setup — running nevla-vln on a Linux + NVIDIA machine

Bring-up for the full stack on a Linux + NVIDIA box via Docker. The host OS distro
is irrelevant (everything runs in the Jazzy container), but you DO need: a working
**NVIDIA driver on the host**, Docker + the Compose v2 plugin, the NVIDIA Container
Toolkit, and a GPU. Validated on Ubuntu + NVIDIA (e.g. an RTX 4050 6 GB laptop).

> Secrets: never commit `.env`. The Gemini key below stays local (gitignored).

## Hardware notes (read first)
- **VRAM is the binding constraint.** Bring-up (sim + orchestrator, fallback mode)
  fits easily. **Perception (SAM2 + YOLO-World + sim together) is heavy** — on
  **6 GB** use small model variants (SAM2-tiny/small, a light detector) and avoid
  running everything at full res at once; ≥12 GB is comfortable. Offload heavy
  model R&D/benchmarking to Colab (big VRAM) and keep the local box for the
  Docker/ROS/sim integration.
- **Borrowed / laptop machine?** Sustained GPU load won't damage it (thermal
  limits protect the hardware), but it runs hot and loud. Get the owner's OK,
  keep it on **AC power** with good airflow, **monitor temps** (`watch -n2
  nvidia-smi`, throttles ~87 °C), and run sessions rather than 24/7 marathons.

## 0. Prerequisites — do these in order

### 0a. Host NVIDIA driver (must work before anything else)
```bash
nvidia-smi               # must print the GPU + driver version
```
If it fails with "couldn't communicate with the NVIDIA driver", no driver is
installed/loaded. Install one (needs a reboot — on a shared machine, get the
owner's OK first):
```bash
lspci | grep -i -E "vga|3d|nvidia"          # confirm an NVIDIA GPU is present
sudo apt update && sudo apt install -y "linux-headers-$(uname -r)"
ubuntu-drivers devices                       # note the recommended driver
sudo ubuntu-drivers autoinstall              # or: sudo apt install -y nvidia-driver-<ver>
sudo reboot
# after reboot: nvidia-smi should print the GPU. (Very new kernels can need a
# newer driver; if autoinstall's pick fails to build, install a recent -driver-<ver>.)
```

### 0b. Docker + Compose v2 plugin
```bash
docker --version
docker compose version || sudo apt install -y docker-compose-plugin   # 'docker compose' (v2)
```
(If only the old hyphenated `docker-compose` exists, use that and swap
`docker compose` → `docker-compose` in the commands below.)

### 0c. GPU-in-Docker (NVIDIA Container Toolkit)
```bash
docker run --rm --gpus all ubuntu nvidia-smi >/dev/null 2>&1 \
  && echo "GPU-in-Docker OK" || echo "need nvidia-container-toolkit"
```
One-time, only if it said "need…":
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

## 1. Clone (private repo → authenticate once)

```bash
cd ~                     # clone into the home dir -> /home/swarnav/nevla-vln

# browser auth (install gh if missing: sudo apt install -y gh)
gh auth login            # GitHub.com -> HTTPS -> login via browser
gh auth setup-git

git clone --recurse-submodules https://github.com/orionop/nevla-vln.git
cd ~/nevla-vln
```

Alternative without `gh` (use a Personal Access Token):
`git clone --recurse-submodules https://<PAT>@github.com/orionop/nevla-vln.git`

The `autonomy_stack_mecanum_wheel_platform` submodule is public and clones over
HTTPS automatically (`.gitmodules` is HTTPS).

## 2. Create `.env` (local secrets, gitignored)

```bash
cd ~/nevla-vln
cat > .env <<'EOF'
GEMINI_API_KEY=<YOUR_GEMINI_KEY>
DASHSCOPE_API_KEY=
# VLA3D_DIR=/abs/path/to/vla3d/Unity   # only needed for eval scoring, not bring-up
EOF
```

The orchestrator auto-loads `.env` at runtime (`reasoning/vlm_client.py`) — no
`source` needed.

## 3. Build + start the containers (GPU compose)

```bash
xhost +
cd ~/nevla-vln/docker
docker compose -f compose_gpu.yml up --build -d
docker compose -f compose_gpu.yml ps           # iros2026_system + iros2026_ai_module up
```

The `ai_module` image build copies + builds `dummy_vlm` and `vln_orchestrator`
and installs the VLM deps (`openai`, `pydantic`). It is baked into the image
(no volume mount), so **after any code change**: `git pull` then
`docker compose -f compose_gpu.yml up --build -d ai_module`.

## 4. Verify (3 terminals)

```bash
# T1 — simulator + autonomy stack
docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```
```bash
# T2 — our orchestrator (Gemini key auto-loaded from .env)
docker exec -it iros2026_ai_module bash
ros2 launch vln_orchestrator vln_orchestrator.launch.py
```
```bash
# T3 — send one question of each type
docker exec -it iros2026_ai_module bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'How many books are on the sofa'}"
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'Find the pillow closest to the book on the stool'}"
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'Go to the stool and stop at the table'}"
```

**Expected (this stage):** T2 classifies each question, decomposes it (via Gemini
if `GEMINI_API_KEY` is exported in that shell, else a rule-based heuristic), and
publishes on the right topic (`/numerical_response`, `/selected_object_marker`,
`/way_point_with_heading`). On the current image the startup log reads
`perception: subscribed to /object_nodes_list.` (the message package is built),
but the handlers still emit **fallback** answers because nothing publishes that
topic yet — that goes live in Section 5.2.

## Output contract (must not change)
- `/numerical_response` — `std_msgs/Int32`
- `/selected_object_marker` — `visualization_msgs/Marker` (CUBE)
- `/way_point_with_heading` — `geometry_msgs/Pose2D`

## 5. Perception (Track B) — bring the semantic map live

Goal: run SysNav's detection + 3D semantic mapping so `/object_nodes_list` is
populated, and the handlers answer from the **real map** (counts, object boxes,
located landmarks) instead of fallbacks. The orchestrator already subscribes
conditionally — once a publisher exists it uses it automatically, no code change.
Design + remaining gaps: see `PERCEPTION.md`.

Models (from the Colab sizing): **`yolov8x-worldv2` + `sam2.1_b` @ imgsz 960**,
detector prompted with our `config/challenge_classes.yaml`. Combined ≈ 2.3 GB —
fits 6 GB with the sim; latency (not VRAM) is the constraint.

### 5.1 Message interfaces — ✅ in the image
`tare_planner` msgs (`ObjectNodeList`, …) build with the image. After `git pull` +
rebuild, verify the type resolves and the orchestrator subscribes:

```bash
# T2 — orchestrator
docker exec -it iros2026_ai_module bash
ros2 launch vln_orchestrator vln_orchestrator.launch.py
#   expect log:  perception: subscribed to /object_nodes_list.
```
```bash
# new terminal — confirm the message type is built
docker exec -it iros2026_ai_module bash
ros2 topic info /object_nodes_list
#   expect:  Type: tare_planner/msg/ObjectNodeList
```
Handlers still fall back here (no publisher yet) — that is correct for 5.1.

### 5.2 Detection + semantic mapping — 🔜 next (large, one-time rebuild)
Adds the publisher. The image gains heavy deps (**torch, ultralytics/YOLO-World,
SAM2, open3d, scipy**) and builds SysNav's `detection_node` + `semantic_mapping`,
so the first `--build` is **slow** and the first launch **downloads model
weights** (YOLO-World + SAM2) once.

> Commands below are the **intended run layout**; the exact launch file/args are
> finalized when 5.2 lands (this section is updated then).

Run it as **4 terminals**:

```bash
# T1 — simulator + autonomy stack
docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```
```bash
# T2 — perception: detection + 3D semantic mapping (loads YOLO-World + SAM2,
#      detector prompted with challenge_classes.yaml; publishes /object_nodes_list)
docker exec -it iros2026_ai_module bash
ros2 launch semantic_mapping semantic_mapping_sim.launch
```
```bash
# T3 — orchestrator (export the key for VLM decomposition + attribute verification)
docker exec -it iros2026_ai_module bash
export GEMINI_API_KEY=<your_key>
ros2 launch vln_orchestrator vln_orchestrator.launch.py one_shot:=false
#   as objects are seen, handlers switch from fallback to the real map
```
```bash
# T4 — ask a question, then read the answer
docker exec -it iros2026_ai_module bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'How many pillows are on the bed'}"
ros2 topic echo /numerical_response --once
```

While it runs, watch (each in its own terminal):
```bash
ros2 topic echo /object_nodes_list --once     # is the map populating? (nodes[] non-empty)
watch -n2 nvidia-smi                           # GPU mem < ~6 GB; note temps
ros2 topic hz /object_nodes_list               # map update rate
```

**What "live" looks like:**
- numerical → a real count from the map (not the fallback `2`)
- object-ref → marker on the matched object's box (not a box at the vehicle)
- instruction → waypoints at located landmarks (not "hold position")

**If it OOMs or lags** (see Troubleshooting): drop SAM2 → `sam2.1_s`, imgsz → 640,
or run perception without the sim at full res. Exploration must also finish
within the **10-min/question** budget — watch end-to-end latency.

## 6. Full SysNav semantic exploration (Track B, proper architecture)

The robot must explore the unknown scene before answering. We run SysNav's
**room-aware semantic exploration** (modified TARE + vlm_node coordinator) inside
ai_module, driving the **plain** system via `/way_point`. Our orchestrator stays
the answerer. Key rule learned the hard way: launch everything **once, against one
clock** — never restart the sim under live perception (resets sim time →
`Detection older than oldest odom` → fusion dies).

After `git pull`, rebuild (heavy: PCL + the tare_planner C++ + or-tools):
```bash
cd ~/nevla-vln && git pull origin main
cd docker && docker compose -f compose_gpu.yml up --build -d ai_module
```
Confirm the new packages built:
```bash
docker exec iros2026_ai_module bash -lc "ros2 pkg list | grep -E 'tare_planner|vlm_node'"
```

### One-command form (eval-faithful)
```bash
# T1 — PLAIN system (NOT the with_exploration variant; TARE lives in ai_module)
docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```
```bash
# T2 — all ai_module nodes, one coordinated launch (perception first, TARE after a
#      delay so SAM2 is warm). Export the key for vlm_node + our VLM path.
docker exec -it iros2026_ai_module bash
export GEMINI_API_KEY=<your_key>
ros2 launch vln_orchestrator full_system.launch.py scenario:=indoor
```
```bash
# T3 — ask
docker exec -it iros2026_ai_module bash
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'How many chairs are there'}"
```

### Terminal-by-terminal form (debuggable; use for first bring-up)
Same as the one-command form but split so each node's output is visible. Bring up
T1 sim, then detection + semantic_mapping (wait for "started" / SAM2 loaded),
**then** vlm_node + TARE + room_segmentation, then the orchestrator, then ask:
```bash
# T2 detection
ros2 run semantic_mapping detection_node --ros-args -p annotate_image:=false -p use_sim_time:=true
# T3 mapping
ros2 run semantic_mapping semantic_mapping_node --ros-args -p use_sim_time:=true \
  -p object_file:=/home/docker/ai_module/src/semantic_mapping/semantic_mapping/config/objects.yaml
# T4 vlm coordinator (after T3 prints "started")
export GEMINI_API_KEY=<your_key>
ros2 launch vlm_node vlm_node_sim.launch use_sim_time:=true
# T5 TARE + room segmentation
ros2 launch tare_planner explore_world_sim.launch scenario:=indoor &
ros2 launch tare_planner room_segmentation.launch scenario:=indoor
# T6 orchestrator
ros2 run vln_orchestrator orchestrator --ros-args -p external_exploration:=true -p use_sim_time:=true
# T7 ask (as above)
```

**What success looks like:** TARE drives the robot **room-aware** (RViz),
`/object_nodes_list` fills, the orchestrator forwards the question to
`/keyboard_input` (vlm_node targets exploration), then answers from the full map
(chairs ≈ ground truth, not 0/partial). No `Detection older than oldest odom`.

**Tuning if needed:** `scenario:=` (indoor/matterport_sim), `explore_delay_s:=`
(SAM2 warm-up), and the convergence/budget params in `config/orchestrator.yaml`.

## Troubleshooting
- **`nvidia-smi` fails / "couldn't communicate with the NVIDIA driver"** → no host
  driver; see step 0a (install + reboot). The GPU showing in `lspci` but no
  `nvidia-driver` in `dpkg -l` confirms it's just not installed.
- **`Detection older than oldest odom` / `No neighboring cloud`** → clock desync.
  Ensure every node uses `use_sim_time:=true` and you did NOT restart the sim
  under live perception (relaunch the whole stack instead).
- **`libortools.so: cannot open shared object`** → the or-tools lib path isn't on
  `LD_LIBRARY_PATH`; the image sets it in `.bashrc` (use a login shell `bash -lc`).
- **`docker: unknown command: docker compose`** → Compose v2 plugin missing:
  `sudo apt install -y docker-compose-plugin` (or use hyphenated `docker-compose`).
- `docker` needs sudo → add user to docker group + re-login: `sudo usermod -aG docker $USER`.
- No sim window → re-run `xhost +` on the host.
- `permission denied (publickey)` on submodule → `.gitmodules` is HTTPS; re-run
  `git submodule update --init --recursive`.
- **GPU OOM during perception (low VRAM)** → use smaller SAM2/detector variants,
  lower the sim resolution, or run perception without the sim up simultaneously.
- Rebuild only our module: `docker compose -f compose_gpu.yml up --build -d ai_module`.
