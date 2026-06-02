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

**Expected (this stage):** T2 classifies each question, decomposes via Gemini,
logs `perception: … fallback mode`, and publishes on the right topic
(`/numerical_response`, `/selected_object_marker`, `/way_point_with_heading`).
Fallback mode is expected until SysNav perception is wired (next phase).

## Output contract (must not change)
- `/numerical_response` — `std_msgs/Int32`
- `/selected_object_marker` — `visualization_msgs/Marker` (CUBE)
- `/way_point_with_heading` — `geometry_msgs/Pose2D`

## Next phase — perception (see PERCEPTION.md)
Stand up SysNav detection + semantic_mapping so `/object_nodes_list` goes live;
the orchestrator then auto-subscribes and the handlers use the real semantic map.

## Troubleshooting
- **`nvidia-smi` fails / "couldn't communicate with the NVIDIA driver"** → no host
  driver; see step 0a (install + reboot). The GPU showing in `lspci` but no
  `nvidia-driver` in `dpkg -l` confirms it's just not installed.
- **`docker: unknown command: docker compose`** → Compose v2 plugin missing:
  `sudo apt install -y docker-compose-plugin` (or use hyphenated `docker-compose`).
- `docker` needs sudo → add user to docker group + re-login: `sudo usermod -aG docker $USER`.
- No sim window → re-run `xhost +` on the host.
- `permission denied (publickey)` on submodule → `.gitmodules` is HTTPS; re-run
  `git submodule update --init --recursive`.
- **GPU OOM during perception (low VRAM)** → use smaller SAM2/detector variants,
  lower the sim resolution, or run perception without the sim up simultaneously.
- Rebuild only our module: `docker compose -f compose_gpu.yml up --build -d ai_module`.
