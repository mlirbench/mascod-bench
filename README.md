# MASCoD-MLIRBench

MASCoD-MLIRBench is a benchmark framework designed for evaluating semantic reasoning and semantic divergence in MLIR-based compiler intermediate representations using multi-agent workflows.

This repository includes:

- Mutation generation workflows
- Execution sandbox validation
- Semantic reasoning infrastructure
- Dataset generation pipelines
- MLIR dialect support for `linalg` and `torch`

---

# System Requirements

The framework was tested on Linux-based HPC infrastructure using:

- NVIDIA A100 GPU nodes
- High-memory compute configuration
- Multi-core CPU allocation
- MLIR + Torch-MLIR toolchains
- Python 3.10+

You may adapt the infrastructure configuration based on your own environment and cluster setup.

---

# Setup Instructions

## 1. Clone / Extract the Repository

Extract the repository zip file and enter the project directory:

```bash
cd mascod-bench
```

---

## 2. Create a Python Virtual Environment

Create and activate a new Python environment:

### Linux / macOS

```bash
python3 -m venv mascod_env
source mascod_env/bin/activate
```

### Windows

```powershell
python -m venv mascod_env
mascod_env\Scripts\activate
```

---

## 3. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

# Running the Framework

The framework typically uses three terminals:

- Terminal 1 -> Execution server
- Terminal 2 -> Reverse tunnel
- Terminal 3 -> Connectivity validation and proxy

---

# Terminal 1 — Execution Server

## Step 1: Start Compute Session

Start a compute session on your HPC or GPU node.

Example configuration used during development:

- High-memory node
- Multi-core CPU allocation
- Long-running interactive session
- NVIDIA A100 GPU

Adjust the configuration according to your infrastructure.

---

## Step 2: Load MLIR Environment

Load your MLIR and Torch-MLIR environment configuration.

Example:

```bash
source ~/mlir_env.sh
```

This environment script should configure:

- LLVM
- MLIR
- Torch-MLIR
- PATH variables
- LD_LIBRARY_PATH variables

---

## Step 3: Start the Server

Run:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

This starts the execution sandbox server.

---

# Terminal 2 — Reverse Tunnel

Create a reverse SSH tunnel from your compute node to your externally reachable host.

Example structure:

```bash
ssh -N -T -R <REMOTE_PORT>:127.0.0.1:<LOCAL_PORT> <USER>@<HOST>
```

Notes:

- The command may appear "stuck"
- This is expected behavior
- The tunnel remains active while the process is running

---

# Terminal 3 — Connectivity Validation

## Step 1: SSH into the External Host

SSH into the host machine where the reverse tunnel is exposed.

---

## Step 2: Validate Connectivity

Verify that the tunneled endpoint is reachable:

```bash
curl http://localhost:<PORT>/
```

If configured correctly, the server should respond successfully.

---

## Step 3: Optional Local Proxy

If additional forwarding is required, you may run a lightweight TCP proxy.

Example proxy server:

```python
import socket
import threading

LISTEN_PORT = 8080
TARGET_HOST = "127.0.0.1"
TARGET_PORT = 9000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", LISTEN_PORT))
server.listen(5)

print(f"Proxy: 0.0.0.0:{LISTEN_PORT} → {TARGET_HOST}:{TARGET_PORT}")

def forward(src, dst):
    try:
        while True:
            data = src.recv(4096)
            if not data:
                break
            dst.sendall(data)
    except:
        pass
    finally:
        src.close()
        dst.close()

while True:
    client, _ = server.accept()

    target = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    target.connect((TARGET_HOST, TARGET_PORT))

    threading.Thread(target=forward, args=(client, target)).start()
    threading.Thread(target=forward, args=(target, client)).start()
```

---

# Notes

- Ensure all ports are open and accessible according to your infrastructure policies
- The execution server must remain running while the framework is active
- Reverse tunnels must remain active throughout execution
- Torch dialect validation may require Torch-MLIR lowering pipelines
- HPC environments may require additional module loading depending on cluster configuration

---

# Supported Dialects

Currently supported MLIR dialects:

- `linalg`
- `torch`

---

# Leaderboard

The MASCoD-MLIRBench leaderboard and benchmark results are available at:

https://mascod-bench.github.io/mascod-bench/index.html

The leaderboard includes:
- Model performance comparisons
- Semantic reasoning evaluation metrics
- Mutation-level breakdowns
- Dialect-specific benchmarking results
- Execution validation statistics

---

# Citation

If you use MASCoD-MLIRBench in academic work, please cite the corresponding paper once published.

---

# License

This repository is provided for research and educational purposes.
