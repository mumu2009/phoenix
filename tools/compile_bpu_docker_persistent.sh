#!/usr/bin/env bash
# Manage a persistent Docker container for BPU compilation.
#
# Instead of spawning a new "docker run --rm" for every single candidate
# (which costs ~5s container startup + ~14s hb_mapper import each time),
# this script keeps a single container running with a compile worker process
# that imports hb_mapper once and accepts compile jobs via a UNIX socket.
#
# Subcommands:
#   start   - Ensure the persistent container and worker are running
#   stop    - Stop the persistent container
#   status  - Check if the container and worker are running
#   compile - Send a batch of compile jobs to the worker
#   restart - Stop then start
#
# The compile subcommand accepts JSON on stdin and writes the result to stdout.
#
# Usage:
#   # Start the worker
#   bash compile_bpu_docker_persistent.sh start
#
#   # Compile a batch (JSON on stdin)
#   echo '{"action":"compile","jobs":[...],"parallel":2}' | \
#       bash compile_bpu_docker_persistent.sh compile
#
#   # Stop the worker
#   bash compile_bpu_docker_persistent.sh stop

set -uo pipefail

DOCKER_IMAGE="${DOCKER_IMAGE:-openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310}"
HOST_WORK="${HOST_WORK:-/home/kali/phoenix}"
CONTAINER_WORK="/workspace"
CONTAINER_NAME="${CONTAINER_NAME:-bpu_compile_worker}"
SOCK_PATH="/tmp/bpu_compile.sock"
HOST_SOCK="/tmp/bpu_compile.sock"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

is_container_running() {
    docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q true
}

_worker_socket_exists() {
    is_container_running && \
    docker exec "$CONTAINER_NAME" test -S "$SOCK_PATH" 2>/dev/null
}

is_worker_running() {
    _worker_socket_exists || return 1
    # Write the ping script to a temp file to avoid shell quoting issues,
    # then execute it inside the container.
    docker exec "$CONTAINER_NAME" python3 -c \
        'import socket,json,sys;s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.settimeout(5);s.connect("/tmp/bpu_compile.sock");s.sendall(b"{\"action\":\"ping\"}\n");d=s.recv(4096);s.close();r=json.loads(d);sys.exit(0 if r.get("msg")=="pong" else 1)' \
        2>/dev/null
}

cmd_start() {
    if is_worker_running; then
        log "Worker already running in container $CONTAINER_NAME"
        return 0
    fi

    # Remove any stale container
    if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        log "Removing stale container $CONTAINER_NAME ..."
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
    fi

    log "Starting persistent container $CONTAINER_NAME ..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        -v "$HOST_WORK:$CONTAINER_WORK" \
        -v "/tmp:/host_tmp" \
        --memory 9g \
        --cpus 10 \
        "$DOCKER_IMAGE" \
        sleep infinity

    # Wait for container to be ready
    for i in $(seq 1 10); do
        if is_container_running; then break; fi
        sleep 1
    done

    if ! is_container_running; then
        log "ERROR: container failed to start"
        return 1
    fi

    log "Starting compile worker inside container ..."
    docker exec -d "$CONTAINER_NAME" \
        python3 -u "$CONTAINER_WORK/tools/compile_bpu_worker.py" \
        --sock "$SOCK_PATH"

    # Wait for socket file to appear (fast check, no docker exec python3).
    # The worker imports hb_mapper (~7-15s) then creates the socket.
    log "Waiting for worker socket ..."
    local ready=false
    for i in $(seq 1 30); do
        if _worker_socket_exists; then
            ready=true
            break
        fi
        sleep 1
    done

    if ! $ready; then
        log "ERROR: socket not created within 30s"
        return 1
    fi

    # Ping with retries (worker may still be initializing after socket creation)
    for try in $(seq 1 5); do
        sleep 1
        if is_worker_running; then
            log "Worker ready!"
            return 0
        fi
        log "Ping attempt $try failed, retrying ..."
    done

    log "ERROR: worker socket exists but ping failed after 5 retries"
    return 1
}

cmd_stop() {
    if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        log "Stopping container $CONTAINER_NAME ..."
        docker stop -t 5 "$CONTAINER_NAME" 2>/dev/null
        docker rm -f "$CONTAINER_NAME" 2>/dev/null
        log "Stopped."
    else
        log "Container $CONTAINER_NAME not running."
    fi
}

cmd_status() {
    if is_worker_running; then
        echo "running"
    elif is_container_running; then
        echo "container_only"
    else
        echo "stopped"
    fi
}

cmd_compile() {
    # Ensure worker is running
    if ! is_worker_running; then
        log "Worker not running, starting ..."
        cmd_start || return 1
    fi

    # Read JSON from stdin, write to a temp file, then send via socket
    local input tmpfile
    input=$(cat)
    tmpfile=$(mktemp /tmp/bpu_compile_req_XXXXXX.json)
    echo "$input" > "$tmpfile"

    # Copy temp file into container and send to worker via socket
    docker cp "$tmpfile" "$CONTAINER_NAME:/tmp/bpu_compile_req.json"
    rm -f "$tmpfile"

    docker exec "$CONTAINER_NAME" python3 -c '
import socket, json, sys

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(900)
s.connect("/tmp/bpu_compile.sock")

with open("/tmp/bpu_compile_req.json", "r") as f:
    data = f.read().strip()
s.sendall(data.encode() + b"\n")

# Read response (may be large, read until newline)
buf = b""
while True:
    chunk = s.recv(1048576)
    if not chunk:
        break
    buf += chunk
    if b"\n" in buf:
        break
s.close()

line = buf.split(b"\n", 1)[0]
sys.stdout.write(line.decode() + "\n")
sys.stdout.flush()
'
}

cmd_restart() {
    cmd_stop
    cmd_start
}

# --- Fallback: legacy single-model compile (for backward compat) ---
cmd_compile_legacy() {
    # This emulates the old compile_bpu_docker.sh interface for a single model.
    # Parse the same arguments and convert to a batch job.
    local model_name="" onnx="" calib_dir="" input_name="" input_shape=""
    local out_dir="" per_channel="True" calib_type="kl" march="bayes-e"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-name) model_name="$2"; shift 2 ;;
            --onnx) onnx="$2"; shift 2 ;;
            --calib-dir) calib_dir="$2"; shift 2 ;;
            --input-name) input_name="$2"; shift 2 ;;
            --input-shape) input_shape="$2"; shift 2 ;;
            --out-dir) out_dir="$2"; shift 2 ;;
            --per-channel) per_channel="$2"; shift 2 ;;
            --calib-type) calib_type="$2"; shift 2 ;;
            --march) march="$2"; shift 2 ;;
            --run-hb-mapper) shift 2 ;;  # ignored, worker handles it
            *) shift ;;
        esac
    done

    # Rewrite host paths to container paths
    onnx="${onnx/#$HOST_WORK/$CONTAINER_WORK}"
    calib_dir="${calib_dir/#$HOST_WORK/$CONTAINER_WORK}"
    out_dir="${out_dir/#$HOST_WORK/$CONTAINER_WORK}"

    local pc_json="true"
    [[ "$per_channel" == "False" || "$per_channel" == "false" ]] && pc_json="false"

    local job_json
    job_json=$(cat <<ENDJSON
{
  "action": "compile",
  "parallel": 1,
  "jobs": [{
    "job_id": "$model_name",
    "model_name": "$model_name",
    "onnx_path": "$onnx",
    "calib_dir": "$calib_dir",
    "input_name": "$input_name",
    "input_shape": "$input_shape",
    "out_dir": "$out_dir",
    "per_channel": $pc_json,
    "calib_type": "$calib_type",
    "march": "$march"
  }]
}
ENDJSON
)

    echo "$job_json" | cmd_compile
}


# --- Main ---
case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    compile) shift; cmd_compile "$@" ;;
    restart) cmd_restart ;;
    # Legacy compatibility: if called with --model-name etc., use the old interface
    --model-name|--onnx|--backend)
        cmd_compile_legacy "$@" ;;
    *)
        echo "Usage: $0 {start|stop|status|compile|restart}" >&2
        echo "  Or legacy: $0 --model-name ... --onnx ... (same as compile_bpu_docker.sh)" >&2
        exit 1 ;;
esac
