#!/usr/bin/env bash
# 将本项目同步到远程服务器。
# 优先使用 rsync；若 Git Bash 中无 rsync，则回退为「逐项 scp -r + 排除列表」。
#
# 用法（在 Git Bash 中）:
#   bash scripts/sync_project_to_remote.sh
#   bash scripts/sync_project_to_remote.sh --dry-run
#   bash scripts/sync_project_to_remote.sh --delete   # 仅 rsync 模式有效

set -euo pipefail

REMOTE_USER="${REMOTE_USER:-hanshuaiteng}"
REMOTE_HOST="${REMOTE_HOST:-10.69.208.121}"
REMOTE_BASE="${REMOTE_BASE:-/data_160TB/2024/hanshuaiteng/LLM}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_PROJECT_NAME="${REMOTE_PROJECT_NAME:-$(basename "$PROJECT_ROOT")}"
REMOTE_DIR="${REMOTE_BASE}/${REMOTE_PROJECT_NAME}"
REMOTE_SPEC="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# === 显式 SSH 私钥路径（按需修改） ===========================================
# 留空则走 ssh 默认查找逻辑；想强制指定请填写绝对路径。
SSH_KEY="${SSH_KEY:-/c/Users/Raiden/.ssh/id_rsa}"

if [[ -n "$SSH_KEY" && ! -f "$SSH_KEY" ]]; then
  echo "警告: 指定的 SSH_KEY=$SSH_KEY 不存在。" >&2
fi

if [[ -z "${RSYNC_RSH:-}" ]]; then
  if [[ -n "$SSH_KEY" ]]; then
    export RSYNC_RSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o BatchMode=no"
  else
    export RSYNC_RSH="ssh -o StrictHostKeyChecking=accept-new -o BatchMode=no"
  fi
fi

DRY_RUN=0
DELETE=0

# 顶层排除名（scp 回退模式按「顶层目录/文件名」匹配）
EXCLUDE_NAMES=(
  .venv
  venv
  .git
  __pycache__
  .pytest_cache
  .ruff_cache
  .mypy_cache
  outputs
  results
  .env
)

# rsync 排除模式
RSYNC_EXCLUDES=(
  ".venv/"
  "venv/"
  ".git/"
  "__pycache__/"
  "*.pyc"
  ".pytest_cache/"
  ".ruff_cache/"
  ".mypy_cache/"
  "outputs/"
  "results/"
  ".env"
  "space/.env"
)

for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    --delete) DELETE=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "未知参数: $arg" >&2
      exit 1
      ;;
  esac
done

is_excluded() {
  local name="$1"
  local x
  for x in "${EXCLUDE_NAMES[@]}"; do
    if [[ "$name" == "$x" ]]; then
      return 0
    fi
  done
  return 1
}

echo "Local source : ${PROJECT_ROOT}/"
echo "Remote target: ${REMOTE_SPEC}"
echo

sync_with_rsync() {
  local args=(-avz --progress)
  local ex
  [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)
  [[ "$DELETE" -eq 1 ]] && args+=(--delete)
  for ex in "${RSYNC_EXCLUDES[@]}"; do
    args+=(--exclude="$ex")
  done
  args+=("${PROJECT_ROOT}/" "$REMOTE_SPEC")
  echo "==> rsync ${args[*]}"
  rsync "${args[@]}"
}

sync_with_scp() {
  if [[ "$DELETE" -eq 1 ]]; then
    echo "警告: scp 回退模式不支持 --delete，已忽略。" >&2
  fi

  echo "提示: 未检测到 rsync，使用 scp -r 回退（按顶层项排除）。"
  echo "      若要安装 rsync，见脚本末尾说明。"
  echo

  local items=()
  local p name
  for p in "$PROJECT_ROOT"/* "$PROJECT_ROOT"/.[!.]* "$PROJECT_ROOT"/..?*; do
    [[ -e "$p" ]] || continue
    name="$(basename "$p")"
    [[ "$name" == "." || "$name" == ".." ]] && continue
    if is_excluded "$name"; then
      echo "  skip: $name"
      continue
    fi
    items+=("$p")
  done

  if [[ ${#items[@]} -eq 0 ]]; then
    echo "没有可上传的内容。" >&2
    exit 1
  fi

  echo "将上传 ${#items[@]} 项:"
  local item
  for item in "${items[@]}"; do
    echo "  - $(basename "$item")"
  done
  echo

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DryRun] ssh ${REMOTE_USER}@${REMOTE_HOST} \"mkdir -p '${REMOTE_DIR}'\""
    for item in "${items[@]}"; do
      echo "[DryRun] scp -r \"$item\" \"${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/\""
    done
    return 0
  fi

  echo "==> 创建远端目录..."
  ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}'"

  echo "==> scp -r 逐项上传..."
  for item in "${items[@]}"; do
    echo "  -> $(basename "$item")"
    scp -r "$item" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
  done
}

if command -v rsync >/dev/null 2>&1; then
  sync_with_rsync
else
  sync_with_scp
fi

echo
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DryRun] 演练完成。"
else
  echo "同步完成: ${REMOTE_SPEC}"
fi

cat <<'EOF'

------------------------------------------------------------
若希望在 Git Bash 中使用真正的 rsync，可选其一：

1) Scoop（推荐）
   scoop install rsync

2) MSYS2
   pacman -S rsync

3) WSL
   wsl -e bash -lc 'sudo apt-get update && sudo apt-get install -y rsync'
   然后在 WSL 里进入项目目录再跑本脚本，或直接用 wsl 调 rsync。

安装后重新打开 Git Bash，确认：
   which rsync
------------------------------------------------------------
EOF
