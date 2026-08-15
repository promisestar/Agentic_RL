#!/usr/bin/env bash
# 将远程服务器上的项目修改同步回本地。
# 优先使用 rsync；若 Git Bash 中无 rsync，则回退为「逐项 scp -r + 排除列表」。
#
# 用法（在 Git Bash / WSL 中，于仓库根目录或任意处执行）:
#   bash scripts/sync_project_from_remote.sh
#   bash scripts/sync_project_from_remote.sh --dry-run
#   bash scripts/sync_project_from_remote.sh --delete   # 仅 rsync：删除本地多余文件

set -euo pipefail

REMOTE_USER="${REMOTE_USER:-hanshuaiteng}"
REMOTE_HOST="${REMOTE_HOST:-10.69.208.121}"
REMOTE_BASE="${REMOTE_BASE:-/data_160TB/2024/hanshuaiteng/LLM}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_PROJECT_NAME="${REMOTE_PROJECT_NAME:-$(basename "$PROJECT_ROOT")}"
REMOTE_DIR="${REMOTE_BASE}/${REMOTE_PROJECT_NAME}"
REMOTE_SPEC="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# === 显式 SSH 私钥路径 =================================================
# 当前 `bash` 实际是 WSL（本地路径形如 /mnt/g/...），不是 Git Bash。
# Windows 私钥在 WSL 中应写作：/mnt/c/Users/Raiden/.ssh/id_rsa
#
# 关键点：/mnt/c 上的文件权限常为 0777，OpenSSH 会拒绝使用并报
#   "UNPROTECTED PRIVATE KEY FILE" → Permission denied (publickey)
# 因此把密钥复制到 WSL 家目录并 chmod 600 后再给 rsync 使用。
WIN_SSH_KEY="${SSH_KEY:-/mnt/c/Users/Raiden/.ssh/id_rsa}"

if [[ ! -f "$WIN_SSH_KEY" ]]; then
  echo "错误: 私钥不存在: $WIN_SSH_KEY" >&2
  echo "请确认 Windows 上有 C:\\Users\\Raiden\\.ssh\\id_rsa" >&2
  exit 1
fi

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh" 2>/dev/null || true
SECURE_KEY="$HOME/.ssh/ecomrlve_sync_id_rsa"
# 仅当源更新或目标不存在时复制，避免每次改时间戳
if [[ ! -f "$SECURE_KEY" ]] || ! cmp -s "$WIN_SSH_KEY" "$SECURE_KEY" 2>/dev/null; then
  cp "$WIN_SSH_KEY" "$SECURE_KEY"
fi
chmod 600 "$SECURE_KEY"

echo "Using SSH key: $SECURE_KEY (copied from $WIN_SSH_KEY)"

if [[ -z "${RSYNC_RSH:-}" ]]; then
  export RSYNC_RSH="ssh -i $SECURE_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
fi

DRY_RUN=0
DELETE=0

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
  unsloth_compiled_cache
)

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
  "unsloth_compiled_cache/"
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
  local name="$1" x
  for x in "${EXCLUDE_NAMES[@]}"; do
    [[ "$name" == "$x" ]] && return 0
  done
  return 1
}

echo "Remote source: ${REMOTE_SPEC}"
echo "Local target : ${PROJECT_ROOT}/"
echo

pull_with_rsync() {
  local args=(-avz --progress) ex
  [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)
  [[ "$DELETE" -eq 1 ]] && args+=(--delete)
  for ex in "${RSYNC_EXCLUDES[@]}"; do
    args+=(--exclude="$ex")
  done
  # 注意方向：远程 -> 本地
  args+=("$REMOTE_SPEC" "${PROJECT_ROOT}/")
  echo "==> rsync ${args[*]}"
  rsync "${args[@]}"
}

pull_with_scp() {
  if [[ "$DELETE" -eq 1 ]]; then
    echo "警告: scp 回退模式不支持 --delete，已忽略。" >&2
  fi

  echo "提示: 未检测到 rsync，使用 scp -r 回退。"
  echo

  # 列出远端顶层项
  local remote_listing
  remote_listing="$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "ls -A '${REMOTE_DIR}'")"

  local names=()
  local name
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    if is_excluded "$name"; then
      echo "  skip: $name"
      continue
    fi
    names+=("$name")
  done <<< "$remote_listing"

  if [[ ${#names[@]} -eq 0 ]]; then
    echo "远端没有可同步的内容。" >&2
    exit 1
  fi

  echo "将拉取 ${#names[@]} 项:"
  for name in "${names[@]}"; do
    echo "  - $name"
  done
  echo

  if [[ "$DRY_RUN" -eq 1 ]]; then
    for name in "${names[@]}"; do
      echo "[DryRun] scp -r \"${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${name}\" \"${PROJECT_ROOT}/\""
    done
    return 0
  fi

  mkdir -p "$PROJECT_ROOT"
  echo "==> scp -r 逐项拉取..."
  for name in "${names[@]}"; do
    echo "  <- $name"
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${name}" "${PROJECT_ROOT}/"
  done
}

if command -v rsync >/dev/null 2>&1; then
  pull_with_rsync
else
  pull_with_scp
fi

echo
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DryRun] 演练完成，未实际写入本地。"
else
  echo "拉取完成: ${PROJECT_ROOT}/"
  echo "建议检查 diff 后再提交: git status / git diff"
fi
