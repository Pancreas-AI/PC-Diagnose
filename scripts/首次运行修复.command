#!/bin/zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/../app_source" && pwd)"

/usr/bin/xattr -dr com.apple.quarantine "$APP_DIR" 2>/dev/null || true
/bin/chmod +x "$SCRIPT_DIR/启动胰腺癌辅助诊断.command" 2>/dev/null || true
/bin/chmod +r "$APP_DIR/diagnosis_app.py" "$APP_DIR/diagnosis_model.py" 2>/dev/null || true

osascript -e 'display alert "修复完成" message "已解除下载隔离标记并修复启动脚本权限。请重新运行启动脚本。"'
