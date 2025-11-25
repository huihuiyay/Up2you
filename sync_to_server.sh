#!/bin/bash

# UP2You - 同步修复到服务器脚本
# 用途：将本地修复的文件同步到服务器路径

LOCAL_PATH="/media/yuzihui/17A26BDB4251F93F/project/UP2You"
SERVER_PATH="/share/mas/huangping/yuzihui/up2you"

echo "================================================"
echo "UP2You 代码同步工具"
echo "================================================"
echo ""
echo "本地路径: $LOCAL_PATH"
echo "服务器路径: $SERVER_PATH"
echo ""

# 检查服务器路径是否存在
if [ ! -d "$SERVER_PATH" ]; then
    echo "❌ 错误：服务器路径不存在: $SERVER_PATH"
    echo "请确认路径是否正确，或者服务器是否已挂载"
    exit 1
fi

echo "准备同步以下文件:"
echo ""

# 需要同步的文件列表
FILES_TO_SYNC=(
    "inference_thg.py"
    "up2you/schedulers/scheduling_thg.py"
    "up2you/pipelines/pipeline_mvpuzzle_i2mv_sd21.py"
    "up2you/pipelines/pipeline_mvpuzzle_mv2normal_sd21.py"
    "up2you/utils/mesh_utils/reconstructor.py"
    "run_thg.sh"
)

# 检查哪些文件存在差异
NEEDS_SYNC=0
for file in "${FILES_TO_SYNC[@]}"; do
    if [ ! -f "$LOCAL_PATH/$file" ]; then
        echo "⚠️  本地文件不存在: $file"
        continue
    fi

    if [ ! -f "$SERVER_PATH/$file" ]; then
        echo "🆕 新增文件: $file"
        NEEDS_SYNC=1
    elif ! diff -q "$LOCAL_PATH/$file" "$SERVER_PATH/$file" > /dev/null 2>&1; then
        echo "📝 已修改: $file"
        NEEDS_SYNC=1
    else
        echo "✅ 相同: $file"
    fi
done

echo ""

if [ $NEEDS_SYNC -eq 0 ]; then
    echo "✅ 所有文件已同步，无需更新"
    exit 0
fi

echo "================================================"
read -p "是否继续同步? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消同步"
    exit 0
fi

echo ""
echo "开始同步..."
echo ""

# 创建备份目录
BACKUP_DIR="$SERVER_PATH/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "📦 备份目录: $BACKUP_DIR"
echo ""

# 同步文件
SUCCESS_COUNT=0
FAIL_COUNT=0

for file in "${FILES_TO_SYNC[@]}"; do
    if [ ! -f "$LOCAL_PATH/$file" ]; then
        continue
    fi

    # 备份服务器上的旧文件（如果存在）
    if [ -f "$SERVER_PATH/$file" ]; then
        backup_file="$BACKUP_DIR/$file"
        mkdir -p "$(dirname "$backup_file")"
        cp "$SERVER_PATH/$file" "$backup_file"
        echo "  💾 备份: $file"
    fi

    # 确保目标目录存在
    target_dir="$(dirname "$SERVER_PATH/$file")"
    mkdir -p "$target_dir"

    # 复制文件
    if cp "$LOCAL_PATH/$file" "$SERVER_PATH/$file"; then
        echo "  ✅ 同步: $file"
        ((SUCCESS_COUNT++))
    else
        echo "  ❌ 失败: $file"
        ((FAIL_COUNT++))
    fi
done

echo ""
echo "================================================"
echo "同步完成"
echo "================================================"
echo "成功: $SUCCESS_COUNT 个文件"
echo "失败: $FAIL_COUNT 个文件"
echo "备份: $BACKUP_DIR"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✅ 所有文件同步成功！"
    echo ""
    echo "下一步："
    echo "1. cd $SERVER_PATH"
    echo "2. chmod +x run_thg.sh"
    echo "3. ./run_thg.sh"
    echo ""
else
    echo "⚠️  部分文件同步失败，请检查权限"
fi

echo "================================================"
