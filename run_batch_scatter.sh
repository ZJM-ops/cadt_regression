#!/bin/bash

# ============================================================
# 批量为 results4 下带 emotion 的实验生成散点图
# ============================================================

RESULTS_DIR="/root/zjm/cadt/results4"
DATA_DIR="/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new"
PLOT_SCRIPT="/root/zjm/cadt/plot_scatter.py"
LOG_FILE="${RESULTS_DIR}/batch_scatter_log.txt"

# 清空日志
echo "批量散点图生成日志 - $(date)" > "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

# 遍历 results4 下所有带 emotion 的目录
for dir in "$RESULTS_DIR"/*emotion*; do
    if [ -d "$dir" ]; then
        exp_name=$(basename "$dir")
        echo "" | tee -a "$LOG_FILE"
        echo ">>> 正在处理实验: $exp_name" | tee -a "$LOG_FILE"
        
        results_file="$dir/results.txt"
        target_subject=""
        target_domain_ids=""
        
        if [ -f "$results_file" ]; then
            # 1. 尝试获取 target_subject
            target_subject=$(grep -i 'Target Subject' "$results_file" | grep -oP '\d+' || echo "")
            
            # 2. 尝试获取 target_domain_ids
            # 格式: Target Domain IDs: [1, 2, 3]
            target_domain_ids=$(grep -i 'Target Domain IDs' "$results_file" | grep -oP '\[\K[^\]]+' | tr -d ' ' || echo "")
        fi
        
        # 决定传递给脚本的参数
        extra_args=""
        if [ -n "$target_domain_ids" ]; then
            echo "  [信息] 找到 Target Domain IDs: $target_domain_ids" | tee -a "$LOG_FILE"
            extra_args="--target_domain_ids $target_domain_ids"
        elif [ -n "$target_subject" ]; then
            echo "  [信息] 找到 Target Subject: $target_subject" | tee -a "$LOG_FILE"
            extra_args="--target_subject $target_subject"
        else
            echo "  [警告] 既找不到 Target Subject 也找不到 Target Domain IDs, 跳过" | tee -a "$LOG_FILE"
            continue
        fi
        
        # 3. 查找模型文件
        model_path="$dir/model_best_target0.pth"
        if [ ! -f "$model_path" ]; then
            # 如果没有默认名称，尝试找第一个 .pth 文件
            model_path=$(find "$dir" -maxdepth 1 -name "*.pth" -o -name "*.pt" | head -n 1)
        fi
        
        if [ -z "$model_path" ] || [ ! -f "$model_path" ]; then
            echo "  [警告] 找不到模型文件 (.pth), 跳过" | tee -a "$LOG_FILE"
            continue
        fi
        echo "  [信息] model_path = $model_path" | tee -a "$LOG_FILE"
        
        # 4. 设置输出目录 (在实验目录下创建 scatter_plots 文件夹)
        output_dir="$dir/scatter_plots"
        mkdir -p "$output_dir"
        echo "  [信息] output_dir = $output_dir" | tee -a "$LOG_FILE"
        
        # 5. 执行散点图绘制脚本
        echo "  [执行] python $PLOT_SCRIPT ..." | tee -a "$LOG_FILE"
        cd /root/zjm/cadt
        
        # 使用 eval 来正确处理可能包含空格的参数
        eval python "$PLOT_SCRIPT" \
            --model_path "$model_path" \
            --data_dir "$DATA_DIR" \
            --output_dir "$output_dir" \
            --device 0 \
            --batch_size 128 \
            $extra_args >> "$LOG_FILE" 2>&1
        
        # 检查执行结果
        if [ $? -eq 0 ]; then
            echo "  [成功] 完成散点图绘制!" | tee -a "$LOG_FILE"
        else
            echo "  [失败] 脚本执行出错, 请查看日志" | tee -a "$LOG_FILE"
        fi
        
        echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "所有任务完成! 日志已保存至: $LOG_FILE" | tee -a "$LOG_FILE"