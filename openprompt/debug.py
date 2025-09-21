import sys
import os

print("--- 终极诊断脚本开始 ---")

# 1. 打印原始 sys.path
print("\n[1] 原始 Python 模块搜索路径 (sys.path):")
for p in sys.path:
    print(f"  - {p}")

# 2. 手动将工作区添加到路径 (再次确认)
workspace_path = '/workspace'
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)
print("\n[2] 将 /workspace 添加到搜索路径后:")
for p in sys.path:
    print(f"  - {p}")
    
# 3. 递归列出 /workspace 下的所有文件和目录
print("\n[3] 用 Python 递归列出 /workspace 目录下的所有内容:")
found_openprompt = False
try:
    for root, dirs, files in os.walk(workspace_path):
        # 计算相对于 /workspace 的路径，以便于查看
        relative_root = os.path.relpath(root, workspace_path)
        if relative_root == '.':
            relative_root = ''

        print(f"  目录: ./{relative_root}")
        
        # 打印子目录
        for d in sorted(dirs):
            print(f"    - 子目录: {d}")
            if d == 'openprompt':
                found_openprompt = True
        
        # 打印文件
        for f in sorted(files):
            print(f"    - 文件: {f}")
except FileNotFoundError:
    print(f"  错误: 无法找到目录 '{workspace_path}'！这说明卷挂载可能完全失败了。")
    
# 4. 打印最终诊断结论
print("\n[4] 诊断结论:")
if found_openprompt:
    print("  ✅ 成功在 /workspace 的子目录中找到了 'openprompt' 文件夹。")
    print("  ... 这意味着问题极其罕见，可能与文件权限或Python的内部导入缓存有关。")
else:
    print("  ❌ 未能在 /workspace 的子目录中找到 'openprompt' 文件夹。")
    print("  ... 这意味着 Docker 的卷挂载未能按预期工作，或者您本地的项目结构不正确。")
    print("  ... 请务必再次确认您本地的 MSP-master 文件夹下，确实存在一个名为 openprompt 的子文件夹。")

print("\n--- 终极诊断脚本结束 ---")