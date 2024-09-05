import os

# 定义当前目录路径
current_directory = '.'

# 初始化一个计数器用于文件编号
file_counter = 1

# 遍历当前目录下的所有文件（包括子文件夹）
for root, dirs, files in os.walk(current_directory):
    for file_name in sorted(files):
        # 获取文件的扩展名
        file_ext = os.path.splitext(file_name)[1]

        # 构建新的文件名，例如 'cel1.jpg', 'cel2.jpg'
        new_name = f"cel{file_counter}{file_ext}"

        # 构建旧的和新的文件路径
        old_path = os.path.join(root, file_name)
        new_path = os.path.join(root, new_name)

        # 检查是否已存在同名文件，避免冲突
        if os.path.exists(new_path):
            print(f"文件 {new_path} 已存在，跳过重命名。")
        else:
            # 重命名文件
            os.rename(old_path, new_path)
            print(f"重命名: {old_path} -> {new_path}")

        # 更新计数器
        file_counter += 1

print("文件已成功排序并重命名！")
