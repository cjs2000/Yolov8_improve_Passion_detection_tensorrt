import os

# 设置文件夹路径
folder_path = "labels"  # 替换为你的文件夹路径

# 初始化计数器
count = 0

# 遍历文件夹中的所有txt文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            # 读取文件内容，统计有效行数
            valid_lines = sum(1 for line in file if line.strip())  # 统计非空行
        if valid_lines > 8:
            count += 1

# 输出结果
print(f"满足条件的txt文件数量: {count}")
