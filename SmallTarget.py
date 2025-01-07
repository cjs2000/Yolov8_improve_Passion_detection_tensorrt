import os

# 文件夹路径
folder_path = "labels"

# 初始化计数器
count_less_than_threshold = 0
total_count = 0
threshold = 0.01
mid = 0
big =0

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 确保只处理 .txt 文件
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        # 打开并读取文件内容
        with open(file_path, 'r') as file:
            for line in file:
                # 去掉首尾空格并分割每行数据
                data = line.strip().split()
                
                # 确保每行有足够的数据（至少 4 列）
                if len(data) >= 4:
                    # 提取后两列并计算乘积
                    num1 = float(data[-2])
                    num2 = float(data[-1]) 
                    print(str(num1) + " " + str(num2))            
                    product = num1 * num2
                    total_count += 1
                    # 判断乘积是否小于阈值
                    if product < 0.01:
                        count_less_than_threshold += 1
                    if product >= 0.01 and product < 0.02:
                        mid += 1
                    if product >= 0.02:
                        big += 1


# 输出结果
print(f"总数为: {total_count}")
print(f"small: {count_less_than_threshold}")
print(f"mid: {mid}")
print(f"big: {big}")
