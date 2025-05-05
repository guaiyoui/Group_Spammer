import os

# 读取有向图数据
input_file = "J01Network_direct.txt"
output_file = "J01Network.txt"

# 确保输出目录存在
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 存储所有边的集合，用于去重
edges = set()

# 读取原始有向边
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # 解析边的两个顶点
        parts = line.split()
        if len(parts) >= 2:
            src, dst = parts[0], parts[1]
            
            # 添加原始边和反向边
            edges.add((src, dst))
            edges.add((dst, src))

# 写入无向图（包含双向边）
with open(output_file, 'w') as f:
    for src, dst in edges:
        f.write(f"{src} {dst}\n")

print(f"转换完成：从 {input_file} 创建无向图 {output_file}")
print(f"边数量：{len(edges)}")
