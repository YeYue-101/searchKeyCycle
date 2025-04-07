import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
import ast  # 用于安全解析_pos列的数组

def find_key_cycles(G):
    # Step 1: 计算拉普拉斯矩阵
    L = nx.laplacian_matrix(G).astype(float)

    # Step 2: 计算Fiedler值和Fiedler向量
    # 使用稀疏矩阵计算前3个最小特征值（避免0特征值）
    eigenvalues, eigenvectors = eigsh(L, k=3, which='SM')
    # 找到第二小的特征值（Fiedler值）
    fiedler_index = np.argsort(eigenvalues)[1]
    fiedler_vector = eigenvectors[:, fiedler_index]
    # 归一化Fiedler向量
    fiedler_vector = fiedler_vector / np.linalg.norm(fiedler_vector)

    # Step 3: 查找所有简单循环（限制最大长度以提高效率）
    cycles = list(nx.simple_cycles(G))
    # 过滤掉长度小于3的循环（非环）
    cycles = [c for c in cycles if len(c) >= 3]

    # Step 4: 计算每个循环的重要性指标I_C
    cycle_scores = []
    for cycle in cycles:
        score = 0.0
        # 将循环转换为边列表（注意节点顺序）
        edges = list(zip(cycle, cycle[1:] + cycle[:1]))
        for u, v in edges:
            # 获取节点在图中的索引（需确保节点标签与索引一致）
            u_idx = list(G.nodes()).index(u)
            v_idx = list(G.nodes()).index(v)
            diff = fiedler_vector[u_idx] - fiedler_vector[v_idx]
            score += diff ** 2
        cycle_scores.append((cycle, score))

    # Step 5: 按I_C从大到小排序
    sorted_cycles = sorted(cycle_scores, key=lambda x: -x[1])

    return sorted_cycles


# 读取节点数据
def parse_pos(pos_str):
    """将字符串'array([x, y])'转换为元组(x, y)"""
    try:
        # 提取括号内的内容并转换为元组
        return ast.literal_eval(pos_str.replace("array", ""))
    except:
        return (0.0, 0.0)

nodes = pd.read_csv(
    "./network.csv/1nodes.csv",
    comment="#",
    names=["index", "_graphml_vertex_id", "id", "name", "x", "y", "z", "_pos"],
    converters={"_pos": parse_pos}  # 自动转换_pos列
)

# 读取边数据
edges = pd.read_csv(
    "./network.csv/1edges.csv",
    comment="#",
    names=["source", "target", "_graphml_edge_id", "weight"],
    dtype={"source": int, "target": int}
)

# 数据验证
print("节点数量:", len(nodes))
print("边数量:", len(edges))
print("\n前5个节点示例:")
print(nodes.head())
print("\n前5条边示例:")
print(edges.head())


# 可选：转换为networkx图对象
try:
    G = nx.from_pandas_edgelist(edges, "source", "target", edge_attr="weight")
    print("\n已创建网络图，包含 {} 个节点和 {} 条边".format(G.number_of_nodes(), G.number_of_edges()))
except ImportError:
    print("\n（安装networkx包后可生成图对象）")

# 查找关键循环
key_cycles = find_key_cycles(G)
 # 输出结果
for i, (cycle, score) in enumerate(key_cycles):
    print(f"第{i + 1}名: 循环{cycle}, 重要性得分: {score:.4f}")






































