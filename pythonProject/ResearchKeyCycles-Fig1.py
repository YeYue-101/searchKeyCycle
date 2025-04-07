import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh


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


# 示例用法
if __name__ == "__main__":

    # 创建空图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from([1, 2, 3, 4, 5,6,7,8])  # 支持数字和字符串混合标签

    # 添加边（可自定义权重）
    G.add_edges_from([
        (1,2, {"weight": 1.0}),
        (1, 3),  # 默认权重为1
        (2, 3),
        (2,4),
        (3, 5),
        (4, 5),
        (4,6),
        (5, 6),
        (6,7),
        (6,8)
    ])

    # 验证图结构
    print("节点列表:", G.nodes())
    print("边列表:", G.edges(data=True))  # 显示权重信息





    # 查找关键循环
    key_cycles = find_key_cycles(G)

    # 输出结果
    print("关键循环排序结果（I_C降序）：")
    for i, (cycle, score) in enumerate(key_cycles):
        print(f"第{i + 1}名: 循环{cycle}, 重要性得分: {score:.4f}")