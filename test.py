import torch

# 示例：模拟特征和原型数据
# 假设我们有 5 个中等置信度的无标签样本的特征 (feats_mid_h)
N_mid = 5
dim_feature = 64  # 特征维度，例如 dim_afterRP = 64
feats_mid_h = torch.randn(N_mid, dim_feature)
print("中等置信度样本特征 feats_mid_h 的形状:", feats_mid_h.shape) # torch.Size([5, 64])

# 假设我们有 3 个已知类别的原型 (proto_tensors_h)
C_known = 3
proto_tensors_h = torch.randn(C_known, dim_feature)
print("原型张量 proto_tensors_h 的形状:", proto_tensors_h.shape) # torch.Size([3, 64])

# 假设 proto_keys 存储了原型对应的类别标签
proto_keys = ["Class_A", "Class_B", "Class_C"]
print("原型类别标签 proto_keys:", proto_keys)

# 存储结果的列表
mid_conf_feats = []
mid_conf_labels = []

# 模拟 lam_mode 和 delta (在您的代码中，delta 和距离差的判断被注释掉了，lam_mode 默认为 "dist" 或其他)
lam_mode = "dist" # 假设 lam_mode 为 "dist"
delta = 1.0 # 假设 delta 阈值为 1.0 (尽管在您的代码中未使用)

if feats_mid_h.size(0) > 0: # 确保有中等置信度样本
    # 计算中等置信度样本与原型之间的欧氏距离 (L2 距离)
    dists = torch.cdist(feats_mid_h, proto_tensors_h, p=2)
    print("距离矩阵 dists 的形状:", dists.shape) # torch.Size([5, 3])
    print("距离矩阵 dists (部分):\n", dists[:,:]) # 打印前两行

    # 找到每个样本距离最近的两个原型及其距离和索引
    top2_dists, top2_idx = torch.topk(dists, k=2, largest=False, dim=1)
    print("最近的两个距离 top2_dists 的形状:", top2_dists.shape) # torch.Size([5, 2])
    print("最近的两个距离 top2_dists (部分):\n", top2_dists[:,:])
    print("最近的两个原型的索引 top2_idx 的形状:", top2_idx.shape) # torch.Size([5, 2])
    print("最近的两个原型的索引 top2_idx (部分):\n", top2_idx[:,:])

    for i_m in range(feats_mid_h.size(0)): # 遍历每个中等置信度样本
        d1, d2 = top2_dists[i_m, 0].item(), top2_dists[i_m, 1].item()
        c1_idx, c2_idx = top2_idx[i_m, 0].item(), top2_idx[i_m, 1].item()
        c1_label = proto_keys[c1_idx]
        c2_label = proto_keys[c2_idx]

        print(f"\n样本 {i_m}:")
        print(f"  最近原型索引 c1_idx: {c1_idx}, 类别标签 c1_label: {c1_label}, 距离 d1: {d1:.4f}")
        print(f"  次近原型索引 c2_idx: {c2_idx}, 类别标签 c2_label: {c2_label}, 距离 d2: {d2:.4f}")

        # --- 生成软标签的代码 (与您提供的代码片段相同) ---
        if False: # 您的代码中条件始终为 False，所以这里也设为 False
            y_ = torch.zeros(len(proto_keys), dtype=torch.float) # 使用 proto_keys 的长度，假设 proto_keys 包含所有已知类别
            y_[c1_idx] = 1.0 # 硬标签 - 最近的原型
            print("  生成硬标签 y_:", y_)
        else:
            if lam_mode == "dist":
                alpha = float(d2 / (d1 + d2 + 1e-9))
            else:
                alpha = 0.5
            y_ = torch.zeros(len(proto_keys), dtype=torch.float) # 使用 proto_keys 的长度
            y_[c1_idx] = alpha
            y_[c2_idx] = 1 - alpha
            print("  生成软标签 y_:", y_)

        mid_conf_feats.append(feats_mid_h[i_m].cpu())
        mid_conf_labels.append(y_)

print("\n收集到的中等置信度样本特征数量:", len(mid_conf_feats))
print("收集到的中等置信度样本软标签数量:", len(mid_conf_labels))