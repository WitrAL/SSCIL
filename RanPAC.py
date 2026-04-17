import copy
import logging
import numpy as np
import os
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from inc_net import ResNetCosineIncrementalNet,SimpleVitNet
from utils.toolkit import tensor2numpy, accuracy
from torch.utils.tensorboard import SummaryWriter

from itertools import cycle

num_workers = 8
dir_path = os.path.dirname(os.path.realpath(__file__))
tensorboard_dir = os.path.join(dir_path, 'runs')
class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments=[]
        self._network = None
        
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

        self.all_prototype = []

        self.writer = SummaryWriter(log_dir=tensorboard_dir)

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        acc_total,grouped = self._evaluate(y_pred, y_true)
        return acc_total,grouped,y_pred[:,0],y_true

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1] 
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  
    
    def _evaluate(self, y_pred, y_true):
        ret = {}
        acc_total,grouped = accuracy(y_pred.T[0], y_true, self._known_classes,self.class_increments)
        return acc_total,grouped 
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]

            predicts = torch.max(outputs, dim=1)[1]

            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args["model_name"]!='ncm':
            if args["model_name"]=='adapter' and '_adapter' not in args["convnet_type"]:
                raise NotImplementedError('Adapter requires Adapter backbone')
            if args["model_name"]=='ssf' and '_ssf' not in args["convnet_type"]:
                raise NotImplementedError('SSF requires SSF backbone')
            if args["model_name"]=='vpt' and '_vpt' not in args["convnet_type"]:
                raise NotImplementedError('VPT requires VPT backbone')

            if 'resnet' in args['convnet_type']:
                self._network = ResNetCosineIncrementalNet(args, True)
                self._batch_size=128
            else:
                self._network = SimpleVitNet(args, True)
                self._batch_size= args["batch_size"]
            
            self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
            self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        else:
            self._network = SimpleVitNet(args, True)
            self._batch_size= args["batch_size"]
        self.args=args

    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def replace_fc(self,train_loader_lb_cp,train_loader_ulb_cp):
        self._network = self._network.eval()

        # =====================================
        # 1) 超参与初始变量
        # =====================================
        filter_threshold = self.args['confidence_threshold']
        threshold_low = 0.3

        lam_mode = "dist"   # 例: 通过距离反比计算混合 α = d2 / (d1 + d2)
        use_similarity_filter = True  # 是否使用相似度过滤
        use_cos_sim = True  # 控制是否使用余弦相似度, True: 使用; False: 使用欧氏距离 【对于中等置信度样本，计算投影后的特征与投影原型距离/相似度用】
        use_logits_assist = True  # 控制是否使用 logits 辅助生成软标签
        beta = 0.95  # 偏向原型距离/相似度的权重 (仅当 use_logits_assist 为 True 时使用)


        #ilter_threshold = 0
        pseudo_correct_total = 0  # 累积正确的伪标签数量
        pseudo_total_total = 0  # 累积伪标签的总数量

        # ======================= 新增：创建图像保存目录 =======================
        
        # 中等置信度图片的保存目录
        # ===================== END OF 新增 =====================


        if self.args['use_RP']:
            #these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.fc.use_RP=True
            if self.args['M']>0:
                self._network.fc.W_rand=self.W_rand
            else:
                self._network.fc.W_rand=None

    
        # =====================================
        # 2) 提取有标签数据特征并计算原型
        # =====================================
        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader_lb_cp):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self._network.convnet(data)
                #embedding size: (batch_size, feature_dim)   (48,768)
                # logging.info('embedding.shape:{}'.format(embedding.shape))
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())

        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        # 记录有标签数据的 One-hot label
        # 计算(投影前)原型, 并保存到 self.prototypes

        self.prototypes = {}  # 使用一个字典来保存原始的原型特征
        unique_labels = torch.unique(label_list)
        for label in unique_labels:
            indices = (label_list == label).nonzero().squeeze(-1)
            prototype = Features_f[indices].mean(0)
            self.prototypes[label.item()] = prototype  # 保存原型特征到字典中

        # 合并到全局的原型字典中
        if not hasattr(self, 'all_prototypes'):
            self.all_prototypes = {}  # 初始化全局原型字典

        # 更新全局字典
        self.all_prototypes.update(self.prototypes)

        if self.args['use_RP']:
            self.prototypes_h = {}
            for lbl in unique_labels:
                proto_f = self.prototypes[lbl.item()]  # 取投影前原型
                if self.args['M'] > 0:
                    # 先要把 proto_f shape=[feat_dim], 扩成 [1, feat_dim] 再 @ W_rand
                    proto_f_h = torch.nn.functional.relu(proto_f.unsqueeze(0) @ self._network.fc.W_rand.to(proto_f.device))
                    proto_f_h = proto_f_h.squeeze(0)  # 回到 [feat_dim_afterRP]
                else:
                    proto_f_h = proto_f
                self.prototypes_h[lbl.item()] = proto_f_h
        else:
            # 不用 RP 就没必要维护 prototypes_h, 直接在原空间计算
            self.prototypes_h = self.prototypes


        # =====================================
        # 3) 处理 无标签数据: 引入 "低-中-高" 置信度区间 + 混合原型
        # =====================================
        proto_keys = sorted(list(self.prototypes.keys()))  # 获取原型字典的键
        proto_tensors_h = torch.stack([self.prototypes_h[k] for k in proto_keys], dim=0).to(device='cuda')

        # -- 用于后面相似度过滤: 先把有标签数据也投影后收集起来 --
        if self.args['use_RP']:
            if self.args['M'] > 0:
                # 投影有标签特征
                Features_h_labeled = torch.nn.functional.relu(
                    Features_f @ self._network.fc.W_rand.to(Features_f.device)
                )
            else:
                Features_h_labeled = Features_f
        else:
            Features_h_labeled = Features_f

        # labeled_feats_tensor 形状 [N_l, feat_dim_afterRP]
        labeled_feats_tensor = Features_h_labeled.to(Features_f.device)
        # labeled_labels_tensor: 整数label（不是one-hot），便于对比
        labeled_labels_tensor = label_list.to(Features_f.device)  # [N_l]

        # 准备收集 "高置信度" 样本特征 + 伪标签
    
        high_conf_feats = []
        high_conf_labels = []

        # 准备收集 "中等置信度" 样本特征(投影后) + soft label
        mid_conf_feats = []
        mid_conf_labels = []

        pseudo_label_list = []
        Features_f_ulb = []
        # --- 初始化用于保存 logit 和真实标签的列表 ---
        


        with torch.no_grad():
            high_conf_count = 0
            mid_conf_count = 0
            total_samples = 0
            for i, batch in enumerate(train_loader_ulb_cp):
                (_, data_ulb, _,real_targets) = batch
                data_ulb = data_ulb.cuda()

                feats_ulb_f = self._network.convnet(data_ulb) 
                # ---- 对特征再投影(若 use_RP & M>0) ----
                if self.args['use_RP']:
                    if self.args['M'] > 0:
                        feats_ulb_h = torch.nn.functional.relu(feats_ulb_f @ self._network.fc.W_rand.to(feats_ulb_f.device))
                    else:
                        feats_ulb_h = feats_ulb_f
                else:
                    feats_ulb_h = feats_ulb_f  # 统一处理

                # ---- 用线性分类器打伪标签 (logits) ----
                logits_ulb = self._network.linear_fc(feats_ulb_f)["logits"]
                probs_ulb  = torch.softmax(logits_ulb, dim=1)
                max_prob, hard_label = probs_ulb.max(dim=1)


                total_samples += data_ulb.size(0)

                # --- 保存 logits 和 real_targets ---


                # ---- 根据置信度做区分 ----

                mask_high = (max_prob > filter_threshold)                # 高置信度
                mask_low  = (max_prob < threshold_low)                   # 低置信度
                mask_mid  = ~(mask_high | mask_low)                      # 剩下的就是中等置信度
                


                # 1) 直接丢弃 低置信度样本
                # 2) 高置信度: 用硬标签
                feats_high = feats_ulb_h[mask_high]      # shape [N_high, dim_afterRP]
                lbl_high   = hard_label[mask_high]       # shape [N_high]
                high_conf_count += mask_high.sum().item()
                
                # --- 相似度过滤 ---
                # 先对 feats_high 归一化，再对 labeled_feats_tensor 归一化
                if feats_high.size(0) > 0:
                    feats_high_norm = torch.nn.functional.normalize(feats_high, p=2, dim=1)
                    labeled_feats_norm = torch.nn.functional.normalize(labeled_feats_tensor, p=2, dim=1)
                    labeled_feats_norm = labeled_feats_norm.to(device='cuda')
                    feats_high_norm = feats_high_norm.to(device='cuda')
                    # [N_high, N_labeled]
                    sims = torch.mm(feats_high_norm, labeled_feats_norm.T)

                    # 取相似度最高的有标签索引
                    max_sims, max_idx = sims.max(dim=1)  # shape [N_high]

                # 计算伪标签准确率统计
                lbl_high=lbl_high.to(real_targets.device)
                mask_high=mask_high.to(real_targets.device)
                pseudo_correct = (lbl_high == real_targets[mask_high]).sum().item()
                pseudo_total   = mask_high.sum().item()
                pseudo_correct_total += pseudo_correct
                pseudo_total_total    += pseudo_total
                # 收集到 high_conf_feats / high_conf_labels
                if use_similarity_filter:
                    for i_h in range(feats_high.size(0)):
                        # 预测标签
                        pred_lbl = lbl_high[i_h].item()
                        # 找到最相似的 labeled 样本:
                        match_index = max_idx[i_h].item()
                        # 该 labeled 样本的 真标签
                        true_lbl = labeled_labels_tensor[match_index].item()

                        if pred_lbl == true_lbl:
                            # 若相同 => 通过相似度过滤 => accept
                            f_ = feats_high[i_h].cpu()
                            l_ = torch.zeros(self.total_classnum, dtype=torch.float)
                            l_[pred_lbl] = 1.0
                            high_conf_feats.append(f_)
                            high_conf_labels.append(l_)
                        else:
                            # 不一致 => 丢弃(不加入高置信度列表)
                            pass
                else:
                    for i_h in range(feats_high.size(0)):
                        f_ = feats_high[i_h].cpu()
                        l_ = torch.zeros(self.total_classnum, dtype=torch.float)
                        l_[lbl_high[i_h].item()] = 1.0
                        high_conf_feats.append(f_)
                        high_conf_labels.append(l_)
               
                # 3) 中等置信度: 用“混合原型” + (可选) logits 生成 软标签
                feats_mid_h = feats_ulb_h[mask_mid]  # shape [N_mid, dim_afterRP]
                logits_mid = logits_ulb[mask_mid] if use_logits_assist else None
                mid_conf_count += mask_mid.sum().item()

                                # ======================= MODIFIED LINE =======================
                # Get the ground truth labels for the medium-confidence samples for logging
                # 将 real_targets 移动到GPU上再进行索引
                # ===================== END OF MODIFICATION =====================
                # ======================= MODIFIED BLOCK =======================
                # Get the ground truth labels for the medium-confidence samples for logging
                # ===================== END OF MODIFIED BLOCK =====================
                # 新增：获取中等置信度样本在当前批次中的索引
                
                if feats_mid_h.size(0) > 0:
                    if use_cos_sim:
                        # --- 余弦相似度 版本 ---
                        feats_mid_h_norm = torch.nn.functional.normalize(feats_mid_h, p=2, dim=1)
                        proto_tensors_h_norm = torch.nn.functional.normalize(proto_tensors_h, p=2, dim=1)
                        sims = torch.mm(feats_mid_h_norm, proto_tensors_h_norm.T)
                        top2_sims, top2_idx = torch.topk(sims, k=2, largest=True, dim=1)

                        if use_logits_assist:
                            top2_logits, top2_logits_idx = torch.topk(logits_mid, k=2, dim=1)

                        for i_m in range(feats_mid_h.size(0)):
                            # --- 原型部分 (总是执行) ---
                            s1, s2 = top2_sims[i_m, 0].item(), top2_sims[i_m, 1].item()
                            c1_idx, c2_idx = top2_idx[i_m, 0].item(), top2_idx[i_m, 1].item()
                            c1_label = proto_keys[c1_idx]  # 原型预测的 top1 类别
                            c2_label = proto_keys[c2_idx]  # 原型预测的 top2 类别
                            # ======================= NEW CODE =======================
                            # 2. Log the required information for this sample.
                            # ===================== END OF 新增 =====================
                            if lam_mode == "dist":
                                alpha_proto = float(s1 / (s1 + s2 + 1e-9))
                            else:
                                alpha_proto = 0.5

                            if use_logits_assist:
                                # --- Logits 部分 (仅当 use_logits_assist 为 True 时执行) ---
                                l1, l2 = top2_logits[i_m, 0].item(), top2_logits[i_m, 1].item()
                                c1_label_logits = top2_logits_idx[i_m, 0].item()  # logits 预测的 top1 类别
                                c2_label_logits = top2_logits_idx[i_m, 1].item()  # logits 预测的 top2 类别
                                logits_softmax = torch.nn.functional.softmax(top2_logits[i_m], dim=0)
                                alpha_logits = float(logits_softmax[0] / (logits_softmax[0] + logits_softmax[1] + 1e-9))
                                alpha_final = beta * alpha_proto + (1 - beta) * alpha_logits

                                # --- 构建 y_ (根据原型和 logits 预测的重合程度) ---
                                y_ = torch.zeros(self.total_classnum, dtype=torch.float)

                                # 两个 top 类别完全一致（最好情况）
                                if c1_label == c1_label_logits and c2_label == c2_label_logits:
                                    y_[c1_label] = alpha_final
                                    y_[c2_label] = 1 - alpha_final

                                # 部分交集 (交集一个类别)
                                elif c1_label == c1_label_logits:
                                    y_[c1_label] = alpha_final
                                    y_[c2_label] = beta * (1 - alpha_proto)  # 原型 top2
                                    y_[c2_label_logits] = (1 - beta) * (1 - alpha_logits)  # logits top2

                                elif c1_label == c2_label_logits:
                                    y_[c1_label] = alpha_final
                                    y_[c2_label] = beta * (1 - alpha_proto)  # 原型 top2
                                    y_[c1_label_logits] = (1 - beta) * alpha_logits # logits top1

                                elif c2_label == c1_label_logits:
                                    y_[c2_label] = 1-alpha_final
                                    y_[c1_label] = beta * alpha_proto       # 原型 top1
                                    y_[c2_label_logits] = (1 - beta) * (1-alpha_logits) # logits top2

                                elif c2_label == c2_label_logits:
                                    y_[c2_label] = 1-alpha_final
                                    y_[c1_label] = beta * alpha_proto        # 原型 top1
                                    y_[c1_label_logits] = (1 - beta) * alpha_logits  # logits top1
                                
                                # 完全无交集情况（原型 top1 类别 和 logits top1 类别 分别赋予权重）
                                else:
                                    y_[c1_label] = beta * alpha_proto  #更倾向于相信原型
                                    y_[c2_label] = beta * (1-alpha_proto)
                                    y_[c1_label_logits] = (1 - beta) * alpha_logits
                                    y_[c2_label_logits] = (1 - beta) * (1-alpha_logits)

                            else:  # use_logits_assist == False
                                # 只使用原型信息
                                y_ = torch.zeros(self.total_classnum, dtype=torch.float)
                                y_[c1_label] = alpha_proto
                                y_[c2_label] = 1 - alpha_proto

                            mid_conf_feats.append(feats_mid_h[i_m].cpu())
                            mid_conf_labels.append(y_)

                    else:  # 欧氏距离版本 (与 use_cos_sim 版本类似，也加入 use_logits_assist 控制)
                        dists = torch.cdist(feats_mid_h, proto_tensors_h, p=2)
                        top2_dists, top2_idx = torch.topk(dists, k=2, largest=False, dim=1)

                        if use_logits_assist:
                            top2_logits, top2_logits_idx = torch.topk(logits_mid, k=2, dim=1)

                        for i_m in range(feats_mid_h.size(0)):
                            # --- 原型部分 ---
                            d1, d2 = top2_dists[i_m, 0].item(), top2_dists[i_m, 1].item()
                            c1_idx, c2_idx = top2_idx[i_m, 0].item(), top2_idx[i_m, 1].item()
                            c1_label = proto_keys[c1_idx]  # 原型 top1
                            c2_label = proto_keys[c2_idx]  # 原型 top2

                            if lam_mode == "dist":
                                alpha_proto = float(d2 / (d1 + d2 + 1e-9))
                            else:
                                alpha_proto = 0.5

                            if use_logits_assist:
                                # --- Logits 部分 ---
                                l1, l2 = top2_logits[i_m, 0].item(), top2_logits[i_m, 1].item()
                                c1_label_logits = top2_logits_idx[i_m, 0].item()  # logits top1
                                c2_label_logits = top2_logits_idx[i_m, 1].item()  # logits top2
                                logits_softmax = torch.nn.functional.softmax(top2_logits[i_m], dim=0)
                                alpha_logits = float(logits_softmax[0] / (logits_softmax[0] + logits_softmax[1] + 1e-9))
                                alpha_final = beta * alpha_proto + (1 - beta) * alpha_logits

                                # --- 构建 y_ (与余弦相似度版本完全一致) ---
                                y_ = torch.zeros(self.total_classnum, dtype=torch.float)

                                # 两个 top 类别完全一致（最好情况）
                                if c1_label == c1_label_logits and c2_label == c2_label_logits:
                                    y_[c1_label] = alpha_final
                                    y_[c2_label] = 1 - alpha_final

                                # 部分交集 (交集一个类别)
                                elif c1_label == c1_label_logits:
                                    y_[c1_label] = alpha_final
                                    y_[c2_label] = beta * (1 - alpha_proto)
                                    y_[c2_label_logits] = (1 - beta) * (1 - alpha_logits)

                                elif c1_label == c2_label_logits:
                                    y_[c1_label] = alpha_final
                                    y_[c2_label] = beta * (1 - alpha_proto)
                                    y_[c1_label_logits] = (1 - beta) * alpha_logits

                                elif c2_label == c1_label_logits:
                                    y_[c2_label] = 1 - alpha_final
                                    y_[c1_label] = beta * alpha_proto
                                    y_[c2_label_logits] = (1 - beta) * (1 - alpha_logits)

                                elif c2_label == c2_label_logits:
                                    y_[c2_label] = 1 - alpha_final
                                    y_[c1_label] = beta * alpha_proto
                                    y_[c1_label_logits] = (1 - beta) * alpha_logits

                                # 完全无交集
                                else:
                                    y_[c1_label] = beta * alpha_proto  #更倾向于相信原型
                                    y_[c2_label] = beta * (1-alpha_proto)
                                    y_[c1_label_logits] = (1 - beta) * alpha_logits
                                    y_[c2_label_logits] = (1 - beta) * (1-alpha_logits)

                            else:  # use_logits_assist == False
                                y_ = torch.zeros(self.total_classnum, dtype=torch.float)
                                y_[c1_label] = alpha_proto
                                y_[c2_label] = 1 - alpha_proto

                            mid_conf_feats.append(feats_mid_h[i_m].cpu())
                            mid_conf_labels.append(y_)
                        
        # --- 在训练结束后，打印高、中置信度样本的数量和占比 ---
        logging.info(f"High-confidence samples: {high_conf_count} / {total_samples} = {high_conf_count/total_samples:.4f}")
        logging.info(f"Mid-confidence samples: {mid_conf_count} / {total_samples} = {mid_conf_count/total_samples:.4f}")


        # ===================================== 
        # 4) 准备做 Ridge 回归 (合并有标签 + 无标签)
        # =====================================
        # - 有标签(投影后) + [高/中置信度无标签(软/硬)]
        # 先把 labeled feats 也投影 (若 M>0)
        # if self.args['use_RP']:
        #     if self.args['M'] > 0:
        #         Features_h_labeled = torch.nn.functional.relu(Features_f @ self._network.fc.W_rand.to(Features_f.device))
        #     else:
        #         Features_h_labeled = Features_f
        # else:
        #     Features_h_labeled = Features_f  # 不投影

        # 再把 labeled feats/labels 拼成 list
        labeled_feats_list = []
        labeled_labels_list = []
        for i_l in range(Features_h_labeled.size(0)):
            f_ = Features_h_labeled[i_l].cpu()
            lbl_ = torch.zeros(self.total_classnum, dtype=torch.float)
            lbl_[label_list[i_l].item()] = 1.0
            labeled_feats_list.append(f_)
            labeled_labels_list.append(lbl_)
        # 合并
        combined_feats = labeled_feats_list + high_conf_feats + mid_conf_feats
        combined_labels = labeled_labels_list + high_conf_labels + mid_conf_labels
        if len(combined_feats) == 0:
            logging.info("No data to update ridge. skip.")
            return
        X_combined = torch.stack(combined_feats, dim=0).to(device='cuda')    # [N_all, feat_dim_afterRP]
        Y_combined = torch.stack(combined_labels, dim=0).to(device='cuda')  # [N_all, total_classnum]


        if self.args['use_RP']:

            # 确保输入张量在正确的设备上
            X_combined = X_combined.to(device='cpu')
            Y_combined = Y_combined.to(device='cpu')
            tmp_3 = X_combined.T @ Y_combined
            tmp_4 = X_combined.T @ X_combined

            self.Q = self.Q + tmp_3 
            self.G = self.G + tmp_4 
            ridge = self.optimise_ridge_parameter(X_combined, Y_combined)
            Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q).T
            self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0], :].to(device='cuda')


        else:
            Features_f_combined = torch.cat([Features_f, Features_f_ulb], dim=0)
            pseudo_label_list = torch.cat([label_list, pseudo_label_list], dim=0)
            for class_index in np.unique(self.train_dataset_lb.labels):
                data_index = (pseudo_label_list == class_index).nonzero().squeeze(-1)
                class_prototype = Features_f_combined[data_index].mean(0)
                self._network.fc.weight.data[class_index] = class_prototype

    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(3,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge
    
    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(self._cur_task)
        if self.args['use_RP']:
            #temporarily remove RP weights
            del self._network.fc
            self._network.fc=None
        self._network.update_fc(self._classes_seen_so_far) #creates a new head with a new number of classes (if CIL)
        if self.is_dil == False:
            logging.info("Starting CIL Task {}".format(self._cur_task+1))
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._classes_seen_so_far-1))

        if self._cur_task > 0:
            self.teacher_model = copy.deepcopy(self._network)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        self.class_increments.append([self._known_classes, self._classes_seen_so_far-1])
        self.train_dataset_lb = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train_lb", mode="train_lb", )
        self.train_loader_lb = DataLoader(self.train_dataset_lb, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)

        self.train_dataset_ulb = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train_ulb", mode="train_ulb", )
        self.train_loader_ulb = DataLoader(self.train_dataset_ulb, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)

        train_dataset_lb_cp = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train_lb", mode="test", )
        self.train_loader_lb_cp = DataLoader(train_dataset_lb_cp, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)

        train_dataset_ulb_cp = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train_ulb", mode="test", )
        self.train_loader_ulb_cp = DataLoader(train_dataset_ulb_cp, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)

        test_dataset_this_task = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far), source="test", mode="test" )
        test_dataset = data_manager.get_dataset(np.arange(0, self._classes_seen_so_far), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader_this_task = DataLoader(test_dataset_this_task, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader_lb,self.train_loader_ulb, self.test_loader, self.train_loader_lb_cp,self.train_loader_ulb_cp,self.test_loader_this_task)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def freeze_backbone(self,is_first_session=False):
        # Freeze the parameters for ViT.
        if 'vit' in self.args['convnet_type']:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False
        else:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False

    def show_num_params(self,verbose=False):
        # show total parameters and trainable parameters
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters.')
        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

#只动了第二个分支
    def _train(self, train_loader_lb,train_loader_ulb, test_loader, train_loader_lb_cp,train_loader_ulb_cp,test_loader_this_task):
        self._network.to(self._device)
        if self._cur_task == 0 and self.args["model_name"] in ['ncm','joint_linear']:
             self.freeze_backbone()
        if self.args["model_name"] in ['joint_linear','joint_full']: 
            #this branch updates using SGD on all tasks and should be using classes and does not use a RP head
            if self.args["model_name"] =='joint_linear':
                assert self.args['body_lr']==0.0
            self.show_num_params()
            optimizer = optim.SGD([{'params':self._network.convnet.parameters()},{'params':self._network.fc.parameters(),'lr':self.args['head_lr']}], 
                                        momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000])
            logging.info("Starting joint training on all data using "+self.args["model_name"]+" method")
            self._init_train(train_loader_lb,train_loader_ulb,test_loader, optimizer, scheduler) 
            self.show_num_params()
        else:
            #this branch is either CP updates only, or SGD on a PETL method first task only
            #使用了first adaption，只在第一个任务上使用petl进行微调，然后冻结backbone
            if self._cur_task == 0 and self.dil_init==False:
                if 'ssf' in self.args['convnet_type']:
                    self.freeze_backbone(is_first_session=True)
                if self.args["model_name"] != 'ncm':
                    #this will be a PETL method. Here, 'body_lr' means all parameters
                    self.show_num_params()
                    #optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
                    #scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                    #train the PETL method for the first task:
                    logging.info("Starting PETL training on first task using "+self.args["model_name"]+" method")
                    if self.args['load_model'] == False:
                        #self._init_train(train_loader_lb,train_loader_ulb, test_loader, optimizer, scheduler)
                        #self.freeze_fc_layer(self._network)
                        self._continual_peft(train_loader_lb,train_loader_ulb, test_loader_this_task)
                        #self.unfreeze_fc_layer(self._network)
                    else:
                        model_path = self.args.get('model_path')
                        if not model_path or (isinstance(model_path, float) and np.isnan(model_path)):
                            raise ValueError("load_model=True requires args['model_path']")
                        self.load_model(model_path)
                    #self.freeze_backbone()
                if self.args['use_RP'] and self.dil_init==False:
                    self.setup_RP()
            if self.is_dil and self.dil_init==False:
                self.dil_init=True
                self._network.fc.weight.data.fill_(0.0)
            if self._cur_task!=0 and self.args['keep_peft']:
                logging.info("Starting PETL training on "+str(self._cur_task)+" task using "+self.args["model_name"]+" method")
                
                # optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
                # if self.args['dataset']=='cub':
                #     optimizer = torch.optim.AdamW(self._network.parameters(), lr=1e-2)
                # scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                
                #self.freeze_fc_layer(self._network)             
                self._continual_peft(train_loader_lb,train_loader_ulb, test_loader_this_task)
                #self.unfreeze_fc_layer(self._network)
                #self.freeze_backbone(is_first_session=True)
            self.replace_fc(train_loader_lb_cp,train_loader_ulb_cp)
            self.show_num_params()
        
    def load_model(self, path):
    # Load model weights from a file
        self._network.load_state_dict(torch.load(path))

    def save_model(self, path):
    # Save model weights to a file
        torch.save(self._network.state_dict(), path)

    def setup_RP(self):
        self.initiated_G=False
        self._network.fc.use_RP=True
        if self.args['M']>0:
            #RP with M > 0
            M=self.args['M']
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(device='cuda')) #num classes in task x M
            self._network.fc.reset_parameters()
            self._network.fc.W_rand=torch.randn(self._network.fc.in_features,M).to(device='cuda')
            self.W_rand=copy.deepcopy(self._network.fc.W_rand) #make a copy that gets passed each time the head is replaced
        else:
            #no RP, only decorrelation
            M=self._network.fc.in_features #this M is L in the paper
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)

    def _init_train(self, train_loader_lb,train_loader_ulb, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        #threshold = self.args['confidence_threshold']  # FixMatch伪标签的置信度阈值
        threshold=0.75
        lambda_u = 1.0
        #logging.info("threshold: {}, lambda_u: {}".format(threshold, lambda_u))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            consistency_losses = 0.0
            correct, total = 0, 0
            #有标签部分
            for i, (_, inputs_lb, targets) in enumerate(train_loader_lb):
                inputs_lb, targets = inputs_lb.to(self._device), targets.to(self._device)
                logits = self._network(inputs_lb)["logits"]
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            for i, (_,inputs_w,inputs_s,real_targets) in enumerate(train_loader_ulb):
                inputs_w, inputs_s = inputs_w.to(self._device), inputs_s.to(self._device)

                with torch.no_grad():
                    logits_w = self._network(inputs_w)["logits"]
                    probs_w = torch.softmax(logits_w, dim=1)
                    max_probs,pseudo_labels=torch.max(probs_w,1)

                    mask = max_probs.ge(threshold).float()
                
                logits_s = self._network(inputs_s)["logits"]
                loss_consistency = (F.cross_entropy(logits_s, pseudo_labels, reduction='none') * mask).mean()  # 仅对高置信度伪标签计算损失
                optimizer.zero_grad()
                loss_consistency.backward()
                optimizer.step()

                consistency_losses += loss_consistency.item()

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.5f}, consistency Losses {:.5f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader_lb),
                consistency_losses / len(train_loader_ulb),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _continual_peft(self, train_loader_lb, train_loader_ulb, test_loader):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        threshold = self.args['confidence_threshold']
        lambda_u = 1.0
        lambda_distill = 1.0
        lambda_proto = 0
        lambda_transfer = 1.0
        distill_temp = 1.0

        previous_lora_state = save_lora_state(self._network) if self._cur_task > 0 else None
        param_groups = []
        grouped_params = self.get_grouped_params()

        # if self._cur_task > 0:
        #     # 重置LoRA参数到初始状态


        # 分类器可能需要较大的学习率来快速适应
        param_groups.append({
            'params': grouped_params['other'],
            'lr': self.args['body_lr'],  # 使用单独的分类器学习率
            'weight_decay': self.weight_decay
        })
        
        # LoRA参数使用相对较小的学习率以确保稳定性
        param_groups.append({
            'params': grouped_params['lora'],
            'lr': 0.001,  # 或者使用更小的学习率
            'weight_decay': 0 # LoRA通常不需要权重衰减
        })
        
    
        optimizer = optim.SGD(param_groups, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.args['tuned_epoch'], 
            eta_min=self.min_lr
        )
        # --- 确定迭代次数 (仍在 epoch 循环外，因为 loader 长度不变) ---
        if train_loader_ulb is not None and len(train_loader_ulb) > 0:
            num_iter = len(train_loader_ulb)
            base_on_ulb = True
            logging.info(f"Epoch iterations based on unlabeled loader: {num_iter}")
        elif train_loader_lb is not None and len(train_loader_lb) > 0:
            num_iter = len(train_loader_lb)
            base_on_ulb = False
            logging.info(f"Epoch iterations based on labeled loader: {num_iter}")
            lambda_u = 0 # 没有无标签数据，权重设为0
        else:
            logging.warning("Both labeled and unlabeled loaders are empty or None. Skipping training.")
            return
        # 在创建优化器后调用

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            cls_losses = 0.0
            consistency_losses = 0.0
            distill_losses = 0.0
            proto_losses = 0.0
            transfer_losses = 0.0
            total_loss = 0.0
            correct, total = 0, 0
            pseudo_correct_total = 0
            pseudo_total_total = 0

            # --- 修改：在每个 epoch 开始时创建迭代器 ---
            if base_on_ulb:
                lb_iter = cycle(train_loader_lb)
                ulb_iter = iter(train_loader_ulb)
            else: # base_on_lb
                lb_iter = iter(train_loader_lb)
                ulb_iter = None # 明确设为 None
            # --- 迭代器创建结束 ---
            #for i, ((_, inputs_lb, targets), (_, inputs_w, inputs_s,real_targets)) in enumerate(zip(train_loader_lb, train_loader_ulb)):
            for i in range(num_iter): # 使用确定好的迭代次数
                # --- 获取有标签数据 ---
                # 假设 train_loader_lb 返回: _, inputs_lb, targets
                _, inputs_lb, targets = next(lb_iter) # 使用新创建的 lb_iter
                inputs_lb, targets = inputs_lb.to(self._device), targets.to(self._device)

                # --- 获取无标签数据 ---
                inputs_w, inputs_s, real_targets = None, None, None
                if ulb_iter is not None:
                    try:
                        # 假设 train_loader_ulb 返回: _, inputs_w, inputs_s, real_targets
                        _, inputs_w, inputs_s, real_targets = next(ulb_iter)
                        inputs_w, inputs_s = inputs_w.to(self._device), inputs_s.to(self._device)
                        # real_targets 用于计算伪标签准确率，需要移到设备
                        real_targets = real_targets.to(self._device)
                    except StopIteration:
                        # 理论上不应发生，因为 num_iter 基于 ulb 长度
                        logging.warning("Unlabeled data loader finished unexpectedly within epoch.")
                        continue # 跳过此迭代
                # 处理有标签的数据
                #inputs_lb, targets = inputs_lb.to(self._device), targets.to(self._device)

                # 获取卷积特征
                features = self._network.convnet(inputs_lb)
                
                # 使用线性分类器进行前向传播
                logits_dict = self._network.linear_fc(features)
                logits = logits_dict['logits']
                cls_loss = F.cross_entropy(logits, targets)
                losses = cls_loss
                cls_losses += cls_loss.item()

                #计算知识蒸馏损失（如果不是第一阶段）
                if self._cur_task > 0:
                    with torch.no_grad():
                        teacher_features = self.teacher_model.convnet(inputs_lb)
                        teacher_logits = self.teacher_model.linear_fc(teacher_features)['logits']
                    distill_loss = F.kl_div(
                        F.log_softmax(logits / distill_temp, dim=1),
                        F.softmax(teacher_logits / distill_temp, dim=1),
                        reduction='batchmean'
                    ) * (distill_temp ** 2)
                    losses += lambda_distill * distill_loss
                    distill_losses += distill_loss.item()

                    # 计算原型损失
                    proto_loss = 0.0
                    for label, proto_features_raw in self.prototypes.items():
                        proto_features = proto_features_raw.to(self._device)
                        proto_features_transfer = self._network.transfer(proto_features.unsqueeze(0))
                        proto_features_tensor = proto_features_transfer['logits']  # 从字典中提取张量
                        # 使用线性分类器得到分类输出
                        proto_logits_dict = self._network.linear_fc(proto_features_tensor)
                        proto_logits = proto_logits_dict['logits']
                        
                        # 构造原型的目标标签
                        proto_targets = torch.tensor([label], dtype=torch.long).to(self._device)

                        # 计算原型损失
                        proto_loss += F.cross_entropy(proto_logits / distill_temp, proto_targets)
                    proto_losses += proto_loss.item()
                    losses += lambda_proto * proto_loss

                    # 计算特征转换损失
                    features_old = self.teacher_model.convnet(inputs_lb)
                    feature_transfer_dict = self._network.transfer(features_old)
                    feature_transfer = feature_transfer_dict['logits']
                    l2_loss = F.mse_loss(features, feature_transfer)
                    loss_transfer = lambda_transfer * l2_loss

                    transfer_losses += loss_transfer.item()
                    losses += loss_transfer

                # 处理无标签的数据
                inputs_w, inputs_s = inputs_w.to(self._device), inputs_s.to(self._device)
                with torch.no_grad():
                    features_w = self._network.convnet(inputs_w)
                    logits_w_dict = self._network.linear_fc(features_w)
                    logits_w= logits_w_dict['logits']
                    probs_w = torch.softmax(logits_w, dim=1)
                    max_probs, pseudo_labels = torch.max(probs_w, 1)
                    mask = max_probs.ge(threshold).float()

                features_s = self._network.convnet(inputs_s)
                logits_s_dict = self._network.linear_fc(features_s)
                logits_s = logits_s_dict['logits']
                loss_consistency = (F.cross_entropy(logits_s, pseudo_labels, reduction='none') * mask).mean()
                # 计算伪标签的准确率
                pseudo_correct = (pseudo_labels.to(self._device) == real_targets.to(self._device)).sum().item()  # 将real_targets移动到与pseudo_labels相同的设备

                pseudo_total = len(real_targets)
                pseudo_accuracy = 100.0 * pseudo_correct / pseudo_total if pseudo_total > 0 else 0.0

                # 累加伪标签的正确预测和总样本数
                pseudo_correct_total += pseudo_correct
                pseudo_total_total += pseudo_total

                losses += lambda_u * loss_consistency
                consistency_losses += loss_consistency.item()


                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += losses.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)

            if pseudo_total_total > 0:
                average_pseudo_accuracy = 100.0 * pseudo_correct_total / pseudo_total_total
            else:
                average_pseudo_accuracy = 0.0



            # 使用任务编号记录 SummaryWriter
            task_id = f"Task_{self._cur_task}"
            self.writer.add_scalar(f"{task_id}/Loss/Total", total_loss / len(train_loader_lb), epoch)
            self.writer.add_scalar(f"{task_id}/Loss/Classification", cls_losses / len(train_loader_lb), epoch)
            self.writer.add_scalar(f"{task_id}/Loss/Consistency", consistency_losses / len(train_loader_ulb), epoch)
            self.writer.add_scalar(f"{task_id}/Loss/Distillation", distill_losses / len(train_loader_lb), epoch)
            self.writer.add_scalar(f"{task_id}/Loss/Proto", proto_losses / len(train_loader_lb), epoch)
            self.writer.add_scalar(f"{task_id}/Loss/Transfer", transfer_losses / len(train_loader_lb), epoch)
            self.writer.add_scalar(f"{task_id}/Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar(f"{task_id}/Accuracy/Test", test_acc, epoch)
            self.writer.add_scalar(f"{task_id}/Accuracy/Pseudo-label_acc_PEFT", average_pseudo_accuracy, epoch)  # 记录平均伪标签准确率

            info = "Task {}, Epoch {}/{} => Loss {:.5f}, consistency Losses {:.5f}, Distill Losses {:.5f}, Proto Losses {:.5f}, Transfer Losses {:.5f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                total_loss / len(train_loader_lb),
                consistency_losses / len(train_loader_ulb),
                distill_losses / len(train_loader_lb),
                proto_losses / len(train_loader_lb),
                transfer_losses / len(train_loader_lb),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
        
        logging.info("%s/Accuracy/Pseudo-label_acc_PEFT: %.2f", task_id, average_pseudo_accuracy)

        self.current_lora_state = save_lora_state(self._network)
        # 合并当前session的LoRA状态和上一个session的LoRA状态
        if previous_lora_state is not None:
            merge_ratio = self.args.get('lora_merge_ratio', 0.5)
            try:
                merge_ratio = float(merge_ratio)
            except (TypeError, ValueError):
                merge_ratio = 0.5
            if np.isnan(merge_ratio):
                merge_ratio = 0.5
            merge_lora_states(self._network, self.current_lora_state, previous_lora_state, merge_ratio=merge_ratio)

    def get_grouped_params(self):
        """重新组织参数分组"""
        lora_params = []
        other_params = []  # 分类层参数
        
        for name, param in self._network.named_parameters():
            if not param.requires_grad:  # 跳过冻结参数
                continue
            if 'lora_' in name:
                lora_params.append(param)
            else:  # 应该主要是分类层参数
                other_params.append(param)
                
        return {
            'lora': lora_params,
            'other': other_params
        }
    

from petl.vision_transformer_peft import LoRALinear, LoHALinear, LoKRLinear
def save_lora_state(network):
    """保存当前LoRA参数状态"""
    state = {}
    for name, module in network.named_modules():
        if isinstance(module, (LoRALinear, LoHALinear, LoKRLinear)):
            # LoRA参数
            if hasattr(module, 'lora_A'):
                state[f"{name}.lora_A"] = module.lora_A.data.clone()
                state[f"{name}.lora_B"] = module.lora_B.data.clone()
            # LoHA参数
            elif hasattr(module, 'lora_w1_a'):
                state[f"{name}.lora_w1_a"] = module.lora_w1_a.data.clone()
                state[f"{name}.lora_w1_b"] = module.lora_w1_b.data.clone()
                state[f"{name}.lora_w2_a"] = module.lora_w2_a.data.clone()
                state[f"{name}.lora_w2_b"] = module.lora_w2_b.data.clone()
            # LoKR参数
            elif hasattr(module, 'lokr_w1'):
                if module.use_w1:
                    state[f"{name}.lokr_w1"] = module.lokr_w1.data.clone()
                else:
                    state[f"{name}.lokr_w1_a"] = module.lokr_w1_a.data.clone()
                    state[f"{name}.lokr_w1_b"] = module.lokr_w1_b.data.clone()
                if module.use_w2:
                    state[f"{name}.lokr_w2"] = module.lokr_w2.data.clone()
                else:
                    state[f"{name}.lokr_w2_a"] = module.lokr_w2_a.data.clone()
                    state[f"{name}.lokr_w2_b"] = module.lokr_w2_b.data.clone()
    return state

def merge_lora_states(network, current_state, previous_state, merge_ratio=0.5):
    """合并两个session的LoRA状态"""
    if previous_state is None:
        return
        
    with torch.no_grad():
        for name, module in network.named_modules():
            if isinstance(module, (LoRALinear, LoHALinear, LoKRLinear)):
                if hasattr(module, 'lora_A'):
                    module.lora_A.data = merge_ratio * current_state[f"{name}.lora_A"] + \
                                       (1 - merge_ratio) * previous_state[f"{name}.lora_A"]
                    module.lora_B.data = merge_ratio * current_state[f"{name}.lora_B"] + \
                                       (1 - merge_ratio) * previous_state[f"{name}.lora_B"]
                elif hasattr(module, 'lora_w1_a'):
                    module.lora_w1_a.data = merge_ratio * current_state[f"{name}.lora_w1_a"] + \
                                          (1 - merge_ratio) * previous_state[f"{name}.lora_w1_a"]
                    module.lora_w1_b.data = merge_ratio * current_state[f"{name}.lora_w1_b"] + \
                                          (1 - merge_ratio) * previous_state[f"{name}.lora_w1_b"]
                    module.lora_w2_a.data = merge_ratio * current_state[f"{name}.lora_w2_a"] + \
                                          (1 - merge_ratio) * previous_state[f"{name}.lora_w2_a"]
                    module.lora_w2_b.data = merge_ratio * current_state[f"{name}.lora_w2_b"] + \
                                          (1 - merge_ratio) * previous_state[f"{name}.lora_w2_b"]
                elif hasattr(module, 'lokr_w1'):
                    if module.use_w1:
                        module.lokr_w1.data = merge_ratio * current_state[f"{name}.lokr_w1"] + \
                                            (1 - merge_ratio) * previous_state[f"{name}.lokr_w1"]
                    else:
                        module.lokr_w1_a.data = merge_ratio * current_state[f"{name}.lokr_w1_a"] + \
                                              (1 - merge_ratio) * previous_state[f"{name}.lokr_w1_a"]
                        module.lokr_w1_b.data = merge_ratio * current_state[f"{name}.lokr_w1_b"] + \
                                              (1 - merge_ratio) * previous_state[f"{name}.lokr_w1_b"]
                    if module.use_w2:
                        module.lokr_w2.data = merge_ratio * current_state[f"{name}.lokr_w2"] + \
                                            (1 - merge_ratio) * previous_state[f"{name}.lokr_w2"]
                    else:
                        module.lokr_w2_a.data = merge_ratio * current_state[f"{name}.lokr_w2_a"] + \
                                              (1 - merge_ratio) * previous_state[f"{name}.lokr_w2_a"]
                        module.lokr_w2_b.data = merge_ratio * current_state[f"{name}.lokr_w2_b"] + \
                                              (1 - merge_ratio) * previous_state[f"{name}.lokr_w2_b"]

