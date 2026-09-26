import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class SPGG(nn.Module):
    def __init__(self, L_num, device, r, epoches, 
                    now_time, question, output_path, punishment_strength=0.0):
        super().__init__()
        self.L_num = L_num
        self.device = device
        self.r = r
        self.epoches = epoches
        self.question = question
        self.now_time = now_time
        self.output_path = output_path
        self.punishment_strength = punishment_strength  # 新增惩罚强度参数
        
        # 邻域卷积核
        self.neibor_kernel = torch.tensor(
            [[[[0,1,0], [1,1,1], [0,1,0]]]], 
            dtype=torch.float32, device=device
        )
        
        # 初始化状态
        self.initial_state = self._init_state(question)
        self.current_state = self.initial_state.clone()

    def _init_state(self, question):
        if question == 1:
            state = torch.bernoulli(torch.full((self.L_num, self.L_num), 0.5))
        elif question == 2:
            state = torch.zeros(self.L_num, self.L_num)
            state[self.L_num//2:, :] = 1
        elif question == 3:
            state = torch.zeros(self.L_num, self.L_num)
        elif question == 4:
            state = torch.zeros((self.L_num, self.L_num))
            for i in range(self.L_num):
                for j in range(self.L_num):
                    if (i + j) % 2 == 0:
                        state[i, j] = 1
        return state.to(self.device)

    def encode_state(self, state_matrix):
        """将 2D 网格转换为 4D 张量后填充"""
        state_4d = state_matrix.float().unsqueeze(0).unsqueeze(0)
        padded = F.pad(state_4d, (1, 1, 1, 1), mode='circular')
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()
        global_coop = torch.mean(state_matrix.float())
        return torch.stack([
            state_matrix.float().squeeze(),
            neighbor_coop,
            global_coop.expand_as(state_matrix)
        ], dim=-1).view(-1, 3)

    def calculate_reward(self, state_matrix):
        """计算基础收益（无惩罚）"""
        float_state = state_matrix.float().unsqueeze(0).unsqueeze(0)
        padded_state = F.pad(float_state, (1, 1, 1, 1), mode='circular')
        N_C_g = F.conv2d(padded_state, self.neibor_kernel)
        group_gross_profit = (self.r * N_C_g) / 5.0
        padded_gross = F.pad(group_gross_profit, (1, 1, 1, 1), mode='circular')
        total_gross_profit = F.conv2d(padded_gross, self.neibor_kernel).squeeze()
        total_cost = state_matrix.float() * 5.0
        reward_matrix = total_gross_profit - total_cost
        return reward_matrix
    
    def calculate_punishment(self, state_matrix):
        """计算惩罚收益（与 MFPPO_UBP 完全一致）"""
        # 对状态矩阵进行padding处理（环形边界）
        padded_state = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), 
                        (1, 1, 1, 1), mode='circular')
        
        # 计算每个位置的邻居合作者数量
        neighbor_coop = F.conv2d(padded_state, self.neibor_kernel).squeeze()
        
        # 背叛者位置（state_matrix == 0）
        defector_positions = (state_matrix == 0)
        
        # 背叛者受到的惩罚：与邻居合作者数量成正比
        punishment_matrix = torch.zeros_like(state_matrix, dtype=torch.float32)
        punishment_matrix[defector_positions] = -self.punishment_strength * neighbor_coop[defector_positions]
        
        return punishment_matrix

    def calculate_reward_with_punishment(self, state_matrix):
        """计算包含惩罚机制的收益（与 MFPPO_UBP 完全一致）"""
        base_reward = self.calculate_reward(state_matrix)
        punishment_reward = self.calculate_punishment(state_matrix)
        total_reward = base_reward + punishment_reward
        return total_reward

    def fermi_update(self, type_t_matrix):
        """Fermi 更新规则（使用包含惩罚的收益）"""
        K = 0.1
        
        # 使用包含惩罚的收益计算
        if self.punishment_strength > 0:
            profit = self.calculate_reward_with_punishment(type_t_matrix)
        else:
            profit = self.calculate_reward(type_t_matrix)
        
        # 计算四个方向的 Fermi 概率
        W_left = 1 / (1 + torch.exp((profit - torch.roll(profit, 1, 1))/K))
        W_right = 1 / (1 + torch.exp((profit - torch.roll(profit, -1, 1))/K))
        W_up = 1 / (1 + torch.exp((profit - torch.roll(profit, 1, 0))/K))
        W_down = 1 / (1 + torch.exp((profit - torch.roll(profit, -1, 0))/K))

        learning_direction = torch.randint(0, 4, (self.L_num, self.L_num)).to(self.device)
        learning_probabilities = torch.rand(self.L_num, self.L_num).to(self.device)

        type_t1_matrix = (learning_direction == 0) * ((learning_probabilities <= W_left) * torch.roll(type_t_matrix, 1, 1) + (learning_probabilities > W_left) * type_t_matrix) + \
                        (learning_direction == 1) * ((learning_probabilities <= W_right) * torch.roll(type_t_matrix, -1, 1) + (learning_probabilities > W_right) * type_t_matrix) + \
                        (learning_direction == 2) * ((learning_probabilities <= W_up) * torch.roll(type_t_matrix, 1, 0) + (learning_probabilities > W_up) * type_t_matrix) + \
                        (learning_direction == 3) * ((learning_probabilities <= W_down) * torch.roll(type_t_matrix, -1, 0) + (learning_probabilities > W_down) * type_t_matrix)
        return type_t1_matrix.view(self.L_num, self.L_num)

    def run(self, num):
        """主运行循环（同步版本）"""
        coop_rates = []
        defect_rates = []
        total_values = []
        punishment_values = []  # 记录惩罚值
        
        for epoch in tqdm(range(self.epoches)):
            self.epoch = epoch
            
            if epoch == 0:
                if self.punishment_strength > 0:
                    profit_matrix = self.calculate_reward_with_punishment(self.current_state)
                    punishment_matrix = self.calculate_punishment(self.current_state)
                else:
                    profit_matrix = self.calculate_reward(self.current_state)
                    punishment_matrix = torch.zeros_like(self.current_state, dtype=torch.float32)
                
                # 同步绘图
                self.shot_pic_with_ubp_style(self.current_state, epoch, self.r, profit_matrix, punishment_matrix)
                
                current_coop_rate = self.current_state.float().mean().item()
                current_defect_rate = 1 - current_coop_rate
                current_profit = profit_matrix.mean().item()
                
                coop_rates.append(current_coop_rate)
                defect_rates.append(current_defect_rate)
                total_values.append(current_profit)
            
            # Fermi 更新
            self.current_state = self.fermi_update(self.current_state)

            # 记录惩罚值（如果有惩罚）
            if self.punishment_strength > 0:
                punishment = self.calculate_punishment(self.current_state)
                avg_punishment = punishment.mean().item()
                punishment_values.append(avg_punishment)

            # 在关键时间点保存快照
            if (epoch + 1 in [1, 10, 100, 1000, 10000, 100000]):
                if self.punishment_strength > 0:
                    profit_matrix = self.calculate_reward_with_punishment(self.current_state)
                    punishment_matrix = self.calculate_punishment(self.current_state)
                else:
                    profit_matrix = self.calculate_reward(self.current_state)
                    punishment_matrix = torch.zeros_like(self.current_state, dtype=torch.float32)
                
                self.shot_pic_with_ubp_style(self.current_state, epoch + 1, self.r, profit_matrix, punishment_matrix)

            # 计算当前指标
            current_coop_rate = self.current_state.float().mean().item()
            current_defect_rate = 1 - current_coop_rate
            
            if self.punishment_strength > 0:
                profit_matrix = self.calculate_reward_with_punishment(self.current_state)
            else:
                profit_matrix = self.calculate_reward(self.current_state)
            current_profit = profit_matrix.mean().item()
            
            coop_rates.append(current_coop_rate)
            defect_rates.append(current_defect_rate)
            total_values.append(current_profit)
        
        # 保存惩罚数据
        if self.punishment_strength > 0 and len(punishment_values) > 0:
            self.save_data('Punishment', f'Punishment_r{self.r}', self.r, num, punishment_values)
        
        # 保存快照
        self._save_snapshot(self.epoches, num)
        
        return defect_rates, coop_rates, total_values

    def _save_snapshot(self, epoch, run_num):
        """保存最终快照"""
        plt.figure(figsize=(8, 8))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 2)
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(2)
        plt.imshow(self.current_state.cpu().numpy(), cmap='gray_r')
        plt.title(f"Epoch {epoch}, Coop Rate: {self.current_state.float().mean().item():.2f}")
        plt.savefig(f"{self.output_path}/snapshot_run{run_num}_epoch{epoch}.pdf", 
                    format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_data(self, data_type, name, r, run_num, data):
        """保存数据"""
        output_dir = f'{self.output_path}/{data_type}'
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(f'{output_dir}/{name}_run{run_num}.txt', data)

    def shot_pic_with_ubp_style(self, type_t_matrix, epoch, r, profit_matrix, punishment_matrix=None):
        """绘制包含惩罚信息的快照（与 MFPPO_UBP 风格一致）"""
        plt.clf()
        plt.close("all")
        
        # 创建输出目录
        img_dir = f'{self.output_path}/shot_pic/r={r}/two_type'
        matrix_dir = f'{self.output_path}/shot_pic/r={r}/two_type/type_t_matrix'
        profit_dir = f'{self.output_path}/shot_pic/r={r}/two_type/profit_matrix'
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)
        os.makedirs(profit_dir, exist_ok=True)
        
        if punishment_matrix is not None:
            punishment_dir = f'{self.output_path}/shot_pic/r={r}/two_type/punishment_matrix'
            os.makedirs(punishment_dir, exist_ok=True)

        # 计算统计信息
        coop_rate = type_t_matrix.float().mean().item()
        defect_rate = 1 - coop_rate
        avg_profit = profit_matrix.mean().item()
        std_profit = profit_matrix.std().item()
        
        fig_dpi = 300

        # =============================================
        # 1. 策略分布图（与 MFPPO_UBP 配色一致）
        # =============================================
        plt.figure(figsize=(10, 8))
        color_map = {
            0: [0.8, 0.2, 0.2],  # 红色表示背叛者
            1: [0.2, 0.6, 0.8]   # 蓝色表示合作者
        }

        strategy_image = np.zeros((self.L_num, self.L_num, 3))
        for label, color in color_map.items():
            strategy_image[type_t_matrix.cpu().numpy() == label] = color

        plt.imshow(strategy_image, interpolation='none', aspect='equal')
        plt.axis('off')
        plt.tight_layout()
        
        pdf_path = f'{img_dir}/strategy_distribution_t={epoch}.pdf'
        png_path = f'{img_dir}/strategy_distribution_t={epoch}.png'
        plt.savefig(pdf_path, format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0.1)
        plt.savefig(png_path, format='png', dpi=fig_dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # =============================================
        # 2. 收益分布图
        # =============================================
        plt.figure(figsize=(8, 8))
        profit_data = profit_matrix.cpu().numpy()
        vmin, vmax = 0, 9
        plt.imshow(profit_data, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='none', aspect='equal')
        plt.axis('off')
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=28)
        cbar.set_ticks(np.arange(0, 9, 1))
        plt.tight_layout()
        
        pdf_path = f'{img_dir}/profit_t={epoch}.pdf'
        png_path = f'{img_dir}/profit_t={epoch}.png'
        plt.savefig(pdf_path, format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0)
        plt.savefig(png_path, format='png', dpi=fig_dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

        # =============================================
        # 3. 惩罚分布图（如果有惩罚）
        # =============================================
        if punishment_matrix is not None:
            plt.figure(figsize=(8, 8))
            punishment_data = punishment_matrix.cpu().numpy()
            punishment_abs = np.abs(punishment_data)
            vmin, vmax = 0, 4
            plt.imshow(punishment_abs, cmap='Reds', vmin=vmin, vmax=vmax, interpolation='none', aspect='equal')
            plt.axis('off')
            cbar = plt.colorbar(fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=28)
            cbar.set_ticks(np.arange(0, 4, 1))
            plt.tight_layout()
            
            pdf_path = f'{img_dir}/punishment_t={epoch}.pdf'
            png_path = f'{img_dir}/punishment_t={epoch}.png'
            plt.savefig(pdf_path, format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0)
            plt.savefig(png_path, format='png', dpi=fig_dpi, bbox_inches='tight', pad_inches=0)
            plt.close()

        # =============================================
        # 4. 保存矩阵数据文件
        # =============================================
        np.savetxt(f'{matrix_dir}/T{epoch}.txt',
                    type_t_matrix.cpu().numpy(), fmt='%d')
        np.savetxt(f'{profit_dir}/T{epoch}.txt',
                    profit_matrix.cpu().numpy(), fmt='%.4f')
        
        if punishment_matrix is not None:
            np.savetxt(f'{punishment_dir}/T{epoch}.txt',
                        punishment_matrix.cpu().numpy(), fmt='%.4f')
        
        return 0