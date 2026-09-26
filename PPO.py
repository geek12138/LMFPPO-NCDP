import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import asyncio
from torch.optim.lr_scheduler import StepLR

class ActorCritic(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        shared = self.shared(x)
        action_logits = self.actor(shared)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic(shared)
        state_value=state_value.squeeze()
        return action_probs, state_value

class SPGG(nn.Module):
    def __init__(self, L_num, device, alpha, gamma, clip_epsilon, r, epochs, 
                now_time, question, ppo_epochs, batch_size, gae_lambda,
                output_path, delta, rho, punishment_strength=0.0):
        super().__init__()
        self.L_num = L_num
        self.device = device
        self.r = r
        self.epochs = epochs
        self.question = question
        self.now_time = now_time
        
        # PPO超参数
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.delta=delta
        self.rho=rho
        self.punishment_strength = punishment_strength
        self.output_path=output_path
        
        # 神经网络
        self.policy = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.5)
        
        # 邻域卷积核
        self.neibor_kernel = torch.tensor(
            [[[[0,1,0], [1,1,1], [0,1,0]]]], 
            dtype=torch.float32, device=device
        )
        
        # 初始化状态
        self.initial_state = self._init_state(question)
        self.current_state = self.initial_state.clone()
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

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
        padded_state = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), 
                        (1, 1, 1, 1), mode='circular')
        neighbor_coop = F.conv2d(padded_state, self.neibor_kernel).squeeze()
        defector_positions = (state_matrix == 0)
        punishment_matrix = torch.zeros_like(state_matrix, dtype=torch.float32)
        punishment_matrix[defector_positions] = -self.punishment_strength * neighbor_coop[defector_positions]
        return punishment_matrix

    def calculate_reward_with_punishment(self, state_matrix):
        base_reward = self.calculate_reward(state_matrix)
        punishment_reward = self.calculate_punishment(state_matrix)
        total_reward = base_reward + punishment_reward
        return total_reward
    
    def ppo_update(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        rewards = torch.stack(self.rewards)
        next_states = torch.stack(self.next_states)
        dones = torch.stack(self.dones)

        with torch.no_grad():
            _, values = self.policy(states)
            _, next_values = self.policy(next_states)

        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            dones_float = dones[t].float()
            psi = rewards[t] + self.gamma * next_values[t] * (1 - dones_float) - values[t]
            advantages[t] = psi + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        for _ in range(self.ppo_epochs):
            for batch in self._make_batch(states, actions, old_log_probs, advantages, returns):
                state_b, action_b, old_log_b, adv_b, ret_b = batch
                if ret_b.shape[0]==1:
                    ret_b=ret_b.squeeze()
                probs, value_pred = self.policy(state_b)
                dist = Categorical(probs)
                log_probs = dist.log_prob(action_b).view_as(action_b)
                entropy = dist.entropy().mean()
                
                ratio = (log_probs - old_log_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value_pred, ret_b)

                loss = actor_loss + self.delta * critic_loss - self.rho * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()

    def _make_batch(self, states, actions, old_log_probs, advantages, returns):
        perm = torch.randperm(len(states))
        for i in range(0, len(states), self.batch_size):
            idx = perm[i:i+self.batch_size]
            yield (states[idx], actions[idx], old_log_probs[idx], advantages[idx], returns[idx])

    def run(self):
        coop_rates = []
        defect_rates = []
        total_values = []
        
        for epoch in tqdm(range(self.epochs)):
            self.epoch = epoch
            action, log_prob = self.choose_action(self.current_state)
            next_state = action
            
            if self.punishment_strength > 0:
                reward = self.calculate_reward_with_punishment(next_state)
            else:
                reward = self.calculate_reward(next_state)
            
            done = torch.zeros_like(next_state, dtype=torch.bool)
            
            self.states.append(self.encode_state(self.current_state).view(self.L_num,self.L_num,3))
            self.actions.append(action)
            self.log_probs.append(log_prob)

            self.rewards.append(reward)
            self.next_states.append(self.encode_state(next_state).view(self.L_num,self.L_num,3))
            self.dones.append(done)
            
            if epoch==0:
                profit_matrix = self.calculate_reward(self.current_state)
                self.shot_pic(self.current_state, epoch, self.r, profit_matrix)
                coop_rate = self.current_state.float().mean().item()
                defect_rate = 1 - coop_rate
                total_value = reward.sum().item()
                
                coop_rates.append(coop_rate)
                defect_rates.append(defect_rate)
                total_values.append(total_value)
            if len(self.states) >= self.batch_size * self.ppo_epochs:
                self.ppo_update()
                self.current_state = next_state
                self._reset_buffer()
            else:
                self.current_state = next_state           

            if (epoch+1 in [1, 10, 100, 1000, 10000, 100000]):
                profit_matrix = self.calculate_reward(self.current_state)
                self.shot_pic(self.current_state, epoch+1, self.r, profit_matrix)

            if epoch % 1000 == 0:
                self.save_checkpoint()
            
            coop_rate = self.current_state.float().mean().item()
            defect_rate = 1 - coop_rate
            total_value = reward.sum().item()
            
            coop_rates.append(coop_rate)
            defect_rates.append(defect_rate)
            total_values.append(total_value)
        
        self.save_checkpoint(is_final=True)

        return defect_rates, coop_rates, [], [], total_values

    def save_data(self, data_type, name, r, data):
        output_dir = f'{self.output_path}/{data_type}'
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(f'{output_dir}/{name}.txt', data)

    def shot_pic(self, type_t_matrix, epoch, r, profit_matrix):
        plt.clf()
        plt.close("all")
        
        img_dir = f'{self.output_path}/shot_pic/r={r}/two_type'
        matrix_dir = f'{self.output_path}/shot_pic/r={r}/two_type/type_t_matrix'
        profit_dir = f'{self.output_path}/shot_pic/r={r}/two_type/profit_matrix'
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)  
        os.makedirs(profit_dir, exist_ok=True)

        coop_rate = type_t_matrix.float().mean().item()
        defect_rate = 1 - coop_rate
        avg_profit = profit_matrix.mean().item()
        std_profit = profit_matrix.std().item()
        
        fig_dpi = 300

        # 子图1: 策略分布
        plt.figure(figsize=(10, 8))
        color_map = {
            0: [0.8, 0.2, 0.2],
            1: [0.2, 0.6, 0.8]
        }

        strategy_image = np.zeros((self.L_num, self.L_num, 3))
        for label, color in color_map.items():
            strategy_image[type_t_matrix.cpu().numpy() == label] = color

        plt.imshow(strategy_image, interpolation='none', aspect='equal')
        plt.axis('off')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map[1], label=f'Cooperator (C): {coop_rate:.3f}'),
            Patch(facecolor=color_map[0], label=f'Defector (D): {defect_rate:.3f}')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(f'{img_dir}/strategy_distribution_t={epoch}.pdf', 
                format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 子图2: 收益分布
        plt.figure(figsize=(10, 8))
        profit_data = profit_matrix.cpu().numpy()
        plt.imshow(profit_data, cmap='viridis', interpolation='none', aspect='equal')
        plt.axis('off')
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_label('Profit Value', fontsize=12, rotation=270, labelpad=20)

        plt.tight_layout()
        plt.savefig(f'{img_dir}/profit_distribution_t={epoch}.pdf', 
                format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 子图3: 收益直方图
        plt.figure(figsize=(10, 8))
        profit_flat = profit_data.flatten()
        coop_profits = profit_flat[type_t_matrix.cpu().numpy().flatten() == 1]
        defect_profits = profit_flat[type_t_matrix.cpu().numpy().flatten() == 0]

        has_coop_data = len(coop_profits) > 0
        has_defect_data = len(defect_profits) > 0
        
        if has_coop_data or has_defect_data:
            all_profits = np.concatenate([coop_profits, defect_profits]) if has_coop_data and has_defect_data else (coop_profits if has_coop_data else defect_profits)
            
            if len(all_profits) > 1 and np.ptp(all_profits) > 1e-10:
                bins = np.linspace(all_profits.min(), all_profits.max(), 30)
                
                if has_coop_data:
                    plt.hist(coop_profits, bins=bins, alpha=0.7, color='blue', 
                            label=f'Cooperator (n={len(coop_profits)})', density=True)
                
                if has_defect_data:
                    plt.hist(defect_profits, bins=bins, alpha=0.7, color='red', 
                            label=f'Defector (n={len(defect_profits)})', density=True)
            else:
                bins = 20
                if has_coop_data:
                    plt.hist(coop_profits, bins=bins, alpha=0.7, color='blue', 
                            label=f'Cooperator (n={len(coop_profits)})', density=True)
                
                if has_defect_data:
                    plt.hist(defect_profits, bins=bins, alpha=0.7, color='red', 
                            label=f'Defector (n={len(defect_profits)})', density=True)

            plt.xlabel('Profit Value', fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.legend(loc='upper right', fontsize=12)
            plt.grid(True, alpha=0.3)

            stats_lines = []
            if has_coop_data:
                coop_mean = np.mean(coop_profits)
                stats_lines.append(f'Cooperator Mean: {coop_mean:.3f}')
            else:
                stats_lines.append('Cooperator Mean: N/A')
                
            if has_defect_data:
                defect_mean = np.mean(defect_profits)
                stats_lines.append(f'Defector Mean: {defect_mean:.3f}')
            else:
                stats_lines.append('Defector Mean: N/A')
                
            stats_lines.append(f'Overall Mean: {avg_profit:.3f}')
            stats_lines.append(f'Overall Std: {std_profit:.3f}')
            
            stats_text = '\n'.join(stats_lines)
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        else:
            plt.text(0.5, 0.5, 'No profit data available\nfor histogram analysis', 
                    transform=plt.gca().transAxes, ha='center', va='center', fontsize=14)

        plt.tight_layout()
        plt.savefig(f'{img_dir}/profit_histogram_t={epoch}.pdf', 
                format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 子图4: 空间自相关分析
        plt.figure(figsize=(10, 8))

        padded_state = F.pad(type_t_matrix.float().unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='circular')
        local_coop_density = F.conv2d(padded_state, self.neibor_kernel).squeeze() / 4.0

        local_density_flat = local_coop_density.cpu().numpy().flatten()
        strategy_flat = type_t_matrix.cpu().numpy().flatten()

        coop_mask = strategy_flat == 1
        defect_mask = strategy_flat == 0

        has_coop_points = np.sum(coop_mask) > 0
        has_defect_points = np.sum(defect_mask) > 0
        
        if has_coop_points or has_defect_points:
            if has_coop_points:
                plt.scatter(local_density_flat[coop_mask], profit_flat[coop_mask], 
                        alpha=0.6, color='blue', s=20, label='Cooperator')
            
            if has_defect_points:
                plt.scatter(local_density_flat[defect_mask], profit_flat[defect_mask], 
                        alpha=0.6, color='red', s=20, label='Defector')

            plt.xlabel('Local Cooperation Density', fontsize=14)
            plt.ylabel('Individual Profit', fontsize=14)
            plt.legend(loc='upper right', fontsize=12)
            plt.grid(True, alpha=0.3)

            valid_mask = ~np.isnan(local_density_flat) & ~np.isnan(profit_flat) & ~np.isinf(local_density_flat) & ~np.isinf(profit_flat)
            if np.sum(valid_mask) > 1:
                try:
                    correlation = np.corrcoef(local_density_flat[valid_mask], profit_flat[valid_mask])[0, 1]
                    correlation_text = f'Correlation: {correlation:.3f}'
                except:
                    correlation_text = 'Correlation: N/A'
            else:
                correlation_text = 'Correlation: N/A (insufficient data)'
                
            plt.text(0.02, 0.98, correlation_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            plt.text(0.5, 0.5, 'No data available\nfor spatial correlation analysis', 
                    transform=plt.gca().transAxes, ha='center', va='center', fontsize=14)

        plt.tight_layout()
        plt.savefig(f'{img_dir}/spatial_correlation_t={epoch}.pdf', 
                format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        np.savetxt(f'{matrix_dir}/T{epoch}.txt', type_t_matrix.cpu().numpy(), fmt='%d')
        np.savetxt(f'{profit_dir}/T{epoch}.txt', profit_matrix.cpu().numpy(), fmt='%.4f')
        
        return 0

    def _reset_buffer(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]
        torch.cuda.empty_cache()
    
    def _store_transition(self, state, action, log_prob, reward, next_state, done):
        self.states.append(state.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.next_states.append(next_state.detach().cpu())
        self.dones.append(done.detach().cpu())

    def save_checkpoint(self, is_final=False):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'r': self.r,
            'gamma': self.gamma,
            'clip_epsilon': self.clip_epsilon,
            'punishment_strength': self.punishment_strength,
        }
        model_dir = f"{self.output_path}/checkpoint"
        os.makedirs(model_dir, exist_ok=True)
        filename = f"model_r{self.r}_final.pth" if is_final else f"model_r{self.r}_epoch{self.epoch}.pth"
        torch.save(checkpoint, f"{model_dir}/{filename}")
    
    def choose_action(self, state_matrix):
        with torch.no_grad():
            features = self.encode_state(state_matrix)
            probs, _ = self.policy(features)
            dist = Categorical(probs)
            actions = dist.sample()
        return actions.view_as(state_matrix), dist.log_prob(actions).view_as(state_matrix)