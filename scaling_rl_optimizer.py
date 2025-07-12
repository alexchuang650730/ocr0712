#!/usr/bin/env python3
"""
OCR0712 Scaling RL 優化器實現
基於DeepSWE方法論的強化學習OCR優化系統
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import deque
import random
import json
import time

logger = logging.getLogger(__name__)

@dataclass
class RLState:
    """RL狀態表示"""
    image_features: torch.Tensor      # 圖像特徵
    current_prediction: str           # 當前預測
    confidence_map: torch.Tensor      # 置信度熱力圖
    context_history: List[str]        # 歷史上下文
    recognition_step: int             # 識別步驟
    error_count: int                  # 錯誤計數

@dataclass
class RLAction:
    """RL動作表示"""
    strategy: int                     # 識別策略 (0-4)
    parameters: Dict[str, float]      # 策略參數
    confidence_threshold: float       # 置信度閾值
    context_weight: float            # 上下文權重

class OCREnvironment:
    """OCR強化學習環境"""
    
    def __init__(self, base_ocr_model, max_steps: int = 10):
        self.base_ocr_model = base_ocr_model
        self.max_steps = max_steps
        self.current_step = 0
        self.current_state = None
        self.target_text = None
        self.done = False
        
        # 動作空間定義
        self.action_strategies = {
            0: 'pure_gan',           # 純GAN識別
            1: 'pure_baseline',      # 純基線模型識別  
            2: 'weighted_fusion',    # 加權融合
            3: 'confidence_voting',  # 置信度投票
            4: 'context_correction'  # 上下文修正
        }
        
        # 狀態歷史
        self.state_history = []
        self.reward_history = []
        
    def reset(self, image: torch.Tensor, target_text: str) -> RLState:
        """重置環境"""
        self.current_step = 0
        self.target_text = target_text
        self.done = False
        self.state_history = []
        self.reward_history = []
        
        # 提取初始圖像特徵
        with torch.no_grad():
            image_features = self.base_ocr_model.extract_features(image)
        
        # 初始化狀態
        self.current_state = RLState(
            image_features=image_features,
            current_prediction="",
            confidence_map=torch.zeros(64, 64),
            context_history=[],
            recognition_step=0,
            error_count=0
        )
        
        return self.current_state
    
    def step(self, action: RLAction) -> Tuple[RLState, float, bool, Dict]:
        """執行動作並返回新狀態"""
        
        if self.done:
            return self.current_state, 0.0, True, {}
        
        # 執行OCR動作
        ocr_result = self._execute_ocr_action(action)
        
        # 計算獎勵
        reward = self._calculate_reward(ocr_result, action)
        
        # 更新狀態
        self.current_state = self._update_state(ocr_result, action)
        
        # 檢查是否完成
        self.current_step += 1
        self.done = (
            self.current_step >= self.max_steps or 
            self._check_completion(ocr_result)
        )
        
        # 記錄歷史
        self.state_history.append(self.current_state)
        self.reward_history.append(reward)
        
        info = {
            'ocr_result': ocr_result,
            'action_strategy': self.action_strategies[action.strategy],
            'step': self.current_step
        }
        
        return self.current_state, reward, self.done, info
    
    def _execute_ocr_action(self, action: RLAction) -> Dict:
        """執行OCR動作"""
        
        strategy = action.strategy
        
        if strategy == 0:  # pure_gan
            return self._gan_recognition(action)
        elif strategy == 1:  # pure_baseline
            return self._baseline_recognition(action)
        elif strategy == 2:  # weighted_fusion
            return self._weighted_fusion(action)
        elif strategy == 3:  # confidence_voting
            return self._confidence_voting(action)
        elif strategy == 4:  # context_correction
            return self._context_correction(action)
        else:
            return {"text": "", "confidence": 0.0}
    
    def _gan_recognition(self, action: RLAction) -> Dict:
        """GAN軌跡識別"""
        # 簡化實現
        return {
            "text": f"GAN_pred_{self.current_step}",
            "confidence": np.random.uniform(0.7, 0.95),
            "method": "gan"
        }
    
    def _baseline_recognition(self, action: RLAction) -> Dict:
        """基線模型識別"""
        return {
            "text": f"BASE_pred_{self.current_step}",
            "confidence": np.random.uniform(0.6, 0.9),
            "method": "baseline"
        }
    
    def _weighted_fusion(self, action: RLAction) -> Dict:
        """加權融合識別"""
        gan_result = self._gan_recognition(action)
        base_result = self._baseline_recognition(action)
        
        # 加權融合
        w1 = action.parameters.get('gan_weight', 0.6)
        w2 = 1 - w1
        
        fused_confidence = w1 * gan_result['confidence'] + w2 * base_result['confidence']
        
        return {
            "text": f"FUSED_{self.current_step}",
            "confidence": fused_confidence,
            "method": "weighted_fusion"
        }
    
    def _confidence_voting(self, action: RLAction) -> Dict:
        """置信度投票"""
        results = [
            self._gan_recognition(action),
            self._baseline_recognition(action)
        ]
        
        # 選擇置信度最高的
        best_result = max(results, key=lambda x: x['confidence'])
        
        return {
            "text": f"VOTE_{self.current_step}",
            "confidence": best_result['confidence'],
            "method": "confidence_voting"
        }
    
    def _context_correction(self, action: RLAction) -> Dict:
        """上下文修正"""
        base_result = self._baseline_recognition(action)
        
        # 基於上下文調整
        context_boost = action.parameters.get('context_weight', 0.1)
        adjusted_confidence = min(1.0, base_result['confidence'] + context_boost)
        
        return {
            "text": f"CTX_{self.current_step}",
            "confidence": adjusted_confidence,
            "method": "context_correction"
        }
    
    def _calculate_reward(self, ocr_result: Dict, action: RLAction) -> float:
        """計算獎勵"""
        
        # 基礎準確率獎勵
        predicted_text = ocr_result.get('text', '')
        accuracy_reward = self._calculate_accuracy_reward(predicted_text)
        
        # 置信度獎勵
        confidence = ocr_result.get('confidence', 0.0)
        confidence_reward = confidence * 0.2
        
        # 效率獎勵 (更少步驟更好)
        efficiency_reward = max(0, (self.max_steps - self.current_step) / self.max_steps * 0.1)
        
        # 策略多樣性獎勵
        diversity_reward = self._calculate_diversity_reward(action)
        
        total_reward = accuracy_reward + confidence_reward + efficiency_reward + diversity_reward
        
        return total_reward
    
    def _calculate_accuracy_reward(self, predicted_text: str) -> float:
        """計算準確率獎勵"""
        if not self.target_text:
            return 0.0
        
        # 簡化的相似度計算
        if predicted_text == self.target_text:
            return 1.0
        else:
            # 基於編輯距離的相似度
            import difflib
            similarity = difflib.SequenceMatcher(None, predicted_text, self.target_text).ratio()
            return similarity * 0.8
    
    def _calculate_diversity_reward(self, action: RLAction) -> float:
        """計算策略多樣性獎勵"""
        if len(self.state_history) < 2:
            return 0.0
        
        # 鼓勵探索不同策略
        recent_strategies = [s.recognition_step for s in self.state_history[-3:]]
        unique_strategies = len(set(recent_strategies))
        
        return unique_strategies / 3.0 * 0.05
    
    def _update_state(self, ocr_result: Dict, action: RLAction) -> RLState:
        """更新狀態"""
        
        return RLState(
            image_features=self.current_state.image_features,
            current_prediction=ocr_result.get('text', ''),
            confidence_map=self._update_confidence_map(ocr_result),
            context_history=self.current_state.context_history + [ocr_result.get('text', '')],
            recognition_step=self.current_step,
            error_count=self._update_error_count(ocr_result)
        )
    
    def _update_confidence_map(self, ocr_result: Dict) -> torch.Tensor:
        """更新置信度熱力圖"""
        confidence = ocr_result.get('confidence', 0.0)
        return torch.ones(64, 64) * confidence
    
    def _update_error_count(self, ocr_result: Dict) -> int:
        """更新錯誤計數"""
        predicted_text = ocr_result.get('text', '')
        is_error = predicted_text != self.target_text
        return self.current_state.error_count + (1 if is_error else 0)
    
    def _check_completion(self, ocr_result: Dict) -> bool:
        """檢查是否完成"""
        predicted_text = ocr_result.get('text', '')
        confidence = ocr_result.get('confidence', 0.0)
        
        # 高置信度且預測正確
        return confidence > 0.95 and predicted_text == self.target_text

class PolicyNetwork(nn.Module):
    """策略網絡"""
    
    def __init__(self, state_dim: int = 512, action_dim: int = 5, hidden_dim: int = 256):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 策略選擇分支
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 參數預測分支
        self.parameter_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # gan_weight, confidence_threshold, context_weight, etc.
        )
        
    def forward(self, state_features: torch.Tensor):
        """前向傳播"""
        
        encoded = self.state_encoder(state_features)
        
        # 策略logits
        strategy_logits = self.strategy_head(encoded)
        
        # 參數預測
        parameters = torch.sigmoid(self.parameter_head(encoded))
        
        return strategy_logits, parameters

class ValueNetwork(nn.Module):
    """價值網絡"""
    
    def __init__(self, state_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state_features: torch.Tensor):
        return self.value_net(state_features)

class DeepSWEOptimizedPPO:
    """基於DeepSWE優化的PPO算法"""
    
    def __init__(self, state_dim: int = 512, action_dim: int = 5, lr: float = 3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 網絡
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        # 優化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # DeepSWE優化配置
        self.deepswe_config = {
            'clip_high_dapo': True,      # 裁剪高DAPO
            'remove_kl_loss': True,      # 移除KL損失
            'remove_reward_std': True,   # 移除獎勵標準化
            'length_normalization': True, # 長度標準化
            'one_sample_removal': True,  # 移除單樣本
            'compact_filtering': True,   # 緊湊過濾
            'remove_entropy_loss': True  # 移除熵損失
        }
        
        # 超參數
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.lam = 0.95
        self.epochs = 10
        
        # 經驗緩衝
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
    def state_to_features(self, state: RLState) -> torch.Tensor:
        """將狀態轉換為特徵向量"""
        
        # 簡化的特徵提取
        features = []
        
        # 圖像特徵 (假設已經是512維)
        if state.image_features.dim() > 1:
            img_features = state.image_features.flatten()[:256]
        else:
            img_features = state.image_features[:256]
        features.append(img_features)
        
        # 置信度特徵
        conf_features = state.confidence_map.flatten()[:64]
        features.append(conf_features)
        
        # 步驟和錯誤特徵
        step_features = torch.tensor([
            state.recognition_step / 10.0,  # 標準化
            state.error_count / 10.0,
            len(state.context_history) / 10.0
        ], dtype=torch.float32)
        features.append(step_features)
        
        # 填充到512維
        all_features = torch.cat(features)
        if all_features.size(0) < 512:
            padding = torch.zeros(512 - all_features.size(0))
            all_features = torch.cat([all_features, padding])
        elif all_features.size(0) > 512:
            all_features = all_features[:512]
        
        return all_features
    
    def select_action(self, state: RLState) -> Tuple[RLAction, torch.Tensor, torch.Tensor]:
        """選擇動作"""
        
        state_features = self.state_to_features(state).unsqueeze(0)
        
        with torch.no_grad():
            strategy_logits, parameters = self.policy_net(state_features)
            value = self.value_net(state_features)
        
        # 策略分佈
        strategy_dist = Categorical(logits=strategy_logits)
        strategy = strategy_dist.sample()
        log_prob = strategy_dist.log_prob(strategy)
        
        # 構建動作
        action = RLAction(
            strategy=strategy.item(),
            parameters={
                'gan_weight': parameters[0, 0].item(),
                'confidence_threshold': parameters[0, 1].item() * 0.5 + 0.5,  # 0.5-1.0
                'context_weight': parameters[0, 2].item() * 0.2,  # 0-0.2
                'other_param': parameters[0, 3].item()
            },
            confidence_threshold=parameters[0, 1].item() * 0.5 + 0.5,
            context_weight=parameters[0, 2].item() * 0.2
        )
        
        return action, log_prob, value
    
    def store_transition(self, state: RLState, action: RLAction, reward: float, 
                        value: torch.Tensor, log_prob: torch.Tensor, done: bool):
        """存儲轉移"""
        
        self.memory['states'].append(self.state_to_features(state))
        self.memory['actions'].append(action.strategy)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def calculate_advantages(self, rewards: List[float], values: List[torch.Tensor], 
                           dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """計算優勢函數"""
        
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # DeepSWE優化：移除獎勵標準化
        if not self.deepswe_config['remove_reward_std']:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self):
        """更新策略"""
        
        if len(self.memory['states']) == 0:
            return
        
        # 準備數據
        states = torch.stack(self.memory['states'])
        actions = torch.tensor(self.memory['actions'], dtype=torch.long)
        old_log_probs = torch.stack(self.memory['log_probs'])
        values = torch.stack(self.memory['values']).squeeze()
        
        # 計算優勢和回報
        advantages, returns = self.calculate_advantages(
            self.memory['rewards'], 
            self.memory['values'], 
            self.memory['dones']
        )
        
        # DeepSWE優化：緊湊過濾
        if self.deepswe_config['compact_filtering']:
            # 只保留高優勢樣本
            high_advantage_mask = advantages > advantages.median()
            if high_advantage_mask.sum() > 0:
                states = states[high_advantage_mask]
                actions = actions[high_advantage_mask]
                old_log_probs = old_log_probs[high_advantage_mask]
                advantages = advantages[high_advantage_mask]
                returns = returns[high_advantage_mask]
        
        # PPO更新
        for epoch in range(self.epochs):
            # 策略網絡前向傳播
            strategy_logits, _ = self.policy_net(states)
            dist = Categorical(logits=strategy_logits)
            new_log_probs = dist.log_prob(actions)
            
            # 計算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 計算裁剪損失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # DeepSWE優化：移除熵損失
            if not self.deepswe_config['remove_entropy_loss']:
                entropy_loss = -dist.entropy().mean()
                policy_loss += 0.01 * entropy_loss
            
            # 更新策略
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            
            # DeepSWE優化：裁剪高梯度
            if self.deepswe_config['clip_high_dapo']:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            
            self.policy_optimizer.step()
            
            # 價值網絡更新
            current_values = self.value_net(states).squeeze()
            value_loss = F.mse_loss(current_values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # 清空記憶體
        self.clear_memory()
    
    def clear_memory(self):
        """清空記憶體"""
        for key in self.memory:
            self.memory[key] = []

class ScalingRLTrainer:
    """Scaling RL訓練器"""
    
    def __init__(self, base_ocr_model, config: dict):
        self.base_ocr_model = base_ocr_model
        self.config = config
        
        # 環境和算法
        self.env = OCREnvironment(base_ocr_model)
        self.rl_agent = DeepSWEOptimizedPPO()
        
        # 訓練統計
        self.episode_rewards = []
        self.episode_lengths = []
        self.accuracy_history = []
        
    def train_episode(self, image: torch.Tensor, target_text: str) -> Dict:
        """訓練一個episode"""
        
        state = self.env.reset(image, target_text)
        episode_reward = 0
        episode_length = 0
        
        while not self.env.done:
            # 選擇動作
            action, log_prob, value = self.rl_agent.select_action(state)
            
            # 執行動作
            next_state, reward, done, info = self.env.step(action)
            
            # 存儲轉移
            self.rl_agent.store_transition(state, action, reward, value, log_prob, done)
            
            # 更新統計
            episode_reward += reward
            episode_length += 1
            
            state = next_state
        
        # 記錄統計
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # 計算準確率
        final_prediction = state.current_prediction
        accuracy = 1.0 if final_prediction == target_text else 0.0
        self.accuracy_history.append(accuracy)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'accuracy': accuracy,
            'final_prediction': final_prediction
        }
    
    def train(self, training_data: List[Tuple[torch.Tensor, str]], num_episodes: int = 1000):
        """主訓練循環"""
        
        logger.info(f"Starting Scaling RL training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            # 隨機選擇訓練樣本
            image, target_text = random.choice(training_data)
            
            # 訓練一個episode
            episode_info = self.train_episode(image, target_text)
            
            # 每10個episode更新一次策略
            if (episode + 1) % 10 == 0:
                self.rl_agent.update_policy()
            
            # 記錄進度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_accuracy = np.mean(self.accuracy_history[-100:])
                logger.info(
                    f"Episode {episode + 1}: "
                    f"Avg Reward: {avg_reward:.3f}, "
                    f"Avg Accuracy: {avg_accuracy:.3f}"
                )
        
        logger.info("Scaling RL training completed!")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'accuracy_history': self.accuracy_history
        }
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net': self.rl_agent.policy_net.state_dict(),
            'value_net': self.rl_agent.value_net.state_dict(),
            'config': self.config,
            'statistics': {
                'episode_rewards': self.episode_rewards,
                'accuracy_history': self.accuracy_history
            }
        }, path)
        
        logger.info(f"Scaling RL model saved to {path}")
    
    def load_model(self, path: str):
        """加載模型"""
        checkpoint = torch.load(path)
        
        self.rl_agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.rl_agent.value_net.load_state_dict(checkpoint['value_net'])
        self.episode_rewards = checkpoint['statistics']['episode_rewards']
        self.accuracy_history = checkpoint['statistics']['accuracy_history']
        
        logger.info(f"Scaling RL model loaded from {path}")

def create_sample_training_data() -> List[Tuple[torch.Tensor, str]]:
    """創建示例訓練數據"""
    
    training_data = []
    
    for i in range(100):
        # 創建隨機圖像
        image = torch.randn(3, 224, 224)
        
        # 創建目標文本
        target_text = f"sample_text_{i}"
        
        training_data.append((image, target_text))
    
    return training_data

def main():
    """主函數 - 演示Scaling RL"""
    
    # 創建基礎OCR模型 (簡化)
    class MockOCRModel:
        def extract_features(self, image):
            return torch.randn(512)
    
    base_ocr_model = MockOCRModel()
    
    # 配置
    config = {
        'state_dim': 512,
        'action_dim': 5,
        'learning_rate': 3e-4,
        'episodes': 1000
    }
    
    # 創建訓練器
    trainer = ScalingRLTrainer(base_ocr_model, config)
    
    # 創建訓練數據
    training_data = create_sample_training_data()
    
    # 開始訓練
    results = trainer.train(training_data, num_episodes=config['episodes'])
    
    # 保存模型
    trainer.save_model("scaling_rl_model.pth")
    
    print("Scaling RL training completed!")
    print(f"Final average accuracy: {np.mean(results['accuracy_history'][-100:]):.3f}")

if __name__ == "__main__":
    main()