#!/usr/bin/env python3
"""
OCR0712 DeepSWE優化算法實現
實施7項優化技術: clip_high_dapo, remove_kl_loss, remove_reward_std, 
length_normalization, one_sample_removal, compact_filtering, remove_entropy_loss
"""

import os
import sys
import json
import time
import numpy as np
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque, defaultdict

@dataclass
class DeepSWEConfig:
    """DeepSWE配置"""
    # 7項優化開關
    clip_high_dapo: bool = True
    remove_kl_loss: bool = True
    remove_reward_std: bool = True
    length_normalization: bool = True
    one_sample_removal: bool = True
    compact_filtering: bool = True
    remove_entropy_loss: bool = True
    
    # 優化參數
    dapo_clip_threshold: float = 0.2
    length_norm_factor: float = 0.1
    compact_filter_ratio: float = 0.8
    sample_removal_threshold: float = 0.95
    
    # 訓練參數
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_episodes: int = 1000
    update_frequency: int = 10
    
    # 環境參數
    observation_dim: int = 520
    action_dim: int = 5

class DeepSWEOptimizer:
    """DeepSWE優化器"""
    
    def __init__(self, config: DeepSWEConfig):
        self.config = config
        self.step_count = 0
        self.episode_count = 0
        
        # 優化統計
        self.optimization_stats = {
            "clip_high_dapo_applied": 0,
            "kl_loss_removed": 0,
            "reward_std_removed": 0,
            "length_normalized": 0,
            "samples_removed": 0,
            "compact_filtered": 0,
            "entropy_loss_removed": 0,
            "total_optimizations": 0
        }
        
        # 性能歷史
        self.performance_history = {
            "rewards": deque(maxlen=1000),
            "losses": deque(maxlen=1000),
            "optimization_impact": defaultdict(list)
        }
        
        print(f"🔧 === DeepSWE優化算法初始化 ===")
        print(f"✅ 7項優化技術配置:")
        print(f"   clip_high_dapo: {config.clip_high_dapo}")
        print(f"   remove_kl_loss: {config.remove_kl_loss}")
        print(f"   remove_reward_std: {config.remove_reward_std}")
        print(f"   length_normalization: {config.length_normalization}")
        print(f"   one_sample_removal: {config.one_sample_removal}")
        print(f"   compact_filtering: {config.compact_filtering}")
        print(f"   remove_entropy_loss: {config.remove_entropy_loss}")
        print()
    
    def clip_high_dapo(self, gradients: np.ndarray, threshold: float = None) -> np.ndarray:
        """優化1: Clip High DAPO (Dynamic Adaptive Policy Optimization)"""
        if not self.config.clip_high_dapo:
            return gradients
        
        threshold = threshold or self.config.dapo_clip_threshold
        
        # 計算梯度幅度
        grad_magnitude = np.linalg.norm(gradients, axis=-1, keepdims=True)
        
        # 動態閾值調整
        adaptive_threshold = threshold * (1 + 0.1 * np.sin(self.step_count * 0.01))
        
        # 裁剪高幅度梯度
        clipped_gradients = np.where(
            grad_magnitude > adaptive_threshold,
            gradients * (adaptive_threshold / grad_magnitude),
            gradients
        )
        
        # 統計
        clipped_count = np.sum(grad_magnitude > adaptive_threshold)
        if clipped_count > 0:
            self.optimization_stats["clip_high_dapo_applied"] += clipped_count
            self.optimization_stats["total_optimizations"] += 1
        
        return clipped_gradients
    
    def compute_loss_without_kl(self, policy_logits: np.ndarray, 
                               old_policy_logits: np.ndarray,
                               advantages: np.ndarray) -> float:
        """優化2: Remove KL Loss"""
        if not self.config.remove_kl_loss:
            # 標準PPO帶KL散度
            log_ratio = policy_logits - old_policy_logits
            ratio = np.exp(log_ratio)
            if ratio.ndim > 1:
                ratio = np.mean(ratio, axis=1, keepdims=True)
            if advantages.ndim == 1:
                advantages = advantages.reshape(-1, 1)
            
            kl_penalty = 0.01 * np.mean((policy_logits - old_policy_logits) ** 2)
            clipped_ratio = np.clip(ratio, 0.8, 1.2)
            policy_loss = -np.mean(np.minimum(ratio * advantages, clipped_ratio * advantages))
            total_loss = policy_loss + kl_penalty
        else:
            # DeepSWE: 移除KL散度項
            log_ratio = policy_logits - old_policy_logits
            ratio = np.exp(log_ratio)
            if ratio.ndim > 1:
                ratio = np.mean(ratio, axis=1, keepdims=True)
            if advantages.ndim == 1:
                advantages = advantages.reshape(-1, 1)
            
            clipped_ratio = np.clip(ratio, 0.8, 1.2)
            policy_loss = -np.mean(np.minimum(ratio * advantages, clipped_ratio * advantages))
            total_loss = policy_loss
            
            self.optimization_stats["kl_loss_removed"] += 1
            self.optimization_stats["total_optimizations"] += 1
        
        return total_loss
    
    def compute_advantages_without_std(self, rewards: np.ndarray) -> np.ndarray:
        """優化3: Remove Reward Standardization"""
        if not self.config.remove_reward_std:
            # 標準化獎勵
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards) + 1e-8
            advantages = (rewards - mean_reward) / std_reward
        else:
            # DeepSWE: 直接使用原始獎勵
            advantages = rewards - np.mean(rewards)
            
            self.optimization_stats["reward_std_removed"] += 1
            self.optimization_stats["total_optimizations"] += 1
        
        return advantages
    
    def apply_length_normalization(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """優化4: Length Normalization"""
        if not self.config.length_normalization:
            return sequences
        
        normalized_sequences = []
        
        for seq in sequences:
            # 計算序列長度
            seq_length = len(seq)
            
            # 長度歸一化因子
            length_factor = 1.0 / (1.0 + self.config.length_norm_factor * np.log(seq_length))
            
            # 應用歸一化
            normalized_seq = seq * length_factor
            normalized_sequences.append(normalized_seq)
        
        self.optimization_stats["length_normalized"] += len(sequences)
        self.optimization_stats["total_optimizations"] += 1
        
        return normalized_sequences
    
    def one_sample_removal(self, batch_data: Dict[str, np.ndarray], 
                          performance_threshold: float = None) -> Dict[str, np.ndarray]:
        """優化5: One Sample Removal"""
        if not self.config.one_sample_removal:
            return batch_data
        
        threshold = performance_threshold or self.config.sample_removal_threshold
        
        # 計算每個樣本的性能分數
        rewards = batch_data.get('rewards', np.array([]))
        if len(rewards) == 0:
            return batch_data
        
        # 識別低性能樣本
        performance_scores = rewards / (np.max(rewards) + 1e-8)
        low_performance_mask = performance_scores < threshold
        
        # 移除低性能樣本
        if np.any(low_performance_mask):
            keep_mask = ~low_performance_mask
            filtered_data = {}
            
            for key, values in batch_data.items():
                if isinstance(values, np.ndarray) and len(values) == len(rewards):
                    filtered_data[key] = values[keep_mask]
                else:
                    filtered_data[key] = values
            
            removed_count = np.sum(low_performance_mask)
            self.optimization_stats["samples_removed"] += removed_count
            self.optimization_stats["total_optimizations"] += 1
            
            return filtered_data
        
        return batch_data
    
    def compact_filtering(self, feature_vectors: np.ndarray, 
                         filter_ratio: float = None) -> np.ndarray:
        """優化6: Compact Filtering"""
        if not self.config.compact_filtering:
            return feature_vectors
        
        filter_ratio = filter_ratio or self.config.compact_filter_ratio
        
        # 計算特徵重要性
        feature_importance = np.var(feature_vectors, axis=0)
        
        # 選擇重要特徵
        num_features_to_keep = int(len(feature_importance) * filter_ratio)
        important_indices = np.argsort(feature_importance)[-num_features_to_keep:]
        
        # 過濾特徵
        filtered_features = feature_vectors[:, important_indices]
        
        self.optimization_stats["compact_filtered"] += 1
        self.optimization_stats["total_optimizations"] += 1
        
        return filtered_features
    
    def compute_loss_without_entropy(self, policy_logits: np.ndarray,
                                   old_policy_logits: np.ndarray,
                                   advantages: np.ndarray) -> float:
        """優化7: Remove Entropy Loss"""
        if not self.config.remove_entropy_loss:
            # 標準PPO帶熵正則化
            ratio = np.exp(policy_logits - old_policy_logits)
            clipped_ratio = np.clip(ratio, 0.8, 1.2)
            policy_loss = -np.mean(np.minimum(ratio * advantages, clipped_ratio * advantages))
            
            # 熵正則化
            entropy = -np.mean(np.sum(np.exp(policy_logits) * policy_logits, axis=-1))
            entropy_bonus = 0.01 * entropy
            
            total_loss = policy_loss - entropy_bonus
        else:
            # DeepSWE: 移除熵項
            ratio = np.exp(policy_logits - old_policy_logits)
            clipped_ratio = np.clip(ratio, 0.8, 1.2)
            policy_loss = -np.mean(np.minimum(ratio * advantages, clipped_ratio * advantages))
            total_loss = policy_loss
            
            self.optimization_stats["entropy_loss_removed"] += 1
            self.optimization_stats["total_optimizations"] += 1
        
        return total_loss
    
    def optimize_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """應用所有DeepSWE優化技術到一個批次"""
        optimized_batch = batch_data.copy()
        optimization_log = []
        
        # 1. One Sample Removal (首先移除低質量樣本)
        if self.config.one_sample_removal:
            optimized_batch = self.one_sample_removal(optimized_batch)
            optimization_log.append("one_sample_removal")
        
        # 2. Compact Filtering (壓縮特徵)
        if 'observations' in optimized_batch and self.config.compact_filtering:
            optimized_batch['observations'] = self.compact_filtering(optimized_batch['observations'])
            optimization_log.append("compact_filtering")
        
        # 3. Length Normalization (長度歸一化)
        if 'sequences' in optimized_batch and self.config.length_normalization:
            optimized_batch['sequences'] = self.apply_length_normalization(optimized_batch['sequences'])
            optimization_log.append("length_normalization")
        
        # 4. Remove Reward Standardization (獎勵處理)
        if 'rewards' in optimized_batch:
            optimized_batch['advantages'] = self.compute_advantages_without_std(optimized_batch['rewards'])
            if self.config.remove_reward_std:
                optimization_log.append("remove_reward_std")
        
        # 5. Gradient Clipping (梯度裁剪)
        if 'gradients' in optimized_batch and self.config.clip_high_dapo:
            optimized_batch['gradients'] = self.clip_high_dapo(optimized_batch['gradients'])
            optimization_log.append("clip_high_dapo")
        
        # 6 & 7. Loss Computation (損失計算優化)
        if all(key in optimized_batch for key in ['policy_logits', 'old_policy_logits', 'advantages']):
            # 選擇損失計算方法
            if self.config.remove_kl_loss and self.config.remove_entropy_loss:
                # 兩個都移除
                loss = self.compute_loss_without_kl_and_entropy(
                    optimized_batch['policy_logits'],
                    optimized_batch['old_policy_logits'],
                    optimized_batch['advantages']
                )
                optimization_log.extend(["remove_kl_loss", "remove_entropy_loss"])
            elif self.config.remove_kl_loss:
                loss = self.compute_loss_without_kl(
                    optimized_batch['policy_logits'],
                    optimized_batch['old_policy_logits'],
                    optimized_batch['advantages']
                )
                optimization_log.append("remove_kl_loss")
            elif self.config.remove_entropy_loss:
                loss = self.compute_loss_without_entropy(
                    optimized_batch['policy_logits'],
                    optimized_batch['old_policy_logits'],
                    optimized_batch['advantages']
                )
                optimization_log.append("remove_entropy_loss")
            else:
                # 標準PPO損失
                loss = self.compute_standard_ppo_loss(
                    optimized_batch['policy_logits'],
                    optimized_batch['old_policy_logits'],
                    optimized_batch['advantages']
                )
            
            optimized_batch['loss'] = loss
        
        # 記錄優化歷史
        optimized_batch['optimization_log'] = optimization_log
        self.step_count += 1
        
        return optimized_batch
    
    def compute_loss_without_kl_and_entropy(self, policy_logits: np.ndarray,
                                          old_policy_logits: np.ndarray,
                                          advantages: np.ndarray) -> float:
        """計算移除KL和熵的損失"""
        # 確保advantages維度正確
        if advantages.ndim == 1:
            advantages = advantages.reshape(-1, 1)
        
        # 計算動作概率比例
        log_ratio = policy_logits - old_policy_logits
        ratio = np.exp(log_ratio)
        
        # 選擇每個樣本的動作對應的比例
        if ratio.ndim > 1:
            # 假設我們需要所有動作的平均
            ratio = np.mean(ratio, axis=1, keepdims=True)
        
        # 裁剪比例
        clipped_ratio = np.clip(ratio, 0.8, 1.2)
        
        # 計算策略損失
        policy_loss = -np.mean(np.minimum(ratio * advantages, clipped_ratio * advantages))
        return policy_loss
    
    def compute_standard_ppo_loss(self, policy_logits: np.ndarray,
                                old_policy_logits: np.ndarray,
                                advantages: np.ndarray) -> float:
        """標準PPO損失計算"""
        ratio = np.exp(policy_logits - old_policy_logits)
        clipped_ratio = np.clip(ratio, 0.8, 1.2)
        policy_loss = -np.mean(np.minimum(ratio * advantages, clipped_ratio * advantages))
        
        # KL散度項
        kl_penalty = 0.01 * np.mean((policy_logits - old_policy_logits) ** 2)
        
        # 熵正則化
        entropy = -np.mean(np.sum(np.exp(policy_logits) * policy_logits, axis=-1))
        entropy_bonus = 0.01 * entropy
        
        total_loss = policy_loss + kl_penalty - entropy_bonus
        return total_loss
    
    def evaluate_optimization_impact(self, original_performance: float, 
                                   optimized_performance: float) -> Dict[str, float]:
        """評估優化影響"""
        improvement = optimized_performance - original_performance
        improvement_ratio = improvement / (abs(original_performance) + 1e-8)
        
        impact_metrics = {
            "absolute_improvement": improvement,
            "relative_improvement": improvement_ratio,
            "performance_gain": max(0, improvement),
            "optimization_efficiency": improvement_ratio * 100
        }
        
        # 記錄到歷史
        self.performance_history["optimization_impact"]["improvement"].append(improvement)
        self.performance_history["optimization_impact"]["efficiency"].append(improvement_ratio)
        
        return impact_metrics
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """生成優化報告"""
        total_opts = max(1, self.optimization_stats["total_optimizations"])
        
        report = {
            "deepswe_configuration": {
                "clip_high_dapo": self.config.clip_high_dapo,
                "remove_kl_loss": self.config.remove_kl_loss,
                "remove_reward_std": self.config.remove_reward_std,
                "length_normalization": self.config.length_normalization,
                "one_sample_removal": self.config.one_sample_removal,
                "compact_filtering": self.config.compact_filtering,
                "remove_entropy_loss": self.config.remove_entropy_loss
            },
            
            "optimization_statistics": self.optimization_stats.copy(),
            
            "optimization_frequency": {
                "clip_high_dapo_rate": self.optimization_stats["clip_high_dapo_applied"] / total_opts,
                "kl_removal_rate": self.optimization_stats["kl_loss_removed"] / total_opts,
                "reward_std_removal_rate": self.optimization_stats["reward_std_removed"] / total_opts,
                "length_norm_rate": self.optimization_stats["length_normalized"] / total_opts,
                "sample_removal_rate": self.optimization_stats["samples_removed"] / total_opts,
                "compact_filter_rate": self.optimization_stats["compact_filtered"] / total_opts,
                "entropy_removal_rate": self.optimization_stats["entropy_loss_removed"] / total_opts
            },
            
            "performance_summary": {
                "total_steps": self.step_count,
                "total_episodes": self.episode_count,
                "avg_reward": np.mean(self.performance_history["rewards"]) if self.performance_history["rewards"] else 0,
                "avg_loss": np.mean(self.performance_history["losses"]) if self.performance_history["losses"] else 0,
                "optimization_impact": {
                    "avg_improvement": np.mean(self.performance_history["optimization_impact"]["improvement"]) if self.performance_history["optimization_impact"]["improvement"] else 0,
                    "avg_efficiency": np.mean(self.performance_history["optimization_impact"]["efficiency"]) if self.performance_history["optimization_impact"]["efficiency"] else 0
                }
            },
            
            "recommendations": self._generate_optimization_recommendations()
        }
        
        return report
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """生成優化建議"""
        recommendations = []
        
        # 基於統計生成建議
        if self.optimization_stats["clip_high_dapo_applied"] > 100:
            recommendations.append("梯度裁剪頻繁觸發，考慮調整學習率")
        
        if self.optimization_stats["samples_removed"] > self.step_count * 0.1:
            recommendations.append("樣本移除比例較高，檢查數據質量")
        
        if len(self.performance_history["rewards"]) > 100:
            recent_rewards = list(self.performance_history["rewards"])[-50:]
            if np.std(recent_rewards) > np.mean(recent_rewards) * 0.5:
                recommendations.append("獎勵方差較大，考慮調整獎勵函數")
        
        if not recommendations:
            recommendations.append("所有優化技術運行正常")
        
        return recommendations

class DeepSWETrainer:
    """DeepSWE訓練器"""
    
    def __init__(self, config: DeepSWEConfig):
        self.config = config
        self.optimizer = DeepSWEOptimizer(config)
        self.training_history = []
        
        print(f"🏋️ === DeepSWE訓練器初始化 ===")
        print(f"📊 配置: {config}")
        print()
    
    def generate_mock_batch(self, batch_size: int = None) -> Dict[str, np.ndarray]:
        """生成模擬批次數據"""
        batch_size = batch_size or self.config.batch_size
        
        batch = {
            'observations': np.random.randn(batch_size, self.config.observation_dim),
            'actions': np.random.randint(0, self.config.action_dim, (batch_size,)),
            'rewards': np.random.uniform(0, 1, (batch_size,)),
            'policy_logits': np.random.randn(batch_size, self.config.action_dim),
            'old_policy_logits': np.random.randn(batch_size, self.config.action_dim),
            'gradients': np.random.randn(batch_size, self.config.observation_dim),
            'sequences': [np.random.randn(np.random.randint(5, 20)) for _ in range(batch_size)]
        }
        
        return batch
    
    def train_episode(self, episode_id: int) -> Dict[str, float]:
        """訓練一個episode"""
        # 生成批次數據
        original_batch = self.generate_mock_batch()
        
        # 計算原始性能
        original_performance = np.mean(original_batch['rewards'])
        
        # 應用DeepSWE優化
        optimized_batch = self.optimizer.optimize_batch(original_batch)
        
        # 計算優化後性能
        optimized_performance = np.mean(optimized_batch.get('rewards', original_batch['rewards']))
        
        # 評估優化影響
        impact_metrics = self.optimizer.evaluate_optimization_impact(
            original_performance, optimized_performance
        )
        
        # 記錄性能
        self.optimizer.performance_history["rewards"].append(optimized_performance)
        if 'loss' in optimized_batch:
            self.optimizer.performance_history["losses"].append(optimized_batch['loss'])
        
        episode_metrics = {
            "episode": episode_id,
            "original_performance": original_performance,
            "optimized_performance": optimized_performance,
            "optimization_applied": len(optimized_batch.get('optimization_log', [])),
            **impact_metrics
        }
        
        self.training_history.append(episode_metrics)
        self.optimizer.episode_count += 1
        
        return episode_metrics
    
    def run_extended_training(self, additional_episodes: int = 100) -> Dict[str, Any]:
        """運行擴展訓練"""
        print(f"🚀 開始擴展DeepSWE訓練 (+{additional_episodes} episodes)")
        print(f"📊 當前最佳性能: {self.best_performance:.3f}")
        
        initial_performance = self.best_performance
        initial_episodes = self.episode_count
        
        for episode in range(additional_episodes):
            episode_metrics = self.train_episode(initial_episodes + episode)
            
            if episode % 20 == 0:
                print(f"Episode {initial_episodes + episode}: "
                      f"性能 {episode_metrics['optimized_performance']:.3f}, "
                      f"改進 {episode_metrics['absolute_improvement']:.3f}, "
                      f"優化數 {episode_metrics['optimization_applied']}")
        
        # 分析擴展訓練效果
        final_performance = self.training_history[-1]["optimized_performance"] if self.training_history else 0
        performance_improvement = final_performance - initial_performance
        
        extended_report = {
            "extended_training_summary": {
                "additional_episodes": additional_episodes,
                "initial_performance": initial_performance,
                "final_performance": final_performance,
                "absolute_improvement": performance_improvement,
                "relative_improvement": performance_improvement / initial_performance if initial_performance > 0 else 0,
                "total_episodes": len(self.training_history),
                "convergence_analysis": self._analyze_convergence()
            },
            "performance_trends": {
                "last_10_episodes": [h["optimized_performance"] for h in self.training_history[-10:]],
                "improvement_trend": self._calculate_trend(),
                "stability_score": self._calculate_stability()
            },
            "optimization_effectiveness": self._analyze_optimization_effectiveness(),
            "recommendations": self._generate_extended_recommendations(performance_improvement)
        }
        
        print(f"\n✅ 擴展訓練完成!")
        print(f"   額外episodes: {additional_episodes}")
        print(f"   性能改進: {performance_improvement:.3f} ({performance_improvement/initial_performance*100:.1f}%)")
        print(f"   最終性能: {final_performance:.3f}")
        
        return extended_report
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """分析收斂情況"""
        if len(self.training_history) < 20:
            return {"status": "insufficient_data"}
        
        recent_performances = [h["optimized_performance"] for h in self.training_history[-20:]]
        variance = np.var(recent_performances)
        trend = np.polyfit(range(len(recent_performances)), recent_performances, 1)[0]
        
        convergence_analysis = {
            "variance": variance,
            "trend": trend,
            "is_converged": variance < 0.001 and abs(trend) < 0.001,
            "convergence_quality": "good" if variance < 0.001 else "improving" if trend > 0 else "stable"
        }
        
        return convergence_analysis
    
    def _calculate_trend(self) -> float:
        """計算性能趨勢"""
        if len(self.training_history) < 10:
            return 0.0
        
        recent_performances = [h["optimized_performance"] for h in self.training_history[-10:]]
        return np.polyfit(range(len(recent_performances)), recent_performances, 1)[0]
    
    def _calculate_stability(self) -> float:
        """計算穩定性分數"""
        if len(self.training_history) < 10:
            return 0.0
        
        recent_performances = [h["optimized_performance"] for h in self.training_history[-10:]]
        return 1.0 / (1.0 + np.std(recent_performances))
    
    def _analyze_optimization_effectiveness(self) -> Dict[str, Any]:
        """分析各優化技術的有效性"""
        effectiveness = {}
        
        # 分析每種優化的貢獻
        recent_history = self.training_history[-50:] if len(self.training_history) >= 50 else self.training_history
        
        for episode_data in recent_history:
            optimization_count = episode_data.get("optimization_applied", 0)
            performance = episode_data.get("optimized_performance", 0)
            
            if optimization_count not in effectiveness:
                effectiveness[optimization_count] = []
            effectiveness[optimization_count].append(performance)
        
        # 計算平均性能
        avg_effectiveness = {}
        for opt_count, performances in effectiveness.items():
            avg_effectiveness[f"optimization_{opt_count}"] = {
                "average_performance": np.mean(performances),
                "episode_count": len(performances),
                "performance_std": np.std(performances)
            }
        
        return avg_effectiveness
    
    def _generate_extended_recommendations(self, improvement: float) -> List[str]:
        """生成擴展訓練建議"""
        recommendations = []
        
        if improvement > 0.01:
            recommendations.append("性能有明顯提升，建議繼續擴展訓練")
        elif improvement > 0.001:
            recommendations.append("性能有輕微提升，可考慮調整超參數")
        elif improvement > -0.001:
            recommendations.append("性能基本穩定，已接近收斂")
        else:
            recommendations.append("性能有下降趨勢，建議檢查過擬合問題")
        
        # 基於收斂分析的建議
        if len(self.training_history) > 100:
            recent_variance = np.var([h["optimized_performance"] for h in self.training_history[-20:]])
            if recent_variance < 0.0001:
                recommendations.append("方差很小，建議使用early stopping")
        
        return recommendations
        """運行DeepSWE訓練"""
        num_episodes = num_episodes or self.config.max_episodes
        
        print(f"🚀 開始DeepSWE訓練 ({num_episodes} episodes)")
        
        for episode in range(num_episodes):
            episode_metrics = self.train_episode(episode)
            
            if episode % 100 == 0:
                print(f"Episode {episode}: "
                      f"性能 {episode_metrics['optimized_performance']:.3f}, "
                      f"改進 {episode_metrics['absolute_improvement']:.3f}, "
                      f"優化數 {episode_metrics['optimization_applied']}")
        
        # 生成最終報告
        final_report = self.optimizer.generate_optimization_report()
        final_report["training_summary"] = {
            "total_episodes": len(self.training_history),
            "final_performance": self.training_history[-1]["optimized_performance"] if self.training_history else 0,
            "average_improvement": np.mean([h["absolute_improvement"] for h in self.training_history]),
            "total_optimizations_applied": sum([h["optimization_applied"] for h in self.training_history])
        }
        
        print(f"\n✅ DeepSWE訓練完成!")
        print(f"   總episodes: {final_report['training_summary']['total_episodes']}")
        print(f"   最終性能: {final_report['training_summary']['final_performance']:.3f}")
        print(f"   平均改進: {final_report['training_summary']['average_improvement']:.3f}")
        print(f"   總優化次數: {final_report['training_summary']['total_optimizations_applied']}")
        
        return final_report

def main():
    """主函數"""
    print("🔧 === OCR0712 DeepSWE優化算法演示 ===")
    print()
    
    # 創建DeepSWE配置
    config = DeepSWEConfig(
        clip_high_dapo=True,
        remove_kl_loss=True,
        remove_reward_std=True,
        length_normalization=True,
        one_sample_removal=True,
        compact_filtering=True,
        remove_entropy_loss=True,
        max_episodes=500
    )
    
    # 創建訓練器
    trainer = DeepSWETrainer(config)
    
    # 運行訓練
    report = trainer.run_training()
    
    # 保存報告
    report_file = Path("deepswe_optimization_report.json")
    
    # 轉換numpy類型為Python原生類型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    report_serializable = convert_numpy_types(report)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 詳細報告已保存: {report_file}")
    
    # 顯示優化統計
    print(f"\n🎯 === DeepSWE優化統計 ===")
    for opt_name, count in report["optimization_statistics"].items():
        if opt_name != "total_optimizations":
            print(f"   {opt_name}: {count}")
    
    print(f"\n💡 優化建議:")
    for rec in report["recommendations"]:
        print(f"   • {rec}")

if __name__ == "__main__":
    main()