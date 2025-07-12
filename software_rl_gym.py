#!/usr/bin/env python3
"""
OCR0712 Software-based RL Gym for Scaling RL
純軟件實現的強化學習環境，無需硬件sensor
基於圖像特徵和預測結果的軟件sensor模擬
"""

import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from enum import Enum
import random
import time

logger = logging.getLogger(__name__)

class SoftwareSensorType(Enum):
    """軟件Sensor類型"""
    VISUAL_CLARITY = "visual_clarity"           # 視覺清晰度
    STROKE_CONSISTENCY = "stroke_consistency"   # 筆畫一致性
    PATTERN_CONFIDENCE = "pattern_confidence"   # 模式置信度
    CONTEXT_COHERENCE = "context_coherence"     # 上下文連貫性
    PREDICTION_STABILITY = "prediction_stability"  # 預測穩定性
    ERROR_LIKELIHOOD = "error_likelihood"       # 錯誤可能性
    RECOGNITION_PROGRESS = "recognition_progress"  # 識別進度
    FEATURE_RICHNESS = "feature_richness"       # 特徵豐富度

@dataclass
class SoftwareSensorReading:
    """軟件Sensor讀數"""
    sensor_type: SoftwareSensorType
    value: float                # 0-1標準化值
    confidence: float           # 讀數置信度
    metadata: Dict[str, Any]    # 額外信息

class OCRGymEnvironment(gym.Env):
    """OCR強化學習Gym環境"""
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        self.config = config or {}
        
        # 動作空間：5種OCR策略 + 連續參數
        self.action_space = spaces.Box(
            low=np.array([0, 0.0, 0.0, 0.0, 0.0]),     # [strategy_id, param1, param2, param3, param4]
            high=np.array([4, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # 觀察空間：軟件sensor讀數 + 圖像特徵
        obs_dim = 8 + 512  # 8個軟件sensor + 512維圖像特徵
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 初始化組件
        self.software_sensors = SoftwareSensorSystem()
        self.feature_extractor = ImageFeatureExtractor()
        self.ocr_strategies = OCRStrategyExecutor()
        
        # 環境狀態
        self.current_image = None
        self.target_text = ""
        self.current_prediction = ""
        self.step_count = 0
        self.max_steps = 10
        self.done = False
        
        # 歷史記錄
        self.prediction_history = []
        self.sensor_history = []
        self.reward_history = []
        
        logger.info("OCR Gym Environment initialized (software-based)")
    
    def reset(self, image: Optional[torch.Tensor] = None, 
              target_text: Optional[str] = None) -> np.ndarray:
        """重置環境"""
        
        # 生成或使用提供的數據
        if image is None:
            self.current_image = self._generate_sample_image()
        else:
            self.current_image = image
            
        if target_text is None:
            self.target_text = self._generate_sample_text()
        else:
            self.target_text = target_text
        
        # 重置狀態
        self.current_prediction = ""
        self.step_count = 0
        self.done = False
        self.prediction_history = []
        self.sensor_history = []
        self.reward_history = []
        
        # 獲取初始觀察
        observation = self._get_observation()
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """執行一步動作"""
        
        if self.done:
            return self._get_observation(), 0.0, True, {}
        
        # 解析動作
        strategy_id = int(np.clip(action[0], 0, 4))
        parameters = {
            'weight1': float(action[1]),
            'weight2': float(action[2]), 
            'threshold': float(action[3]),
            'context_weight': float(action[4])
        }
        
        # 執行OCR策略
        ocr_result = self.ocr_strategies.execute(
            strategy_id, self.current_image, parameters
        )
        
        # 更新預測
        self.current_prediction = ocr_result['text']
        self.prediction_history.append(self.current_prediction)
        
        # 收集軟件sensor數據
        sensor_readings = self.software_sensors.collect_readings(
            self.current_image, self.current_prediction, 
            self.target_text, self.prediction_history
        )
        self.sensor_history.append(sensor_readings)
        
        # 計算獎勵
        reward = self._calculate_reward(ocr_result, sensor_readings)
        self.reward_history.append(reward)
        
        # 更新狀態
        self.step_count += 1
        self.done = self._check_done(ocr_result, sensor_readings)
        
        # 準備返回信息
        observation = self._get_observation()
        info = {
            'strategy_used': strategy_id,
            'ocr_result': ocr_result,
            'sensor_readings': sensor_readings,
            'step_count': self.step_count
        }
        
        return observation, reward, self.done, info
    
    def _get_observation(self) -> np.ndarray:
        """獲取當前觀察"""
        
        # 提取圖像特徵
        image_features = self.feature_extractor.extract(self.current_image)
        
        # 收集軟件sensor讀數
        sensor_readings = self.software_sensors.collect_readings(
            self.current_image, self.current_prediction,
            self.target_text, self.prediction_history
        )
        
        # 組合觀察向量
        sensor_values = [reading.value for reading in sensor_readings.values()]
        observation = np.concatenate([sensor_values, image_features.numpy()])
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self, ocr_result: Dict, 
                         sensor_readings: Dict[SoftwareSensorType, SoftwareSensorReading]) -> float:
        """計算獎勵"""
        
        rewards = []
        
        # 1. 準確率獎勵 (主要)
        accuracy = self._calculate_accuracy(ocr_result['text'], self.target_text)
        rewards.append(accuracy * 0.4)
        
        # 2. 軟件sensor獎勵
        # 視覺清晰度獎勵
        clarity_reward = sensor_readings[SoftwareSensorType.VISUAL_CLARITY].value * 0.1
        rewards.append(clarity_reward)
        
        # 預測穩定性獎勵
        stability_reward = sensor_readings[SoftwareSensorType.PREDICTION_STABILITY].value * 0.1
        rewards.append(stability_reward)
        
        # 上下文連貫性獎勵
        coherence_reward = sensor_readings[SoftwareSensorType.CONTEXT_COHERENCE].value * 0.1
        rewards.append(coherence_reward)
        
        # 錯誤可能性懲罰
        error_penalty = sensor_readings[SoftwareSensorType.ERROR_LIKELIHOOD].value * (-0.1)
        rewards.append(error_penalty)
        
        # 3. 置信度獎勵
        confidence_reward = ocr_result['confidence'] * 0.1
        rewards.append(confidence_reward)
        
        # 4. 進度獎勵
        progress_reward = sensor_readings[SoftwareSensorType.RECOGNITION_PROGRESS].value * 0.1
        rewards.append(progress_reward)
        
        # 5. 效率獎勵 (更少步驟更好)
        efficiency_reward = max(0, (self.max_steps - self.step_count) / self.max_steps * 0.1)
        rewards.append(efficiency_reward)
        
        total_reward = sum(rewards)
        return total_reward
    
    def _calculate_accuracy(self, prediction: str, target: str) -> float:
        """計算準確率"""
        if not target:
            return 0.0
        
        if prediction == target:
            return 1.0
        
        # 使用編輯距離計算相似度
        import difflib
        similarity = difflib.SequenceMatcher(None, prediction, target).ratio()
        return similarity
    
    def _check_done(self, ocr_result: Dict, 
                   sensor_readings: Dict[SoftwareSensorType, SoftwareSensorReading]) -> bool:
        """檢查是否完成"""
        
        # 達到最大步數
        if self.step_count >= self.max_steps:
            return True
        
        # 高準確率且高置信度
        accuracy = self._calculate_accuracy(ocr_result['text'], self.target_text)
        confidence = ocr_result['confidence']
        
        if accuracy > 0.95 and confidence > 0.9:
            return True
        
        return False
    
    def _generate_sample_image(self) -> torch.Tensor:
        """生成示例圖像"""
        return torch.randn(3, 224, 224)
    
    def _generate_sample_text(self) -> str:
        """生成示例文本"""
        samples = ["手寫文字", "測試樣本", "識別目標", "Sample Text", "目標字符"]
        return random.choice(samples)

class SoftwareSensorSystem:
    """軟件Sensor系統"""
    
    def __init__(self):
        self.sensors = {
            SoftwareSensorType.VISUAL_CLARITY: VisualClaritySensor(),
            SoftwareSensorType.STROKE_CONSISTENCY: StrokeConsistencySensor(),
            SoftwareSensorType.PATTERN_CONFIDENCE: PatternConfidenceSensor(),
            SoftwareSensorType.CONTEXT_COHERENCE: ContextCoherenceSensor(),
            SoftwareSensorType.PREDICTION_STABILITY: PredictionStabilitySensor(),
            SoftwareSensorType.ERROR_LIKELIHOOD: ErrorLikelihoodSensor(),
            SoftwareSensorType.RECOGNITION_PROGRESS: RecognitionProgressSensor(),
            SoftwareSensorType.FEATURE_RICHNESS: FeatureRichnessSensor(),
        }
    
    def collect_readings(self, image: torch.Tensor, current_prediction: str,
                        target_text: str, prediction_history: List[str]) -> Dict[SoftwareSensorType, SoftwareSensorReading]:
        """收集所有軟件sensor讀數"""
        
        readings = {}
        context = {
            'image': image,
            'current_prediction': current_prediction,
            'target_text': target_text,
            'prediction_history': prediction_history
        }
        
        for sensor_type, sensor in self.sensors.items():
            try:
                reading = sensor.read(context)
                readings[sensor_type] = reading
            except Exception as e:
                logger.warning(f"Failed to read from {sensor_type}: {e}")
                readings[sensor_type] = SoftwareSensorReading(
                    sensor_type=sensor_type,
                    value=0.5,  # 默認中性值
                    confidence=0.0,
                    metadata={}
                )
        
        return readings

class BaseSoftwareSensor:
    """基礎軟件Sensor"""
    
    def __init__(self, sensor_type: SoftwareSensorType):
        self.sensor_type = sensor_type
        
    def read(self, context: Dict) -> SoftwareSensorReading:
        raise NotImplementedError

class VisualClaritySensor(BaseSoftwareSensor):
    """視覺清晰度Sensor"""
    
    def __init__(self):
        super().__init__(SoftwareSensorType.VISUAL_CLARITY)
    
    def read(self, context: Dict) -> SoftwareSensorReading:
        image = context['image']
        
        # 計算圖像清晰度 (Laplacian方差)
        if image.dim() == 3:
            gray = torch.mean(image, dim=0)
        else:
            gray = image
        
        # 轉換為numpy進行OpenCV處理
        gray_np = gray.detach().cpu().numpy().astype(np.uint8)
        laplacian_var = cv2.Laplacian(gray_np, cv2.CV_64F).var()
        
        # 標準化到 [0, 1]
        clarity = min(1.0, laplacian_var / 1000.0)
        
        return SoftwareSensorReading(
            sensor_type=self.sensor_type,
            value=clarity,
            confidence=0.9,
            metadata={'laplacian_variance': laplacian_var}
        )

class StrokeConsistencySensor(BaseSoftwareSensor):
    """筆畫一致性Sensor"""
    
    def __init__(self):
        super().__init__(SoftwareSensorType.STROKE_CONSISTENCY)
    
    def read(self, context: Dict) -> SoftwareSensorReading:
        image = context['image']
        
        # 提取邊緣並分析筆畫一致性
        if image.dim() == 3:
            gray = torch.mean(image, dim=0)
        else:
            gray = image
        
        gray_np = gray.detach().cpu().numpy().astype(np.uint8)
        edges = cv2.Canny(gray_np, 50, 150)
        
        # 計算筆畫寬度一致性
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 計算筆畫寬度變化
            widths = []
            for contour in contours:
                if len(contour) > 10:
                    # 簡化的寬度計算
                    rect = cv2.boundingRect(contour)
                    width = rect[2]
                    widths.append(width)
            
            if widths:
                width_std = np.std(widths)
                consistency = max(0.0, 1.0 - width_std / 50.0)  # 標準化
            else:
                consistency = 0.5
        else:
            consistency = 0.3
        
        return SoftwareSensorReading(
            sensor_type=self.sensor_type,
            value=consistency,
            confidence=0.7,
            metadata={'num_contours': len(contours)}
        )

class PatternConfidenceSensor(BaseSoftwareSensor):
    """模式置信度Sensor"""
    
    def __init__(self):
        super().__init__(SoftwareSensorType.PATTERN_CONFIDENCE)
        # 簡化的模式檢測器
        self.pattern_detector = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 1),
            nn.Sigmoid()
        )
    
    def read(self, context: Dict) -> SoftwareSensorReading:
        image = context['image']
        
        # 轉換為灰度並預測模式置信度
        if image.dim() == 3:
            gray = torch.mean(image, dim=0, keepdim=True).unsqueeze(0)
        else:
            gray = image.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            confidence = self.pattern_detector(gray).item()
        
        return SoftwareSensorReading(
            sensor_type=self.sensor_type,
            value=confidence,
            confidence=0.8,
            metadata={'pattern_score': confidence}
        )

class ContextCoherenceSensor(BaseSoftwareSensor):
    """上下文連貫性Sensor"""
    
    def __init__(self):
        super().__init__(SoftwareSensorType.CONTEXT_COHERENCE)
    
    def read(self, context: Dict) -> SoftwareSensorReading:
        current_prediction = context['current_prediction']
        prediction_history = context['prediction_history']
        
        if not prediction_history or len(prediction_history) < 2:
            coherence = 0.5
        else:
            # 計算與歷史預測的連貫性
            similarities = []
            for historical_pred in prediction_history[-3:]:  # 最近3個
                if historical_pred and current_prediction:
                    import difflib
                    sim = difflib.SequenceMatcher(
                        None, current_prediction, historical_pred
                    ).ratio()
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                # 適度相似性最好
                if 0.3 <= avg_similarity <= 0.7:
                    coherence = 0.9
                else:
                    coherence = 0.6
            else:
                coherence = 0.5
        
        return SoftwareSensorReading(
            sensor_type=self.sensor_type,
            value=coherence,
            confidence=0.75,
            metadata={'prediction_count': len(prediction_history)}
        )

class PredictionStabilitySensor(BaseSoftwareSensor):
    """預測穩定性Sensor"""
    
    def __init__(self):
        super().__init__(SoftwareSensorType.PREDICTION_STABILITY)
    
    def read(self, context: Dict) -> SoftwareSensorReading:
        prediction_history = context['prediction_history']
        
        if len(prediction_history) < 3:
            stability = 0.5
        else:
            # 計算最近幾次預測的穩定性
            recent_predictions = prediction_history[-3:]
            unique_predictions = len(set(recent_predictions))
            
            # 預測越穩定（變化越少）越好
            stability = 1.0 - (unique_predictions - 1) / (len(recent_predictions) - 1)
        
        return SoftwareSensorReading(
            sensor_type=self.sensor_type,
            value=stability,
            confidence=0.8,
            metadata={'unique_recent_predictions': unique_predictions if len(prediction_history) >= 3 else 0}
        )

class ErrorLikelihoodSensor(BaseSoftwareSensor):
    """錯誤可能性Sensor"""
    
    def __init__(self):
        super().__init__(SoftwareSensorType.ERROR_LIKELIHOOD)
    
    def read(self, context: Dict) -> SoftwareSensorReading:
        current_prediction = context['current_prediction']
        target_text = context['target_text']
        
        if not current_prediction or not target_text:
            error_likelihood = 0.8  # 高錯誤可能性
        else:
            # 基於編輯距離計算錯誤可能性
            import difflib
            similarity = difflib.SequenceMatcher(
                None, current_prediction, target_text
            ).ratio()
            error_likelihood = 1.0 - similarity
        
        return SoftwareSensorReading(
            sensor_type=self.sensor_type,
            value=error_likelihood,
            confidence=0.9,
            metadata={'similarity_to_target': 1.0 - error_likelihood}
        )

class RecognitionProgressSensor(BaseSoftwareSensor):
    """識別進度Sensor"""
    
    def __init__(self):
        super().__init__(SoftwareSensorType.RECOGNITION_PROGRESS)
    
    def read(self, context: Dict) -> SoftwareSensorReading:
        prediction_history = context['prediction_history']
        target_text = context['target_text']
        
        if not prediction_history:
            progress = 0.0
        else:
            # 計算識別進度 - 與目標的相似度提升
            import difflib
            current_similarity = difflib.SequenceMatcher(
                None, prediction_history[-1], target_text
            ).ratio() if prediction_history[-1] and target_text else 0.0
            
            if len(prediction_history) > 1:
                # 與之前的比較
                prev_similarity = difflib.SequenceMatcher(
                    None, prediction_history[-2], target_text
                ).ratio() if prediction_history[-2] and target_text else 0.0
                
                progress = max(0.0, current_similarity - prev_similarity + 0.5)
            else:
                progress = current_similarity
        
        return SoftwareSensorReading(
            sensor_type=self.sensor_type,
            value=min(1.0, progress),
            confidence=0.85,
            metadata={'current_similarity': current_similarity if 'current_similarity' in locals() else 0.0}
        )

class FeatureRichnessSensor(BaseSoftwareSensor):
    """特徵豐富度Sensor"""
    
    def __init__(self):
        super().__init__(SoftwareSensorType.FEATURE_RICHNESS)
    
    def read(self, context: Dict) -> SoftwareSensorReading:
        image = context['image']
        
        # 計算圖像特徵豐富度
        if image.dim() == 3:
            gray = torch.mean(image, dim=0)
        else:
            gray = image
        
        gray_np = gray.detach().cpu().numpy().astype(np.uint8)
        
        # 多種特徵指標
        # 1. 邊緣密度
        edges = cv2.Canny(gray_np, 50, 150)
        edge_density = np.mean(edges > 0)
        
        # 2. 紋理複雜度 (標準差)
        texture_complexity = np.std(gray_np) / 255.0
        
        # 3. 對比度
        contrast = (np.max(gray_np) - np.min(gray_np)) / 255.0
        
        # 綜合特徵豐富度
        richness = (edge_density + texture_complexity + contrast) / 3.0
        
        return SoftwareSensorReading(
            sensor_type=self.sensor_type,
            value=richness,
            confidence=0.8,
            metadata={
                'edge_density': edge_density,
                'texture_complexity': texture_complexity,
                'contrast': contrast
            }
        )

class ImageFeatureExtractor:
    """圖像特徵提取器"""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 16, feature_dim)
        )
    
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """提取圖像特徵"""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            features = self.extractor(image)
        
        return features.squeeze(0)

class OCRStrategyExecutor:
    """OCR策略執行器"""
    
    def __init__(self):
        self.strategies = {
            0: self._pure_baseline_strategy,
            1: self._enhanced_confidence_strategy,
            2: self._context_aware_strategy,
            3: self._multi_scale_strategy,
            4: self._ensemble_strategy
        }
    
    def execute(self, strategy_id: int, image: torch.Tensor, 
                parameters: Dict[str, float]) -> Dict[str, Any]:
        """執行OCR策略"""
        
        if strategy_id in self.strategies:
            return self.strategies[strategy_id](image, parameters)
        else:
            return self._pure_baseline_strategy(image, parameters)
    
    def _pure_baseline_strategy(self, image: torch.Tensor, params: Dict) -> Dict:
        """純基線策略"""
        return {
            'text': f'baseline_result_{random.randint(1, 100)}',
            'confidence': random.uniform(0.6, 0.85),
            'method': 'baseline'
        }
    
    def _enhanced_confidence_strategy(self, image: torch.Tensor, params: Dict) -> Dict:
        """增強置信度策略"""
        confidence_boost = params.get('threshold', 0.5)
        return {
            'text': f'enhanced_result_{random.randint(1, 100)}',
            'confidence': min(0.99, random.uniform(0.7, 0.9) + confidence_boost * 0.1),
            'method': 'enhanced_confidence'
        }
    
    def _context_aware_strategy(self, image: torch.Tensor, params: Dict) -> Dict:
        """上下文感知策略"""
        context_weight = params.get('context_weight', 0.3)
        return {
            'text': f'context_result_{random.randint(1, 100)}',
            'confidence': random.uniform(0.75, 0.95),
            'method': 'context_aware',
            'context_weight': context_weight
        }
    
    def _multi_scale_strategy(self, image: torch.Tensor, params: Dict) -> Dict:
        """多尺度策略"""
        return {
            'text': f'multiscale_result_{random.randint(1, 100)}',
            'confidence': random.uniform(0.8, 0.95),
            'method': 'multi_scale'
        }
    
    def _ensemble_strategy(self, image: torch.Tensor, params: Dict) -> Dict:
        """集成策略"""
        weight1 = params.get('weight1', 0.5)
        weight2 = params.get('weight2', 0.5)
        
        return {
            'text': f'ensemble_result_{random.randint(1, 100)}',
            'confidence': random.uniform(0.85, 0.98),
            'method': 'ensemble',
            'weights': [weight1, weight2]
        }

def create_training_data(num_samples: int = 1000) -> List[Tuple[torch.Tensor, str]]:
    """創建訓練數據"""
    
    data = []
    sample_texts = [
        "手寫識別", "文字檢測", "深度學習", "人工智能", "機器學習",
        "神經網絡", "計算機視覺", "自然語言", "數據科學", "模式識別"
    ]
    
    for i in range(num_samples):
        # 生成隨機圖像
        image = torch.randn(3, 224, 224)
        
        # 隨機選擇目標文本
        target_text = random.choice(sample_texts)
        
        data.append((image, target_text))
    
    return data

def main():
    """主函數 - 演示軟件RL Gym"""
    
    # 創建環境
    env = OCRGymEnvironment()
    
    # 創建訓練數據
    training_data = create_training_data(100)
    
    print("=== OCR Software RL Gym Demo ===")
    
    # 運行幾個episode
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        
        # 隨機選擇訓練樣本
        image, target_text = random.choice(training_data)
        
        # 重置環境
        obs = env.reset(image, target_text)
        print(f"Target text: {target_text}")
        print(f"Initial observation shape: {obs.shape}")
        
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 5:  # 限制步數用於演示
            # 隨機動作
            action = env.action_space.sample()
            
            # 執行動作
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            step += 1
            
            print(f"Step {step}: Strategy {info['strategy_used']}, "
                  f"Prediction: {info['ocr_result']['text']}, "
                  f"Reward: {reward:.3f}")
        
        print(f"Episode {episode + 1} completed. Total reward: {total_reward:.3f}")
    
    print("\nSoftware RL Gym demonstration completed!")

if __name__ == "__main__":
    main()