#!/usr/bin/env python3
"""
OCR0712 Handwriting Sensors System
基於DeepSWE方法論的手寫識別專用Sensor系統
模擬手寫過程中的物理和認知sensors
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class SensorType(Enum):
    """Sensor類型"""
    PRESSURE = "pressure"           # 壓力sensor
    VELOCITY = "velocity"           # 速度sensor  
    ACCELERATION = "acceleration"   # 加速度sensor
    ANGLE = "angle"                # 筆尖角度sensor
    CONFIDENCE = "confidence"       # 置信度sensor
    STROKE_ORDER = "stroke_order"   # 筆順sensor
    CONTEXT = "context"            # 上下文sensor
    ERROR_DETECTION = "error_detection"  # 錯誤檢測sensor

@dataclass
class SensorReading:
    """Sensor讀數"""
    sensor_type: SensorType
    value: float
    timestamp: float
    confidence: float
    metadata: Dict

class HandwritingSensorSystem:
    """手寫Sensor系統"""
    
    def __init__(self):
        # 初始化所有sensors
        self.sensors = {
            SensorType.PRESSURE: PressureSensor(),
            SensorType.VELOCITY: VelocitySensor(),
            SensorType.ACCELERATION: AccelerationSensor(),
            SensorType.ANGLE: AngleSensor(),
            SensorType.CONFIDENCE: ConfidenceSensor(),
            SensorType.STROKE_ORDER: StrokeOrderSensor(),
            SensorType.CONTEXT: ContextSensor(),
            SensorType.ERROR_DETECTION: ErrorDetectionSensor()
        }
        
        # Sensor融合權重
        self.sensor_weights = {
            SensorType.PRESSURE: 0.15,
            SensorType.VELOCITY: 0.12,
            SensorType.ACCELERATION: 0.10,
            SensorType.ANGLE: 0.08,
            SensorType.CONFIDENCE: 0.20,
            SensorType.STROKE_ORDER: 0.15,
            SensorType.CONTEXT: 0.12,
            SensorType.ERROR_DETECTION: 0.08
        }
        
        logger.info("Handwriting Sensor System initialized with 8 sensors")
    
    def collect_readings(self, image: torch.Tensor, trajectory: np.ndarray, 
                        context: Dict) -> Dict[SensorType, SensorReading]:
        """收集所有sensor讀數"""
        
        readings = {}
        current_time = context.get('timestamp', 0.0)
        
        for sensor_type, sensor in self.sensors.items():
            try:
                reading = sensor.read(image, trajectory, context, current_time)
                readings[sensor_type] = reading
            except Exception as e:
                logger.warning(f"Failed to read from {sensor_type}: {e}")
                # 提供默認讀數
                readings[sensor_type] = SensorReading(
                    sensor_type=sensor_type,
                    value=0.0,
                    timestamp=current_time,
                    confidence=0.0,
                    metadata={}
                )
        
        return readings
    
    def fuse_sensor_data(self, readings: Dict[SensorType, SensorReading]) -> torch.Tensor:
        """融合sensor數據為特徵向量"""
        
        features = []
        
        for sensor_type, weight in self.sensor_weights.items():
            if sensor_type in readings:
                reading = readings[sensor_type]
                # 加權sensor值
                weighted_value = reading.value * weight * reading.confidence
                features.append(weighted_value)
            else:
                features.append(0.0)
        
        return torch.tensor(features, dtype=torch.float32)

class BaseSensor:
    """基礎Sensor類"""
    
    def __init__(self, sensor_type: SensorType):
        self.sensor_type = sensor_type
        self.history = []
        self.calibration_data = {}
    
    def read(self, image: torch.Tensor, trajectory: np.ndarray, 
             context: Dict, timestamp: float) -> SensorReading:
        """讀取sensor數據 - 子類需要實現"""
        raise NotImplementedError
    
    def calibrate(self, calibration_data: Dict):
        """校準sensor"""
        self.calibration_data = calibration_data
    
    def get_history(self, n: int = 10) -> List[SensorReading]:
        """獲取歷史讀數"""
        return self.history[-n:]

class PressureSensor(BaseSensor):
    """壓力Sensor - 模擬書寫壓力"""
    
    def __init__(self):
        super().__init__(SensorType.PRESSURE)
        self.pressure_model = self._build_pressure_model()
    
    def _build_pressure_model(self):
        """構建壓力預測模型"""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 壓力值 0-1
        )
    
    def read(self, image: torch.Tensor, trajectory: np.ndarray, 
             context: Dict, timestamp: float) -> SensorReading:
        """讀取壓力數據"""
        
        # 從圖像預測壓力
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            pressure = self.pressure_model(image).item()
        
        # 基於筆跡粗細調整壓力
        if len(trajectory) > 0:
            # 計算筆跡密度作為壓力指標
            stroke_density = self._calculate_stroke_density(trajectory)
            pressure = pressure * 0.7 + stroke_density * 0.3
        
        reading = SensorReading(
            sensor_type=self.sensor_type,
            value=pressure,
            timestamp=timestamp,
            confidence=0.85,
            metadata={'stroke_density': stroke_density if len(trajectory) > 0 else 0.0}
        )
        
        self.history.append(reading)
        return reading
    
    def _calculate_stroke_density(self, trajectory: np.ndarray) -> float:
        """計算筆跡密度"""
        if len(trajectory) < 2:
            return 0.0
        
        # 計算相鄰點距離
        distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
        avg_distance = np.mean(distances)
        
        # 距離越小，密度越高，暗示壓力越大
        density = max(0.0, min(1.0, 1.0 / (avg_distance + 1e-6)))
        return density

class VelocitySensor(BaseSensor):
    """速度Sensor - 模擬書寫速度"""
    
    def __init__(self):
        super().__init__(SensorType.VELOCITY)
        self.previous_trajectory = None
        self.previous_timestamp = None
    
    def read(self, image: torch.Tensor, trajectory: np.ndarray, 
             context: Dict, timestamp: float) -> SensorReading:
        """讀取速度數據"""
        
        velocity = 0.0
        confidence = 0.9
        
        if (self.previous_trajectory is not None and 
            self.previous_timestamp is not None and 
            len(trajectory) > 0):
            
            # 計算軌跡變化
            dt = timestamp - self.previous_timestamp
            if dt > 0:
                # 簡化速度計算
                if len(self.previous_trajectory) > 0:
                    displacement = np.linalg.norm(
                        trajectory[-1] - self.previous_trajectory[-1]
                    )
                    velocity = displacement / dt
                    velocity = min(velocity, 10.0)  # 限制最大速度
        
        # 從筆跡特徵推估速度
        stroke_velocity = self._estimate_stroke_velocity(trajectory)
        velocity = velocity * 0.6 + stroke_velocity * 0.4
        
        reading = SensorReading(
            sensor_type=self.sensor_type,
            value=velocity,
            timestamp=timestamp,
            confidence=confidence,
            metadata={'stroke_velocity': stroke_velocity}
        )
        
        # 更新歷史
        self.previous_trajectory = trajectory.copy() if len(trajectory) > 0 else None
        self.previous_timestamp = timestamp
        self.history.append(reading)
        
        return reading
    
    def _estimate_stroke_velocity(self, trajectory: np.ndarray) -> float:
        """從筆跡估計速度"""
        if len(trajectory) < 3:
            return 0.0
        
        # 計算曲率變化率 - 急轉彎通常意味著慢速
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            # 簡化曲率計算
            v1 = p2 - p1
            v2 = p3 - p2
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                angle = np.arccos(np.clip(np.dot(v1, v2) / 
                                        (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                curvatures.append(angle)
        
        if curvatures:
            avg_curvature = np.mean(curvatures)
            # 曲率越大，速度越慢
            estimated_velocity = max(0.1, 5.0 - avg_curvature * 2.0)
            return min(estimated_velocity, 10.0)
        
        return 2.0  # 默認中等速度

class AccelerationSensor(BaseSensor):
    """加速度Sensor"""
    
    def __init__(self):
        super().__init__(SensorType.ACCELERATION)
        self.velocity_history = []
    
    def read(self, image: torch.Tensor, trajectory: np.ndarray, 
             context: Dict, timestamp: float) -> SensorReading:
        """讀取加速度數據"""
        
        acceleration = 0.0
        
        # 從速度歷史計算加速度
        if len(self.velocity_history) >= 2:
            dt = timestamp - self.velocity_history[-1]['timestamp']
            if dt > 0:
                dv = context.get('current_velocity', 0.0) - self.velocity_history[-1]['velocity']
                acceleration = dv / dt
        
        # 從軌跡曲率變化估計加速度
        trajectory_acceleration = self._estimate_trajectory_acceleration(trajectory)
        acceleration = acceleration * 0.5 + trajectory_acceleration * 0.5
        
        reading = SensorReading(
            sensor_type=self.sensor_type,
            value=acceleration,
            timestamp=timestamp,
            confidence=0.75,
            metadata={'trajectory_acceleration': trajectory_acceleration}
        )
        
        # 更新速度歷史
        self.velocity_history.append({
            'velocity': context.get('current_velocity', 0.0),
            'timestamp': timestamp
        })
        if len(self.velocity_history) > 10:
            self.velocity_history.pop(0)
        
        self.history.append(reading)
        return reading
    
    def _estimate_trajectory_acceleration(self, trajectory: np.ndarray) -> float:
        """從軌跡估計加速度"""
        if len(trajectory) < 4:
            return 0.0
        
        # 計算二階差分近似加速度
        second_derivatives = []
        for i in range(2, len(trajectory) - 2):
            # 使用中心差分
            second_deriv = trajectory[i+2] - 2*trajectory[i] + trajectory[i-2]
            second_derivatives.append(np.linalg.norm(second_deriv))
        
        if second_derivatives:
            return np.mean(second_derivatives)
        return 0.0

class AngleSensor(BaseSensor):
    """筆尖角度Sensor"""
    
    def __init__(self):
        super().__init__(SensorType.ANGLE)
    
    def read(self, image: torch.Tensor, trajectory: np.ndarray, 
             context: Dict, timestamp: float) -> SensorReading:
        """讀取筆尖角度"""
        
        angle = self._estimate_pen_angle(trajectory)
        
        reading = SensorReading(
            sensor_type=self.sensor_type,
            value=angle,
            timestamp=timestamp,
            confidence=0.70,
            metadata={'trajectory_length': len(trajectory)}
        )
        
        self.history.append(reading)
        return reading
    
    def _estimate_pen_angle(self, trajectory: np.ndarray) -> float:
        """估計筆尖角度"""
        if len(trajectory) < 2:
            return 0.0
        
        # 計算主要方向
        if len(trajectory) >= 10:
            # 使用最近10個點計算方向
            recent_points = trajectory[-10:]
            direction = recent_points[-1] - recent_points[0]
        else:
            direction = trajectory[-1] - trajectory[0]
        
        # 計算與水平線的角度
        angle = np.arctan2(direction[1], direction[0])
        # 歸一化到 [0, 1]
        normalized_angle = (angle + np.pi) / (2 * np.pi)
        
        return normalized_angle

class ConfidenceSensor(BaseSensor):
    """置信度Sensor"""
    
    def __init__(self):
        super().__init__(SensorType.CONFIDENCE)
        self.confidence_model = self._build_confidence_model()
    
    def _build_confidence_model(self):
        """構建置信度評估模型"""
        return nn.Sequential(
            nn.Linear(10, 32),  # 輸入多個特徵
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def read(self, image: torch.Tensor, trajectory: np.ndarray, 
             context: Dict, timestamp: float) -> SensorReading:
        """讀取置信度"""
        
        # 收集多個置信度指標
        features = self._extract_confidence_features(image, trajectory, context)
        
        with torch.no_grad():
            confidence = self.confidence_model(features).item()
        
        reading = SensorReading(
            sensor_type=self.sensor_type,
            value=confidence,
            timestamp=timestamp,
            confidence=0.90,
            metadata={'features': features.tolist()}
        )
        
        self.history.append(reading)
        return reading
    
    def _extract_confidence_features(self, image: torch.Tensor, 
                                   trajectory: np.ndarray, context: Dict) -> torch.Tensor:
        """提取置信度特徵"""
        
        features = []
        
        # 1. 圖像清晰度
        if image.dim() == 4:
            image = image.squeeze(0)
        gray = torch.mean(image, dim=0)
        laplacian_var = torch.var(self._laplacian_filter(gray))
        features.append(laplacian_var.item())
        
        # 2. 軌跡平滑度
        if len(trajectory) > 2:
            smoothness = self._calculate_trajectory_smoothness(trajectory)
            features.append(smoothness)
        else:
            features.append(0.0)
        
        # 3. 歷史準確率
        historical_accuracy = context.get('historical_accuracy', 0.8)
        features.append(historical_accuracy)
        
        # 4. 上下文一致性
        context_consistency = context.get('context_consistency', 0.7)
        features.append(context_consistency)
        
        # 5-10. 其他特徵 (填充到10維)
        while len(features) < 10:
            features.append(np.random.uniform(0.5, 0.9))
        
        return torch.tensor(features[:10], dtype=torch.float32)
    
    def _laplacian_filter(self, image: torch.Tensor) -> torch.Tensor:
        """拉普拉斯濾波器"""
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                             dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return F.conv2d(image.unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze()
    
    def _calculate_trajectory_smoothness(self, trajectory: np.ndarray) -> float:
        """計算軌跡平滑度"""
        if len(trajectory) < 3:
            return 1.0
        
        # 計算二階導數的變化
        second_derivatives = []
        for i in range(1, len(trajectory) - 1):
            second_deriv = trajectory[i+1] - 2*trajectory[i] + trajectory[i-1]
            second_derivatives.append(np.linalg.norm(second_deriv))
        
        if second_derivatives:
            # 變化越小越平滑
            smoothness = 1.0 / (1.0 + np.std(second_derivatives))
            return min(1.0, smoothness)
        
        return 1.0

class StrokeOrderSensor(BaseSensor):
    """筆順Sensor"""
    
    def __init__(self):
        super().__init__(SensorType.STROKE_ORDER)
        self.stroke_patterns = self._load_stroke_patterns()
    
    def _load_stroke_patterns(self) -> Dict:
        """加載筆順模式"""
        # 簡化的筆順模式
        return {
            'horizontal': [0, 1],      # 從左到右
            'vertical': [1, 0],        # 從上到下
            'dot': [0.5, 0.5],         # 點
            'hook': [0, 1, 0.5]        # 鉤
        }
    
    def read(self, image: torch.Tensor, trajectory: np.ndarray, 
             context: Dict, timestamp: float) -> SensorReading:
        """讀取筆順數據"""
        
        stroke_order_score = self._analyze_stroke_order(trajectory)
        
        reading = SensorReading(
            sensor_type=self.sensor_type,
            value=stroke_order_score,
            timestamp=timestamp,
            confidence=0.65,
            metadata={'trajectory_length': len(trajectory)}
        )
        
        self.history.append(reading)
        return reading
    
    def _analyze_stroke_order(self, trajectory: np.ndarray) -> float:
        """分析筆順正確性"""
        if len(trajectory) < 2:
            return 0.5
        
        # 簡化分析：檢查是否符合從左到右、從上到下的基本原則
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        # 水平方向：從左到右較好
        horizontal_score = 1.0 if end_point[0] >= start_point[0] else 0.3
        
        # 垂直方向：從上到下較好
        vertical_score = 1.0 if end_point[1] >= start_point[1] else 0.7
        
        # 綜合評分
        overall_score = (horizontal_score + vertical_score) / 2.0
        
        return overall_score

class ContextSensor(BaseSensor):
    """上下文Sensor"""
    
    def __init__(self):
        super().__init__(SensorType.CONTEXT)
        self.context_memory = []
    
    def read(self, image: torch.Tensor, trajectory: np.ndarray, 
             context: Dict, timestamp: float) -> SensorReading:
        """讀取上下文數據"""
        
        context_score = self._analyze_context_consistency(context)
        
        reading = SensorReading(
            sensor_type=self.sensor_type,
            value=context_score,
            timestamp=timestamp,
            confidence=0.80,
            metadata={'context_memory_size': len(self.context_memory)}
        )
        
        # 更新上下文記憶
        self.context_memory.append(context)
        if len(self.context_memory) > 20:
            self.context_memory.pop(0)
        
        self.history.append(reading)
        return reading
    
    def _analyze_context_consistency(self, context: Dict) -> float:
        """分析上下文一致性"""
        if len(self.context_memory) == 0:
            return 0.5
        
        # 簡化分析：檢查與歷史上下文的相似性
        current_text = context.get('current_prediction', '')
        
        if not current_text:
            return 0.3
        
        # 與最近的上下文比較
        similarities = []
        for historical_context in self.context_memory[-5:]:  # 最近5個
            historical_text = historical_context.get('current_prediction', '')
            if historical_text:
                # 簡化相似度計算
                similarity = self._calculate_text_similarity(current_text, historical_text)
                similarities.append(similarity)
        
        if similarities:
            # 適度的相似性是好的
            avg_similarity = np.mean(similarities)
            if 0.2 <= avg_similarity <= 0.8:
                return 0.9  # 好的上下文一致性
            else:
                return 0.5  # 中等一致性
        
        return 0.5
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """計算文本相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 字符級相似度
        common_chars = set(text1) & set(text2)
        total_chars = set(text1) | set(text2)
        
        if total_chars:
            return len(common_chars) / len(total_chars)
        
        return 0.0

class ErrorDetectionSensor(BaseSensor):
    """錯誤檢測Sensor"""
    
    def __init__(self):
        super().__init__(SensorType.ERROR_DETECTION)
        self.error_patterns = self._load_error_patterns()
    
    def _load_error_patterns(self) -> Dict:
        """加載錯誤模式"""
        return {
            'stroke_reversal': 0.1,     # 筆畫逆序
            'missing_stroke': 0.2,      # 缺少筆畫
            'extra_stroke': 0.15,       # 多餘筆畫
            'proportion_error': 0.25,   # 比例錯誤
            'position_error': 0.3       # 位置錯誤
        }
    
    def read(self, image: torch.Tensor, trajectory: np.ndarray, 
             context: Dict, timestamp: float) -> SensorReading:
        """讀取錯誤檢測數據"""
        
        error_probability = self._detect_errors(image, trajectory, context)
        
        reading = SensorReading(
            sensor_type=self.sensor_type,
            value=error_probability,
            timestamp=timestamp,
            confidence=0.75,
            metadata={'error_types_detected': list(self.error_patterns.keys())}
        )
        
        self.history.append(reading)
        return reading
    
    def _detect_errors(self, image: torch.Tensor, trajectory: np.ndarray, 
                      context: Dict) -> float:
        """檢測錯誤概率"""
        
        error_scores = []
        
        # 1. 軌跡完整性檢查
        if len(trajectory) == 0:
            error_scores.append(0.9)  # 沒有軌跡是嚴重錯誤
        elif len(trajectory) < 5:
            error_scores.append(0.6)  # 軌跡太短
        else:
            error_scores.append(0.1)  # 軌跡長度正常
        
        # 2. 軌跡連續性檢查
        if len(trajectory) > 1:
            gaps = []
            for i in range(1, len(trajectory)):
                gap = np.linalg.norm(trajectory[i] - trajectory[i-1])
                gaps.append(gap)
            
            if gaps:
                max_gap = max(gaps)
                if max_gap > 50:  # 假設坐標範圍是0-224
                    error_scores.append(0.7)  # 軌跡有大斷裂
                else:
                    error_scores.append(0.2)
        
        # 3. 預測一致性檢查
        current_prediction = context.get('current_prediction', '')
        target_text = context.get('target_text', '')
        
        if target_text and current_prediction:
            if current_prediction == target_text:
                error_scores.append(0.05)  # 預測正確
            else:
                # 基於相似度評估錯誤程度
                similarity = self._calculate_prediction_similarity(
                    current_prediction, target_text
                )
                error_score = 1.0 - similarity
                error_scores.append(error_score)
        else:
            error_scores.append(0.5)  # 無法比較
        
        # 綜合錯誤概率
        overall_error_probability = np.mean(error_scores)
        return min(1.0, overall_error_probability)
    
    def _calculate_prediction_similarity(self, pred: str, target: str) -> float:
        """計算預測相似度"""
        if not pred or not target:
            return 0.0
        
        # 編輯距離相似度
        import difflib
        return difflib.SequenceMatcher(None, pred, target).ratio()

# 使用示例
def create_sensor_enhanced_rl_environment():
    """創建基於sensor的RL環境"""
    
    class SensorEnhancedOCREnvironment:
        def __init__(self, base_ocr_model):
            self.base_ocr_model = base_ocr_model
            self.sensor_system = HandwritingSensorSystem()
            self.sensor_history = []
        
        def step(self, action, image, trajectory, context):
            """執行動作並收集sensor數據"""
            
            # 收集sensor讀數
            sensor_readings = self.sensor_system.collect_readings(
                image, trajectory, context
            )
            
            # 融合sensor數據
            sensor_features = self.sensor_system.fuse_sensor_data(sensor_readings)
            
            # 執行OCR動作 (結合sensor數據)
            ocr_result = self._execute_sensor_guided_ocr(
                action, image, trajectory, sensor_features
            )
            
            # 計算基於sensor的獎勵
            sensor_reward = self._calculate_sensor_reward(sensor_readings, ocr_result)
            
            # 更新sensor歷史
            self.sensor_history.append(sensor_readings)
            
            return ocr_result, sensor_reward, sensor_features
        
        def _execute_sensor_guided_ocr(self, action, image, trajectory, sensor_features):
            """執行基於sensor引導的OCR"""
            # 這裡可以根據sensor數據調整OCR策略
            return {"text": "sensor_guided_result", "confidence": 0.9}
        
        def _calculate_sensor_reward(self, sensor_readings, ocr_result):
            """計算基於sensor的獎勵"""
            
            rewards = []
            
            # 各個sensor的獎勵貢獻
            confidence_score = sensor_readings[SensorType.CONFIDENCE].value
            rewards.append(confidence_score * 0.3)
            
            error_score = 1.0 - sensor_readings[SensorType.ERROR_DETECTION].value
            rewards.append(error_score * 0.2)
            
            context_score = sensor_readings[SensorType.CONTEXT].value
            rewards.append(context_score * 0.1)
            
            stroke_order_score = sensor_readings[SensorType.STROKE_ORDER].value
            rewards.append(stroke_order_score * 0.1)
            
            # 物理sensor獎勵
            pressure_score = sensor_readings[SensorType.PRESSURE].value
            velocity_score = min(1.0, sensor_readings[SensorType.VELOCITY].value / 5.0)
            
            physical_reward = (pressure_score + velocity_score) / 2 * 0.15
            rewards.append(physical_reward)
            
            return sum(rewards)
    
    return SensorEnhancedOCREnvironment

def main():
    """主函數 - 演示sensor系統"""
    
    # 創建sensor系統
    sensor_system = HandwritingSensorSystem()
    
    # 創建測試數據
    test_image = torch.randn(3, 224, 224)
    test_trajectory = np.random.rand(50, 2) * 224
    test_context = {
        'current_prediction': 'test_text',
        'target_text': 'test_text',
        'timestamp': time.time(),
        'historical_accuracy': 0.85
    }
    
    # 收集sensor讀數
    readings = sensor_system.collect_readings(test_image, test_trajectory, test_context)
    
    # 顯示結果
    print("=== Handwriting Sensor Readings ===")
    for sensor_type, reading in readings.items():
        print(f"{sensor_type.value}: {reading.value:.3f} (confidence: {reading.confidence:.3f})")
    
    # 融合sensor數據
    fused_features = sensor_system.fuse_sensor_data(readings)
    print(f"\nFused sensor features: {fused_features}")
    
    print("\nHandwriting Sensor System demonstration completed!")

if __name__ == "__main__":
    main()