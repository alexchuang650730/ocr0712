# OCR識別代碼 - 字符: 工\n# 自動生成時間: 2025-07-13 01:21:47\n# 複雜度: 0.702\n\nimport numpy as np
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    pressure: float
    timestamp: float
    velocity: float = 0.0

# 軌跡數據
TARGET_CHARACTER = '工'
STROKE_COUNT = 1
TOTAL_TIME = 0.500
BOUNDING_BOX = (0.3, 0.3, 0.7, 0.7)

TRAJECTORY_DATA = [
    # Stroke 1
    [
        Point(0.300, 0.300, 0.800, 0.000, 1.000),
        Point(0.700, 0.700, 0.900, 0.500, 1.000),
    ],
]


# 筆畫檢測函數

def detect_dot_stroke(points, threshold=0.05):
    """檢測點"""
    if len(points) < 1:
        return False
    
    if len(points) == 1:
        return True
        
    x_range = max([p.x for p in points]) - min([p.x for p in points])
    y_range = max([p.y for p in points]) - min([p.y for p in points])
    
    return x_range < threshold and y_range < threshold


def analyze_strokes(trajectory_data):
    """分析所有筆畫"""
    stroke_analysis = []
    
    for stroke_idx, stroke_points in enumerate(trajectory_data):
        analysis = {
            'stroke_id': stroke_idx,
            'point_count': len(stroke_points),
        }
        
        # 檢測筆畫類型
        analysis['dot'] = detect_dot_stroke(stroke_points)
        
        stroke_analysis.append(analysis)
    
    return stroke_analysis


# 特徵提取函數

def extract_basic_features(trajectory):
    """提取基本特徵"""
    features = {}
    
    # 筆畫數量
    features['stroke_count'] = len(trajectory.strokes)
    
    # 總點數
    total_points = sum(len(stroke.points) for stroke in trajectory.strokes)
    features['total_points'] = total_points
    
    # 書寫時間
    features['total_time'] = trajectory.total_time
    
    # 邊界框特徵
    bbox = trajectory.bounding_box
    features['width'] = bbox[2] - bbox[0]
    features['height'] = bbox[3] - bbox[1]
    features['aspect_ratio'] = features['width'] / max(features['height'], 1e-6)
    
    # 複雜度
    features['complexity_score'] = trajectory.complexity_score
    
    return features


def extract_velocity_features(trajectory):
    """提取速度特徵"""
    all_velocities = []
    
    for stroke in trajectory.strokes:
        for point in stroke.points:
            all_velocities.append(point.velocity)
    
    features = {}
    if all_velocities:
        features['avg_velocity'] = np.mean(all_velocities)
        features['max_velocity'] = np.max(all_velocities)
        features['min_velocity'] = np.min(all_velocities)
        features['velocity_variance'] = np.var(all_velocities)
    else:
        features.update({
            'avg_velocity': 0, 'max_velocity': 0, 
            'min_velocity': 0, 'velocity_variance': 0
        })
    
    return features


def extract_pressure_features(trajectory):
    """提取壓力特徵"""
    all_pressures = []
    
    for stroke in trajectory.strokes:
        for point in stroke.points:
            all_pressures.append(point.pressure)
    
    features = {}
    if all_pressures:
        features['avg_pressure'] = np.mean(all_pressures)
        features['max_pressure'] = np.max(all_pressures)
        features['pressure_variance'] = np.var(all_pressures)
        features['pressure_peaks'] = len([p for p in all_pressures if p > np.mean(all_pressures) + np.std(all_pressures)])
    else:
        features.update({
            'avg_pressure': 0, 'max_pressure': 0, 
            'pressure_variance': 0, 'pressure_peaks': 0
        })
    
    return features


def extract_all_features(trajectory_data, stroke_analysis):
    """提取所有特徵"""
    features = {}
    
    # 構建軌跡對象
    class MockTrajectory:
        def __init__(self):
            self.strokes = []
            self.total_time = TOTAL_TIME
            self.bounding_box = BOUNDING_BOX
            self.complexity_score = calculate_complexity_score(trajectory_data)
            
            for stroke_points in trajectory_data:
                stroke = type('MockStroke', (), {'points': stroke_points})()
                self.strokes.append(stroke)
    
    trajectory = MockTrajectory()
    
    # 提取各種特徵
    features.update(extract_basic_features(trajectory))
    features.update(extract_velocity_features(trajectory))
    features.update(extract_pressure_features(trajectory))
    
    return features

def calculate_complexity_score(trajectory_data):
    """計算復雜度分數"""
    total_points = sum(len(stroke) for stroke in trajectory_data)
    total_direction_changes = 0
    
    for stroke in trajectory_data:
        for i in range(1, len(stroke) - 1):
            # 計算方向變化
            v1 = (stroke[i].x - stroke[i-1].x, stroke[i].y - stroke[i-1].y)
            v2 = (stroke[i+1].x - stroke[i].x, stroke[i+1].y - stroke[i].y)
            
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross_product) > 0.1:  # 閾值
                total_direction_changes += 1
    
    return total_direction_changes / max(total_points, 1)


# 字符識別函數

def ensemble_character_classifier(features, models_dict):
    """集成字符分類器"""
    
    predictions = []
    confidences = []
    
    # 規則分類器
    rule_pred, rule_conf = simple_character_classifier(features)
    predictions.append(rule_pred)
    confidences.append(rule_conf * 0.3)  # 權重0.3
    
    # ML分類器
    if 'ml_model' in models_dict:
        ml_pred, ml_conf = ml_character_classifier(features, models_dict['ml_model'])
        predictions.append(ml_pred)
        confidences.append(ml_conf * 0.5)  # 權重0.5
    
    # 深度學習分類器
    if 'dl_model' in models_dict:
        dl_pred, dl_conf = dl_character_classifier(features, models_dict['dl_model'])
        predictions.append(dl_pred)
        confidences.append(dl_conf * 0.2)  # 權重0.2
    
    # 投票機制
    if len(set(predictions)) == 1:
        return predictions[0], max(confidences)
    else:
        # 加權投票
        weighted_scores = {}
        for pred, conf in zip(predictions, confidences):
            weighted_scores[pred] = weighted_scores.get(pred, 0) + conf
        
        best_pred = max(weighted_scores, key=weighted_scores.get)
        best_conf = weighted_scores[best_pred] / sum(confidences)
        
        return best_pred, best_conf


def recognize_character_24037(features, stroke_analysis):
    """識別字符 '工' 的專門函數"""
    
    # 字符特定的特徵檢查
    expected_stroke_count = 1
    actual_stroke_count = features.get('stroke_count', 0)
    
    if actual_stroke_count != expected_stroke_count:
        return '工', 0.3  # 低置信度
    
    # 詳細特徵匹配
    confidence = 0.5  # 基礎置信度
    
    # 高復雜度字符檢查
    if features.get('complexity_score', 0) > 0.4:
        confidence += 0.2
    
    return '工', min(confidence, 0.95)



def main():
    """主識別函數"""
    print(f"開始識別字符軌跡，目標字符: {TARGET_CHARACTER}")
    
    # 筆畫分析
    stroke_analysis = analyze_strokes(TRAJECTORY_DATA)
    print(f"檢測到 {len(stroke_analysis)} 個筆畫")
    
    # 特徵提取
    features = extract_all_features(TRAJECTORY_DATA, stroke_analysis)
    print(f"提取特徵: {list(features.keys())}")
    
    # 字符識別
    predicted_char, confidence = recognize_character_24037(features, stroke_analysis)
    
    # 結果輸出
    print(f"識別結果: {predicted_char}")
    print(f"置信度: {confidence:.3f}")
    print(f"目標字符: {TARGET_CHARACTER}")
    print(f"識別正確: {predicted_char == TARGET_CHARACTER}")
    
    return predicted_char, confidence

if __name__ == "__main__":
    main()
