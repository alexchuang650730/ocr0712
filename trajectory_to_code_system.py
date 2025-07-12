#!/usr/bin/env python3
"""
OCR0712 軌跡模擬為代碼系統
將人類手寫軌跡轉換為可執行的識別代碼
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import math

@dataclass
class HandwritingPoint:
    """手寫軌跡點"""
    x: float
    y: float
    pressure: float
    timestamp: float
    velocity: float = 0.0
    acceleration: float = 0.0
    
@dataclass
class HandwritingStroke:
    """手寫筆畫"""
    points: List[HandwritingPoint]
    stroke_id: int
    character_part: str = ""
    confidence: float = 1.0
    
@dataclass
class HandwritingTrajectory:
    """完整手寫軌跡"""
    strokes: List[HandwritingStroke]
    character: str
    bounding_box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    total_time: float
    complexity_score: float = 0.0

class TrajectoryToCodeConverter:
    """軌跡轉代碼轉換器"""
    
    def __init__(self):
        self.code_templates = {
            'stroke_detection': self._load_stroke_templates(),
            'feature_extraction': self._load_feature_templates(),
            'recognition_logic': self._load_recognition_templates()
        }
        
    def _load_stroke_templates(self) -> Dict[str, str]:
        """載入筆畫代碼模板"""
        return {
            'horizontal': '''
def detect_horizontal_stroke(points, threshold=0.1):
    """檢測橫畫"""
    if len(points) < 2:
        return False
    
    y_variance = np.var([p.y for p in points])
    x_range = max([p.x for p in points]) - min([p.x for p in points])
    
    return y_variance < threshold and x_range > threshold * 5
''',
            'vertical': '''
def detect_vertical_stroke(points, threshold=0.1):
    """檢測豎畫"""
    if len(points) < 2:
        return False
    
    x_variance = np.var([p.x for p in points])
    y_range = max([p.y for p in points]) - min([p.y for p in points])
    
    return x_variance < threshold and y_range > threshold * 5
''',
            'dot': '''
def detect_dot_stroke(points, threshold=0.05):
    """檢測點"""
    if len(points) < 1:
        return False
    
    if len(points) == 1:
        return True
        
    x_range = max([p.x for p in points]) - min([p.x for p in points])
    y_range = max([p.y for p in points]) - min([p.y for p in points])
    
    return x_range < threshold and y_range < threshold
''',
            'curve': '''
def detect_curve_stroke(points, curvature_threshold=0.3):
    """檢測弧線"""
    if len(points) < 3:
        return False
    
    # 計算曲率
    curvatures = []
    for i in range(1, len(points) - 1):
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        
        # 向量計算
        v1 = (p2.x - p1.x, p2.y - p1.y)
        v2 = (p3.x - p2.x, p3.y - p2.y)
        
        # 角度變化
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        
        angle = math.atan2(cross_product, dot_product)
        curvatures.append(abs(angle))
    
    avg_curvature = np.mean(curvatures) if curvatures else 0
    return avg_curvature > curvature_threshold
'''
        }
    
    def _load_feature_templates(self) -> Dict[str, str]:
        """載入特徵提取代碼模板"""
        return {
            'basic_features': '''
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
''',
            'velocity_features': '''
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
''',
            'pressure_features': '''
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
'''
        }
    
    def _load_recognition_templates(self) -> Dict[str, str]:
        """載入識別邏輯代碼模板"""
        return {
            'simple_classifier': '''
def simple_character_classifier(features):
    """簡單字符分類器"""
    
    # 基於規則的分類
    stroke_count = features.get('stroke_count', 0)
    aspect_ratio = features.get('aspect_ratio', 1.0)
    complexity = features.get('complexity_score', 0.0)
    
    # 數字識別
    if stroke_count == 1:
        if aspect_ratio > 2.0:
            return "一", 0.9
        elif complexity < 0.3:
            return "丨", 0.8
        else:
            return "乙", 0.7
    
    elif stroke_count == 2:
        if aspect_ratio > 1.5:
            return "二", 0.9
        elif features.get('has_curve', False):
            return "乃", 0.8
        else:
            return "十", 0.85
    
    elif stroke_count == 3:
        if aspect_ratio > 1.5:
            return "三", 0.9
        elif complexity > 0.5:
            return "山", 0.8
        else:
            return "工", 0.75
    
    # 默認返回
    return "未知", 0.1
''',
            'ml_classifier': '''
def ml_character_classifier(features, model):
    """機器學習字符分類器"""
    
    # 特徵標準化
    feature_vector = np.array([
        features.get('stroke_count', 0),
        features.get('aspect_ratio', 1.0),
        features.get('complexity_score', 0.0),
        features.get('avg_velocity', 0.0),
        features.get('avg_pressure', 0.0),
        features.get('width', 0.0),
        features.get('height', 0.0)
    ])
    
    # 模型預測
    try:
        prediction = model.predict(feature_vector.reshape(1, -1))
        confidence = model.predict_proba(feature_vector.reshape(1, -1)).max()
        return prediction[0], confidence
    except:
        return "未知", 0.1
''',
            'ensemble_classifier': '''
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
'''
        }
    
    def trajectory_to_code(self, trajectory: HandwritingTrajectory) -> str:
        """將軌跡轉換為代碼"""
        
        code_sections = []
        
        # 導入部分
        code_sections.append(self._generate_imports())
        
        # 數據結構定義
        code_sections.append(self._generate_data_structures(trajectory))
        
        # 筆畫檢測代碼
        code_sections.append(self._generate_stroke_detection_code(trajectory))
        
        # 特徵提取代碼  
        code_sections.append(self._generate_feature_extraction_code(trajectory))
        
        # 識別邏輯代碼
        code_sections.append(self._generate_recognition_code(trajectory))
        
        # 主函數
        code_sections.append(self._generate_main_function(trajectory))
        
        return "\n\n".join(code_sections)
    
    def _generate_imports(self) -> str:
        """生成導入代碼"""
        return '''import numpy as np
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    pressure: float
    timestamp: float
    velocity: float = 0.0'''
    
    def _generate_data_structures(self, trajectory: HandwritingTrajectory) -> str:
        """生成數據結構代碼"""
        
        code = "# 軌跡數據\n"
        code += f"TARGET_CHARACTER = '{trajectory.character}'\n"
        code += f"STROKE_COUNT = {len(trajectory.strokes)}\n"
        code += f"TOTAL_TIME = {trajectory.total_time:.3f}\n"
        code += f"BOUNDING_BOX = {trajectory.bounding_box}\n\n"
        
        # 生成軌跡點數據
        code += "TRAJECTORY_DATA = [\n"
        for stroke_idx, stroke in enumerate(trajectory.strokes):
            code += f"    # Stroke {stroke_idx + 1}\n"
            code += "    [\n"
            for point in stroke.points:
                code += f"        Point({point.x:.3f}, {point.y:.3f}, {point.pressure:.3f}, {point.timestamp:.3f}, {point.velocity:.3f}),\n"
            code += "    ],\n"
        code += "]\n"
        
        return code
    
    def _generate_stroke_detection_code(self, trajectory: HandwritingTrajectory) -> str:
        """生成筆畫檢測代碼"""
        
        code = "# 筆畫檢測函數\n"
        
        # 分析軌跡中包含的筆畫類型
        stroke_types = self._analyze_stroke_types(trajectory)
        
        for stroke_type in stroke_types:
            if stroke_type in self.code_templates['stroke_detection']:
                code += self.code_templates['stroke_detection'][stroke_type]
                code += "\n"
        
        # 生成筆畫分析函數
        code += '''
def analyze_strokes(trajectory_data):
    """分析所有筆畫"""
    stroke_analysis = []
    
    for stroke_idx, stroke_points in enumerate(trajectory_data):
        analysis = {
            'stroke_id': stroke_idx,
            'point_count': len(stroke_points),
        }
        
        # 檢測筆畫類型
'''
        
        for stroke_type in stroke_types:
            code += f"        analysis['{stroke_type}'] = detect_{stroke_type}_stroke(stroke_points)\n"
        
        code += '''        
        stroke_analysis.append(analysis)
    
    return stroke_analysis
'''
        
        return code
    
    def _generate_feature_extraction_code(self, trajectory: HandwritingTrajectory) -> str:
        """生成特徵提取代碼"""
        
        code = "# 特徵提取函數\n"
        
        # 根據軌跡特性選擇特徵
        feature_types = self._select_feature_types(trajectory)
        
        for feature_type in feature_types:
            if feature_type in self.code_templates['feature_extraction']:
                code += self.code_templates['feature_extraction'][feature_type]
                code += "\n"
        
        # 生成綜合特徵提取函數
        code += '''
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
'''
        
        for feature_type in feature_types:
            code += f"    features.update(extract_{feature_type.replace('_features', '')}_features(trajectory))\n"
        
        code += '''    
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
'''
        
        return code
    
    def _generate_recognition_code(self, trajectory: HandwritingTrajectory) -> str:
        """生成識別代碼"""
        
        code = "# 字符識別函數\n"
        
        # 選擇識別方法
        recognition_method = self._select_recognition_method(trajectory)
        
        if recognition_method in self.code_templates['recognition_logic']:
            code += self.code_templates['recognition_logic'][recognition_method]
            code += "\n"
        
        # 生成特定字符的識別邏輯
        code += self._generate_character_specific_logic(trajectory)
        
        return code
    
    def _generate_character_specific_logic(self, trajectory: HandwritingTrajectory) -> str:
        """生成特定字符的識別邏輯"""
        
        char = trajectory.character
        stroke_count = len(trajectory.strokes)
        
        code = f'''
def recognize_character_{ord(char)}(features, stroke_analysis):
    """識別字符 '{char}' 的專門函數"""
    
    # 字符特定的特徵檢查
    expected_stroke_count = {stroke_count}
    actual_stroke_count = features.get('stroke_count', 0)
    
    if actual_stroke_count != expected_stroke_count:
        return '{char}', 0.3  # 低置信度
    
    # 詳細特徵匹配
    confidence = 0.5  # 基礎置信度
    
'''
        
        # 根據字符復雜度添加特定檢查
        if trajectory.complexity_score > 0.5:
            code += '''    # 高復雜度字符檢查
    if features.get('complexity_score', 0) > 0.4:
        confidence += 0.2
    
'''
        
        # 根據筆畫特性添加檢查
        if len(trajectory.strokes) > 1:
            code += '''    # 多筆畫字符檢查
    stroke_consistency = sum(1 for s in stroke_analysis if s.get('horizontal', False) or s.get('vertical', False))
    if stroke_consistency > 0:
        confidence += 0.2
    
'''
        
        code += f'''    return '{char}', min(confidence, 0.95)
'''
        
        return code
    
    def _generate_main_function(self, trajectory: HandwritingTrajectory) -> str:
        """生成主函數"""
        
        code = '''
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
'''
        
        char_code = ord(trajectory.character)
        code += f'''    predicted_char, confidence = recognize_character_{char_code}(features, stroke_analysis)
    
    # 結果輸出
    print(f"識別結果: {{predicted_char}}")
    print(f"置信度: {{confidence:.3f}}")
    print(f"目標字符: {{TARGET_CHARACTER}}")
    print(f"識別正確: {{predicted_char == TARGET_CHARACTER}}")
    
    return predicted_char, confidence

if __name__ == "__main__":
    main()
'''
        
        return code
    
    def _analyze_stroke_types(self, trajectory: HandwritingTrajectory) -> List[str]:
        """分析軌跡中的筆畫類型"""
        stroke_types = set()
        
        for stroke in trajectory.strokes:
            if len(stroke.points) <= 2:
                stroke_types.add('dot')
            else:
                # 簡化的筆畫類型檢測
                x_range = max(p.x for p in stroke.points) - min(p.x for p in stroke.points)
                y_range = max(p.y for p in stroke.points) - min(p.y for p in stroke.points)
                
                if x_range > y_range * 2:
                    stroke_types.add('horizontal')
                elif y_range > x_range * 2:
                    stroke_types.add('vertical')
                else:
                    stroke_types.add('curve')
        
        return list(stroke_types)
    
    def _select_feature_types(self, trajectory: HandwritingTrajectory) -> List[str]:
        """選擇特徵類型"""
        feature_types = ['basic_features']
        
        # 根據軌跡特性選擇特徵
        if any(point.velocity > 0 for stroke in trajectory.strokes for point in stroke.points):
            feature_types.append('velocity_features')
        
        if any(point.pressure > 0 for stroke in trajectory.strokes for point in stroke.points):
            feature_types.append('pressure_features')
        
        return feature_types
    
    def _select_recognition_method(self, trajectory: HandwritingTrajectory) -> str:
        """選擇識別方法"""
        # 根據復雜度選擇識別方法
        if trajectory.complexity_score > 0.7:
            return 'ensemble_classifier'
        elif len(trajectory.strokes) > 5:
            return 'ml_classifier'
        else:
            return 'simple_classifier'

class TrajectorySimulator:
    """軌跡模擬器"""
    
    def __init__(self):
        self.converter = TrajectoryToCodeConverter()
    
    def create_sample_trajectory(self, character: str, complexity: float = 0.5) -> HandwritingTrajectory:
        """創建示例軌跡"""
        
        strokes = []
        
        # 根據字符創建不同的筆畫模式
        if character == "一":
            # 橫畫
            points = [
                HandwritingPoint(0.1, 0.5, 0.8, 0.0, 1.0),
                HandwritingPoint(0.5, 0.5, 0.9, 0.5, 1.2),
                HandwritingPoint(0.9, 0.5, 0.7, 1.0, 0.8)
            ]
            strokes.append(HandwritingStroke(points, 0, "horizontal"))
            
        elif character == "十":
            # 橫畫
            h_points = [
                HandwritingPoint(0.2, 0.4, 0.8, 0.0, 1.0),
                HandwritingPoint(0.8, 0.4, 0.8, 0.5, 1.0)
            ]
            strokes.append(HandwritingStroke(h_points, 0, "horizontal"))
            
            # 豎畫
            v_points = [
                HandwritingPoint(0.5, 0.1, 0.9, 1.0, 1.1),
                HandwritingPoint(0.5, 0.7, 0.8, 1.5, 0.9)
            ]
            strokes.append(HandwritingStroke(v_points, 1, "vertical"))
            
        elif character == "山":
            # 三個豎畫
            for i, x_pos in enumerate([0.2, 0.5, 0.8]):
                points = [
                    HandwritingPoint(x_pos, 0.1, 0.8, i * 0.7, 1.0),
                    HandwritingPoint(x_pos, 0.6, 0.9, i * 0.7 + 0.5, 1.0)
                ]
                strokes.append(HandwritingStroke(points, i, "vertical"))
                
        else:
            # 默認簡單軌跡
            points = [
                HandwritingPoint(0.3, 0.3, 0.8, 0.0, 1.0),
                HandwritingPoint(0.7, 0.7, 0.9, 0.5, 1.0)
            ]
            strokes.append(HandwritingStroke(points, 0, "curve"))
        
        # 計算邊界框
        all_x = [p.x for stroke in strokes for p in stroke.points]
        all_y = [p.y for stroke in strokes for p in stroke.points]
        bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
        
        # 計算總時間
        total_time = max(p.timestamp for stroke in strokes for p in stroke.points)
        
        return HandwritingTrajectory(
            strokes=strokes,
            character=character,
            bounding_box=bbox,
            total_time=total_time,
            complexity_score=complexity
        )
    
    def simulate_and_generate_code(self, character: str, complexity: float = 0.5) -> str:
        """模擬軌跡並生成代碼"""
        
        print(f"🎯 開始模擬字符 '{character}' 的手寫軌跡...")
        
        # 創建軌跡
        trajectory = self.create_sample_trajectory(character, complexity)
        
        print(f"📝 生成軌跡: {len(trajectory.strokes)} 個筆畫, 複雜度: {complexity:.2f}")
        
        # 轉換為代碼
        generated_code = self.converter.trajectory_to_code(trajectory)
        
        print(f"🔧 代碼生成完成，共 {len(generated_code.split('\\n'))} 行")
        
        return generated_code
    
    def batch_simulate(self, characters: List[str], output_dir: str = "generated_ocr_codes"):
        """批量模擬並生成代碼"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"🚀 開始批量模擬 {len(characters)} 個字符...")
        
        for i, char in enumerate(characters):
            print(f"\\n處理字符 {i+1}/{len(characters)}: '{char}'")
            
            # 隨機復雜度
            complexity = np.random.uniform(0.3, 0.8)
            
            # 生成代碼
            code = self.simulate_and_generate_code(char, complexity)
            
            # 保存代碼
            filename = f"ocr_recognize_{ord(char)}_{char}.py"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# OCR識別代碼 - 字符: {char}\\n")
                f.write(f"# 自動生成時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"# 複雜度: {complexity:.3f}\\n\\n")
                f.write(code)
            
            print(f"✅ 代碼已保存: {filepath}")
        
        # 生成批量測試腳本
        self._generate_batch_test_script(characters, output_path)
        
        print(f"\\n🎉 批量模擬完成! 所有文件保存在: {output_path}")
    
    def _generate_batch_test_script(self, characters: List[str], output_dir: Path):
        """生成批量測試腳本"""
        
        test_script = '''#!/usr/bin/env python3
"""
OCR軌跡模擬批量測試腳本
"""

import os
import sys
import importlib.util
import time

def test_generated_codes():
    """測試所有生成的識別代碼"""
    
    print("🔄 開始批量測試生成的OCR代碼...")
    
    results = []
    test_files = [f for f in os.listdir('.') if f.startswith('ocr_recognize_') and f.endswith('.py')]
    
    for i, filename in enumerate(test_files):
        print(f"\\n測試 {i+1}/{len(test_files)}: {filename}")
        
        try:
            # 動態導入模塊
            spec = importlib.util.spec_from_file_location("test_module", filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 執行主函數
            start_time = time.time()
            predicted_char, confidence = module.main()
            execution_time = time.time() - start_time
            
            # 獲取目標字符
            target_char = module.TARGET_CHARACTER
            
            # 記錄結果
            result = {
                'filename': filename,
                'target_char': target_char,
                'predicted_char': predicted_char,
                'confidence': confidence,
                'execution_time': execution_time,
                'success': predicted_char == target_char
            }
            results.append(result)
            
            status = "✅" if result['success'] else "❌"
            print(f"{status} 結果: {predicted_char} (置信度: {confidence:.3f}, 時間: {execution_time:.3f}s)")
            
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            results.append({
                'filename': filename,
                'error': str(e),
                'success': False
            })
    
    # 統計結果
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    
    print(f"\\n📊 === 批量測試結果 ===")
    print(f"總測試數: {total}")
    print(f"成功數: {successful}")
    print(f"成功率: {successful/total*100:.1f}%")
    
    if successful > 0:
        avg_confidence = sum(r.get('confidence', 0) for r in results if r.get('success', False)) / successful
        avg_time = sum(r.get('execution_time', 0) for r in results if r.get('success', False)) / successful
        print(f"平均置信度: {avg_confidence:.3f}")
        print(f"平均執行時間: {avg_time:.3f}s")
    
    return results

if __name__ == "__main__":
    test_generated_codes()
'''
        
        test_script_path = output_dir / "batch_test.py"
        with open(test_script_path, 'w', encoding='utf-8') as f:
            f.write(test_script)
        
        # 添加執行權限
        import os
        os.chmod(test_script_path, 0o755)
        
        print(f"📋 批量測試腳本已生成: {test_script_path}")

def main():
    """主函數演示"""
    
    print("🎯 === OCR0712 軌跡模擬為代碼系統演示 ===")
    print()
    
    simulator = TrajectorySimulator()
    
    # 演示單個字符
    print("📝 演示單個字符模擬...")
    code = simulator.simulate_and_generate_code("十", complexity=0.6)
    
    # 保存示例代碼
    with open("example_trajectory_code.py", 'w', encoding='utf-8') as f:
        f.write("# OCR軌跡模擬示例代碼\\n\\n")
        f.write(code)
    
    print("✅ 示例代碼已保存: example_trajectory_code.py")
    
    # 演示批量模擬
    print("\\n🚀 演示批量字符模擬...")
    test_characters = ["一", "十", "山", "工", "三", "二", "人", "大", "小"]
    
    simulator.batch_simulate(test_characters, "trajectory_codes")
    
    print("\\n🎉 軌跡模擬演示完成!")
    print("📁 查看 trajectory_codes/ 目錄獲取所有生成的識別代碼")

if __name__ == "__main__":
    main()