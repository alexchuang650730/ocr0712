#!/usr/bin/env python3
"""
OCR0712 Mac MPS硬件加速優化系統
針對Apple Silicon Mac進行GPU加速和訓練效率優化
"""

import os
import sys
import json
import time
import numpy as np
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import subprocess
import psutil

@dataclass
class MPSConfig:
    """MPS配置"""
    # MPS設置
    enable_mps: bool = True
    memory_fraction: float = 0.8
    allow_tf32: bool = True
    
    # 批次優化
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    max_sequence_length: int = 512
    
    # 內存管理
    memory_optimization: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # 並行化
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # 性能調優
    compile_model: bool = True
    cache_dataset: bool = True
    prefetch_factor: int = 2
    
    # Mac特定優化
    use_metal_performance_shaders: bool = True
    optimize_for_m1_m2: bool = True
    energy_efficient_mode: bool = False

class SystemProfiler:
    """系統性能分析器"""
    
    def __init__(self):
        self.system_info = self._collect_system_info()
        self.performance_baseline = {}
        
        print(f"💻 === Mac系統性能分析器 ===")
        print(f"🖥️  系統: {self.system_info['system']}")
        print(f"🔋 處理器: {self.system_info['processor']}")
        print(f"💾 記憶體: {self.system_info['memory_gb']:.1f} GB")
        print()
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """收集系統信息"""
        info = {
            "system": platform.system(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "disk_usage": psutil.disk_usage('/').percent
        }
        
        # Mac特定信息
        if platform.system() == "Darwin":
            try:
                # 獲取Mac型號
                result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Model Name' in line:
                            info['mac_model'] = line.split(':')[1].strip()
                        elif 'Chip' in line and ('Apple' in line or 'M1' in line or 'M2' in line):
                            info['chip'] = line.split(':')[1].strip()
                        elif 'Total Number of Cores' in line:
                            info['total_cores'] = line.split(':')[1].strip()
                
                # 檢查MPS可用性
                info['mps_available'] = self._check_mps_availability()
                
            except Exception as e:
                print(f"⚠️  獲取Mac信息失敗: {e}")
                info['mac_model'] = "Unknown Mac"
                info['mps_available'] = False
        else:
            info['mps_available'] = False
        
        return info
    
    def _check_mps_availability(self) -> bool:
        """檢查MPS可用性"""
        try:
            # 嘗試導入PyTorch並檢查MPS
            import torch
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except ImportError:
            return False
        except Exception:
            return False
    
    def benchmark_cpu_performance(self) -> Dict[str, float]:
        """基準測試CPU性能"""
        print("🔄 執行CPU性能基準測試...")
        
        # 矩陣運算測試
        start_time = time.time()
        
        # 測試1: 大矩陣乘法
        matrix_size = 1000
        a = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        b = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        
        matrix_start = time.time()
        c = np.dot(a, b)
        matrix_time = time.time() - matrix_start
        
        # 測試2: FFT變換
        fft_start = time.time()
        data = np.random.randn(1024*1024).astype(np.float32)
        fft_result = np.fft.fft(data)
        fft_time = time.time() - fft_start
        
        # 測試3: 多線程計算
        multithread_start = time.time()
        results = []
        for _ in range(4):
            result = np.sum(np.random.randn(1000, 1000) ** 2)
            results.append(result)
        multithread_time = time.time() - multithread_start
        
        total_time = time.time() - start_time
        
        benchmark_results = {
            "matrix_multiplication_time": matrix_time,
            "fft_time": fft_time,
            "multithread_time": multithread_time,
            "total_benchmark_time": total_time,
            "matrix_gflops": (2 * matrix_size**3) / (matrix_time * 1e9),
            "overall_score": 100 / total_time  # 簡化評分
        }
        
        print(f"   矩陣乘法: {matrix_time:.3f}s ({benchmark_results['matrix_gflops']:.1f} GFLOPS)")
        print(f"   FFT變換: {fft_time:.3f}s")
        print(f"   多線程: {multithread_time:.3f}s")
        print(f"   整體評分: {benchmark_results['overall_score']:.1f}")
        
        return benchmark_results
    
    def benchmark_memory_performance(self) -> Dict[str, float]:
        """基準測試內存性能"""
        print("🔄 執行內存性能基準測試...")
        
        # 內存帶寬測試
        data_size = 100 * 1024 * 1024  # 100MB
        
        # 測試1: 順序讀寫
        start_time = time.time()
        data = np.random.randn(data_size // 4).astype(np.float32)
        sequential_write_time = time.time() - start_time
        
        start_time = time.time()
        total = np.sum(data)
        sequential_read_time = time.time() - start_time
        
        # 測試2: 隨機訪問
        indices = np.random.randint(0, len(data), size=len(data)//10)
        start_time = time.time()
        random_data = data[indices]
        random_access_time = time.time() - start_time
        
        # 測試3: 內存拷貝
        start_time = time.time()
        data_copy = data.copy()
        memory_copy_time = time.time() - start_time
        
        # 計算帶寬 (MB/s)
        data_size_mb = data_size / (1024 * 1024)
        
        memory_results = {
            "sequential_write_time": sequential_write_time,
            "sequential_read_time": sequential_read_time,
            "random_access_time": random_access_time,
            "memory_copy_time": memory_copy_time,
            "write_bandwidth_mb_s": data_size_mb / sequential_write_time,
            "read_bandwidth_mb_s": data_size_mb / sequential_read_time,
            "copy_bandwidth_mb_s": data_size_mb / memory_copy_time
        }
        
        print(f"   寫入帶寬: {memory_results['write_bandwidth_mb_s']:.1f} MB/s")
        print(f"   讀取帶寬: {memory_results['read_bandwidth_mb_s']:.1f} MB/s")
        print(f"   拷貝帶寬: {memory_results['copy_bandwidth_mb_s']:.1f} MB/s")
        
        return memory_results

class MPSOptimizer:
    """MPS優化器"""
    
    def __init__(self, config: MPSConfig, system_info: Dict[str, Any]):
        self.config = config
        self.system_info = system_info
        self.optimization_applied = []
        self.performance_metrics = {}
        
        print(f"⚡ === MPS硬件加速優化器 ===")
        print(f"🔧 MPS可用: {system_info.get('mps_available', False)}")
        print(f"💾 內存優化: {config.memory_optimization}")
        print(f"🚀 模型編譯: {config.compile_model}")
        print()
    
    def setup_mps_environment(self) -> bool:
        """設置MPS環境"""
        if not self.config.enable_mps:
            print("❌ MPS已禁用")
            return False
        
        if not self.system_info.get('mps_available', False):
            print("❌ MPS不可用，使用CPU模式")
            return False
        
        try:
            # 設置環境變量
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = str(self.config.memory_fraction)
            
            # 如果是M1/M2 Mac，啟用特定優化
            if self.system_info.get('chip', '').startswith('Apple'):
                os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
                os.environ['MTL_HUD_ENABLED'] = '0'  # 禁用Metal HUD以提高性能
                self.optimization_applied.append("apple_silicon_optimization")
            
            print("✅ MPS環境設置完成")
            self.optimization_applied.append("mps_environment")
            return True
            
        except Exception as e:
            print(f"❌ MPS環境設置失敗: {e}")
            return False
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """優化內存使用"""
        print("🔄 優化內存使用...")
        
        optimizations = {}
        
        # 1. 自動垃圾回收
        try:
            import gc
            gc.collect()
            optimizations["garbage_collection"] = "enabled"
        except:
            pass
        
        # 2. 設置最佳批次大小
        available_memory = self.system_info['memory_gb']
        
        # 基於可用內存調整批次大小
        if available_memory >= 32:
            optimal_batch_size = 128
        elif available_memory >= 16:
            optimal_batch_size = 64
        elif available_memory >= 8:
            optimal_batch_size = 32
        else:
            optimal_batch_size = 16
        
        # 考慮MPS內存限制
        if self.config.enable_mps and optimal_batch_size > 64:
            optimal_batch_size = 64  # MPS建議的最大批次大小
        
        self.config.batch_size = min(self.config.batch_size, optimal_batch_size)
        optimizations["optimal_batch_size"] = self.config.batch_size
        
        # 3. 梯度累積優化
        if available_memory < 16:
            self.config.gradient_accumulation_steps = max(4, self.config.gradient_accumulation_steps)
            optimizations["gradient_accumulation"] = self.config.gradient_accumulation_steps
        
        # 4. 工作進程優化
        cpu_count = self.system_info['cpu_count']
        if cpu_count >= 8:
            optimal_workers = 6
        elif cpu_count >= 4:
            optimal_workers = 4
        else:
            optimal_workers = 2
        
        self.config.num_workers = min(self.config.num_workers, optimal_workers)
        optimizations["num_workers"] = self.config.num_workers
        
        print(f"   批次大小: {optimizations['optimal_batch_size']}")
        print(f"   工作進程: {optimizations['num_workers']}")
        print(f"   梯度累積: {optimizations.get('gradient_accumulation', 'unchanged')}")
        
        self.optimization_applied.append("memory_optimization")
        return optimizations
    
    def setup_data_loading_optimization(self) -> Dict[str, Any]:
        """設置數據加載優化"""
        print("🔄 設置數據加載優化...")
        
        optimizations = {
            "pin_memory": self.config.pin_memory,
            "num_workers": self.config.num_workers,
            "persistent_workers": self.config.persistent_workers,
            "prefetch_factor": self.config.prefetch_factor
        }
        
        # Mac特定優化
        if platform.system() == "Darwin":
            # Mac上建議較少的worker數量
            optimizations["num_workers"] = min(4, optimizations["num_workers"])
            
            # 啟用預取
            optimizations["prefetch_factor"] = max(2, optimizations["prefetch_factor"])
        
        print(f"   Pin Memory: {optimizations['pin_memory']}")
        print(f"   Workers: {optimizations['num_workers']}")
        print(f"   Prefetch Factor: {optimizations['prefetch_factor']}")
        
        self.optimization_applied.append("data_loading_optimization")
        return optimizations
    
    def create_optimized_training_config(self) -> Dict[str, Any]:
        """創建優化的訓練配置"""
        print("🔄 創建優化訓練配置...")
        
        training_config = {
            # 基礎設置
            "device": "mps" if (self.config.enable_mps and self.system_info.get('mps_available')) else "cpu",
            "batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            
            # 內存優化
            "memory_optimization": self.config.memory_optimization,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "mixed_precision": self.config.mixed_precision and self.config.enable_mps,
            
            # 並行化設置
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "persistent_workers": self.config.persistent_workers,
            
            # 性能優化
            "compile_model": self.config.compile_model,
            "cache_dataset": self.config.cache_dataset,
            
            # Mac特定優化
            "use_metal_performance_shaders": self.config.use_metal_performance_shaders,
            "optimize_for_apple_silicon": self.system_info.get('chip', '').startswith('Apple'),
            "energy_efficient_mode": self.config.energy_efficient_mode
        }
        
        # 根據系統調整配置
        if self.system_info['memory_gb'] < 8:
            training_config["gradient_checkpointing"] = True
            training_config["mixed_precision"] = True
        
        print(f"   設備: {training_config['device']}")
        print(f"   批次大小: {training_config['batch_size']}")
        print(f"   混合精度: {training_config['mixed_precision']}")
        print(f"   梯度檢查點: {training_config['gradient_checkpointing']}")
        
        self.optimization_applied.append("training_config_optimization")
        return training_config
    
    def benchmark_training_performance(self, config: Dict[str, Any]) -> Dict[str, float]:
        """基準測試訓練性能"""
        print("🔄 執行訓練性能基準測試...")
        
        # 模擬訓練工作負載
        batch_size = config['batch_size']
        sequence_length = 512
        hidden_dim = 256
        
        # 模擬前向傳播
        start_time = time.time()
        
        # 創建模擬數據
        input_data = np.random.randn(batch_size, sequence_length, hidden_dim).astype(np.float32)
        weights = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        
        # 模擬多層計算
        output = input_data
        for layer in range(6):  # 6層網絡
            output = np.tanh(np.dot(output, weights))
        
        forward_time = time.time() - start_time
        
        # 模擬反向傳播
        start_time = time.time()
        
        # 簡化的梯度計算
        gradients = []
        for layer in range(6):
            grad = np.random.randn(*weights.shape).astype(np.float32)
            gradients.append(grad)
        
        backward_time = time.time() - start_time
        
        # 計算吞吐量
        total_time = forward_time + backward_time
        samples_per_second = batch_size / total_time
        tokens_per_second = (batch_size * sequence_length) / total_time
        
        performance_metrics = {
            "forward_time": forward_time,
            "backward_time": backward_time,
            "total_time": total_time,
            "samples_per_second": samples_per_second,
            "tokens_per_second": tokens_per_second,
            "memory_efficiency": 1.0 / (batch_size * sequence_length / 1000),  # 簡化指標
            "optimization_score": samples_per_second * 10  # 簡化評分
        }
        
        print(f"   前向傳播: {forward_time:.3f}s")
        print(f"   反向傳播: {backward_time:.3f}s") 
        print(f"   樣本/秒: {samples_per_second:.1f}")
        print(f"   Token/秒: {tokens_per_second:.0f}")
        
        self.performance_metrics = performance_metrics
        return performance_metrics
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """生成優化報告"""
        report = {
            "system_information": self.system_info,
            "mps_configuration": {
                "enable_mps": self.config.enable_mps,
                "memory_fraction": self.config.memory_fraction,
                "mixed_precision": self.config.mixed_precision,
                "compile_model": self.config.compile_model
            },
            "optimizations_applied": self.optimization_applied,
            "performance_metrics": self.performance_metrics,
            "recommendations": self._generate_recommendations(),
            "next_steps": [
                "實施實際模型訓練測試",
                "監控訓練過程中的內存使用",
                "調整批次大小以達到最佳性能",
                "測試不同模型架構的性能表現"
            ]
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成優化建議"""
        recommendations = []
        
        # 基於系統配置的建議
        if self.system_info['memory_gb'] < 16:
            recommendations.append("內存較少，建議啟用梯度檢查點和混合精度訓練")
        
        if not self.system_info.get('mps_available', False):
            recommendations.append("MPS不可用，考慮升級到支持MPS的Mac或使用CPU優化")
        
        if self.system_info['cpu_count'] >= 8:
            recommendations.append("多核CPU，可以考慮增加數據加載器的工作進程數")
        
        # 基於性能指標的建議
        if self.performance_metrics:
            if self.performance_metrics.get('samples_per_second', 0) < 10:
                recommendations.append("訓練速度較慢，考慮減少批次大小或啟用更多優化")
            
            if self.performance_metrics.get('memory_efficiency', 0) < 0.5:
                recommendations.append("內存效率較低，建議調整序列長度或批次大小")
        
        if not recommendations:
            recommendations.append("系統配置良好，繼續當前優化設置")
        
        return recommendations

def main():
    """主函數"""
    print("⚡ === OCR0712 Mac MPS硬件加速優化 ===")
    print()
    
    # 1. 系統性能分析
    profiler = SystemProfiler()
    
    # CPU性能基準測試
    cpu_benchmark = profiler.benchmark_cpu_performance()
    
    # 內存性能基準測試  
    memory_benchmark = profiler.benchmark_memory_performance()
    
    # 2. 創建MPS配置
    mps_config = MPSConfig(
        enable_mps=True,
        memory_optimization=True,
        compile_model=True,
        batch_size=64
    )
    
    # 3. 創建MPS優化器
    optimizer = MPSOptimizer(mps_config, profiler.system_info)
    
    # 4. 設置MPS環境
    mps_ready = optimizer.setup_mps_environment()
    
    # 5. 內存優化
    memory_optimizations = optimizer.optimize_memory_usage()
    
    # 6. 數據加載優化
    data_loading_optimizations = optimizer.setup_data_loading_optimization()
    
    # 7. 創建訓練配置
    training_config = optimizer.create_optimized_training_config()
    
    # 8. 性能基準測試
    training_performance = optimizer.benchmark_training_performance(training_config)
    
    # 9. 生成優化報告
    optimization_report = optimizer.generate_optimization_report()
    
    # 合併所有基準測試結果
    optimization_report["system_benchmarks"] = {
        "cpu_performance": cpu_benchmark,
        "memory_performance": memory_benchmark,
        "training_performance": training_performance
    }
    
    optimization_report["training_configuration"] = training_config
    optimization_report["memory_optimizations"] = memory_optimizations
    optimization_report["data_loading_optimizations"] = data_loading_optimizations
    
    # 保存報告
    report_file = Path("mac_mps_optimization_report.json")
    
    def convert_types(obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    report_serializable = convert_types(optimization_report)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 === 優化總結 ===")
    print(f"   MPS可用: {'✅' if profiler.system_info.get('mps_available') else '❌'}")
    print(f"   優化項目: {len(optimizer.optimization_applied)} 項")
    print(f"   CPU評分: {cpu_benchmark['overall_score']:.1f}")
    print(f"   訓練速度: {training_performance['samples_per_second']:.1f} 樣本/秒")
    print(f"   建議批次大小: {training_config['batch_size']}")
    
    print(f"\n📄 詳細報告已保存: {report_file}")
    
    print(f"\n💡 優化建議:")
    for rec in optimization_report["recommendations"]:
        print(f"   • {rec}")
    
    print(f"\n🚀 下一步:")
    for step in optimization_report["next_steps"]:
        print(f"   • {step}")

if __name__ == "__main__":
    main()