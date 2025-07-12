#!/usr/bin/env python3
"""
OCR0712 大型數據集整合器
整合CASIA-HWDB、HIT-MW、SCUT-EPT等學術級中文手寫數據集
"""

import os
import sys
import json
import time
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class LargeDatasetIntegrator:
    """大型數據集整合器"""
    
    def __init__(self, base_dir: str = "./large_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 大型數據集配置
        self.dataset_configs = {
            "casia_hwdb": {
                "name": "CASIA-HWDB2.0/2.1/2.2",
                "description": "中科院自動化所中文手寫數據庫",
                "size_estimate": "10GB+",
                "character_count": "7000+ 字符",
                "sample_count": "1.2M+ 樣本",
                "urls": [
                    "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB2.0Train.zip",
                    "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB2.1Train.zip", 
                    "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB2.2Train.zip"
                ],
                "backup_info": "需要學術機構申請許可",
                "format": "Isolated Character Dataset"
            },
            
            "hit_mw": {
                "name": "HIT-MW (Harbin Institute of Technology)",
                "description": "哈工大中文手寫數據集",
                "size_estimate": "2GB+",
                "character_count": "3000+ 字符",
                "sample_count": "200K+ 樣本",
                "urls": [
                    "https://github.com/hitcszj/HIT-MW",
                    "https://dataset.hitcszj.com/HIT-MW/HIT-MW.zip"
                ],
                "backup_info": "開源數據集，GitHub可獲取",
                "format": "Word-level Dataset"
            },
            
            "scut_ept": {
                "name": "SCUT-EPT",
                "description": "華南理工English and Chinese text dataset",
                "size_estimate": "1.5GB+",
                "character_count": "2000+ 字符",
                "sample_count": "100K+ 樣本",
                "urls": [
                    "https://github.com/HCIILAB/SCUT-EPT_Dataset_Release",
                    "https://pan.baidu.com/s/1QkI7kjah8SPHwOQ40rS6Pw"
                ],
                "backup_info": "百度網盤: code: scut",
                "format": "English+Chinese mixed"
            },
            
            "chn_handwriting": {
                "name": "Chinese Handwriting Dataset Collection",
                "description": "多來源中文手寫數據集合",
                "size_estimate": "5GB+",
                "character_count": "5000+ 字符",
                "sample_count": "500K+ 樣本",
                "urls": [
                    "https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi",
                    "https://github.com/skishore/makemeahanzi/releases/download/1.0/graphics.txt",
                    "https://github.com/chanind/hanzi-writer-data"
                ],
                "backup_info": "Kaggle + GitHub開源",
                "format": "Multiple formats"
            },
            
            "calamari_chinese": {
                "name": "Calamari Chinese OCR Dataset",
                "description": "Calamari OCR框架中文數據集",
                "size_estimate": "800MB+",
                "character_count": "4000+ 字符", 
                "sample_count": "50K+ 樣本",
                "urls": [
                    "https://github.com/Calamari-OCR/calamari_models/tree/master/antiqua_historical",
                    "https://zenodo.org/record/4428683/files/chinese_traditional.zip"
                ],
                "backup_info": "Zenodo學術平台",
                "format": "OCR-ready format"
            }
        }
        
        # 下載統計
        self.download_stats = {
            "total_datasets": len(self.dataset_configs),
            "attempted_downloads": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_size_downloaded": 0,
            "estimated_character_count": 0,
            "estimated_sample_count": 0
        }
        
        print(f"🗄️ === OCR0712 大型數據集整合器 ===")
        print(f"📊 目標數據集: {len(self.dataset_configs)} 個")
        print(f"📁 基礎目錄: {self.base_dir.absolute()}")
        print()
    
    def analyze_dataset_requirements(self):
        """分析數據集需求"""
        print("📋 === 數據集需求分析 ===")
        
        total_size_estimate = 0
        total_characters = 0
        total_samples = 0
        
        for dataset_id, config in self.dataset_configs.items():
            print(f"\n📦 {config['name']}")
            print(f"   描述: {config['description']}")
            print(f"   估計大小: {config['size_estimate']}")
            print(f"   字符數: {config['character_count']}")
            print(f"   樣本數: {config['sample_count']}")
            print(f"   格式: {config['format']}")
            print(f"   獲取說明: {config['backup_info']}")
            
            # 解析數值
            size_str = config['size_estimate'].replace('GB+', '').replace('MB+', '')
            if 'GB' in config['size_estimate']:
                size_gb = float(size_str)
                total_size_estimate += size_gb
            elif 'MB' in config['size_estimate']:
                size_mb = float(size_str) / 1024
                total_size_estimate += size_mb
            
            char_count_str = config['character_count'].split('+')[0].replace('K', '000').replace(' 字符', '')
            char_count = int(char_count_str)
            
            sample_count_str = config['sample_count'].replace('K+', '000').replace('M+', '000000').replace('+', '').replace(' 樣本', '')
            sample_count = int(float(sample_count_str.split('.')[0]))
            
            total_characters += char_count
            total_samples += sample_count
        
        print(f"\n📊 === 總計需求 ===")
        print(f"   總估計大小: {total_size_estimate:.1f} GB")
        print(f"   總字符數: {total_characters:,} 個")
        print(f"   總樣本數: {total_samples:,} 個")
        print(f"   存儲空間需求: {total_size_estimate * 1.5:.1f} GB (含處理空間)")
        
        return {
            "total_size_gb": total_size_estimate,
            "total_characters": total_characters,
            "total_samples": total_samples
        }
    
    def create_dataset_download_plan(self):
        """創建數據集下載計劃"""
        print("\n🗺️ === 數據集下載計劃 ===")
        
        # 按優先級排序
        priority_order = [
            ("chn_handwriting", "開源可直接下載"),
            ("hit_mw", "GitHub開源"),
            ("calamari_chinese", "學術平台可獲取"),
            ("scut_ept", "需要百度網盤"),
            ("casia_hwdb", "需要學術申請")
        ]
        
        download_plan = {
            "immediate_download": [],  # 可立即下載
            "manual_download": [],     # 需要手動獲取
            "academic_request": []     # 需要學術申請
        }
        
        for dataset_id, reason in priority_order:
            config = self.dataset_configs[dataset_id]
            
            if "開源" in reason or "GitHub" in reason:
                download_plan["immediate_download"].append({
                    "id": dataset_id,
                    "config": config,
                    "reason": reason
                })
            elif "百度網盤" in reason or "學術平台" in reason:
                download_plan["manual_download"].append({
                    "id": dataset_id,
                    "config": config,
                    "reason": reason
                })
            else:
                download_plan["academic_request"].append({
                    "id": dataset_id,
                    "config": config,
                    "reason": reason
                })
        
        # 顯示計劃
        print("🚀 立即可下載數據集:")
        for item in download_plan["immediate_download"]:
            print(f"   ✅ {item['config']['name']} ({item['reason']})")
        
        print("\n📥 需要手動獲取數據集:")
        for item in download_plan["manual_download"]:
            print(f"   🔄 {item['config']['name']} ({item['reason']})")
        
        print("\n📋 需要學術申請數據集:")
        for item in download_plan["academic_request"]:
            print(f"   📄 {item['config']['name']} ({item['reason']})")
        
        return download_plan
    
    def download_available_datasets(self, download_plan: Dict):
        """下載可獲取的數據集"""
        print("\n📥 === 開始下載可獲取數據集 ===")
        
        for item in download_plan["immediate_download"]:
            dataset_id = item["id"]
            config = item["config"]
            
            print(f"\n📦 正在下載: {config['name']}")
            
            dataset_dir = self.base_dir / dataset_id
            dataset_dir.mkdir(exist_ok=True)
            
            success_count = 0
            
            for i, url in enumerate(config["urls"]):
                try:
                    print(f"   🔗 URL {i+1}: {url}")
                    
                    # 檢查URL類型
                    if "github.com" in url and not url.endswith(('.zip', '.tar.gz', '.txt')):
                        # GitHub倉庫，嘗試獲取release
                        self._download_github_repo(url, dataset_dir, dataset_id)
                    elif url.endswith('.txt'):
                        # 文本文件直接下載
                        self._download_text_file(url, dataset_dir, f"{dataset_id}_data.txt")
                    elif url.endswith(('.zip', '.tar.gz')):
                        # 壓縮文件下載
                        self._download_compressed_file(url, dataset_dir, dataset_id)
                    else:
                        # 嘗試一般下載
                        self._download_general_file(url, dataset_dir, dataset_id, i)
                    
                    success_count += 1
                    print(f"   ✅ 下載成功")
                    
                except Exception as e:
                    print(f"   ❌ 下載失敗: {e}")
                    continue
            
            if success_count > 0:
                self.download_stats["successful_downloads"] += 1
                print(f"✅ {config['name']} 下載完成 ({success_count}/{len(config['urls'])} 成功)")
            else:
                self.download_stats["failed_downloads"] += 1
                print(f"❌ {config['name']} 下載失敗")
            
            self.download_stats["attempted_downloads"] += 1
    
    def _download_text_file(self, url: str, dataset_dir: Path, filename: str):
        """下載文本文件"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            file_path = dataset_dir / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            file_size = file_path.stat().st_size
            print(f"   📄 已下載: {filename} ({file_size:,} bytes)")
            self.download_stats["total_size_downloaded"] += file_size
            
        except Exception as e:
            raise Exception(f"文本文件下載失敗: {e}")
    
    def _download_compressed_file(self, url: str, dataset_dir: Path, dataset_id: str):
        """下載壓縮文件"""
        try:
            filename = url.split('/')[-1]
            file_path = dataset_dir / filename
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = file_path.stat().st_size
            print(f"   📦 已下載: {filename} ({file_size:,} bytes)")
            self.download_stats["total_size_downloaded"] += file_size
            
            # 如果是小文件，嘗試解壓
            if file_size < 100 * 1024 * 1024:  # 小於100MB
                self._extract_compressed_file(file_path, dataset_dir)
            
        except Exception as e:
            raise Exception(f"壓縮文件下載失敗: {e}")
    
    def _download_github_repo(self, url: str, dataset_dir: Path, dataset_id: str):
        """下載GitHub倉庫"""
        try:
            # 嘗試獲取release信息
            if "/tree/" in url:
                # 特定分支或路徑
                print(f"   📂 GitHub路徑: {url}")
                # 創建說明文件
                info_file = dataset_dir / f"{dataset_id}_github_info.txt"
                with open(info_file, 'w', encoding='utf-8') as f:
                    f.write(f"GitHub Repository: {url}\n")
                    f.write(f"Download Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Instructions: Manual clone required\n")
                    f.write(f"Command: git clone {url.split('/tree/')[0]}\n")
            else:
                # 主倉庫
                repo_url = url.replace("github.com", "api.github.com/repos")
                if not repo_url.endswith('/releases/latest'):
                    repo_url += '/releases/latest'
                
                response = requests.get(repo_url, timeout=30)
                if response.status_code == 200:
                    release_data = response.json()
                    download_url = release_data.get('zipball_url')
                    if download_url:
                        self._download_compressed_file(download_url, dataset_dir, dataset_id)
                    else:
                        raise Exception("No downloadable release found")
                else:
                    raise Exception(f"GitHub API error: {response.status_code}")
            
        except Exception as e:
            raise Exception(f"GitHub下載失敗: {e}")
    
    def _download_general_file(self, url: str, dataset_dir: Path, dataset_id: str, index: int):
        """一般文件下載"""
        try:
            response = requests.head(url, timeout=30)
            content_length = response.headers.get('content-length')
            
            if content_length and int(content_length) > 500 * 1024 * 1024:  # 大於500MB
                print(f"   ⚠️  大文件({int(content_length)/(1024*1024):.1f}MB)，跳過自動下載")
                # 創建下載說明
                info_file = dataset_dir / f"{dataset_id}_download_info_{index}.txt"
                with open(info_file, 'w', encoding='utf-8') as f:
                    f.write(f"Large File URL: {url}\n")
                    f.write(f"Size: {int(content_length)/(1024*1024):.1f} MB\n")
                    f.write(f"Manual download required\n")
                return
            
            # 小文件直接下載
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            filename = f"{dataset_id}_data_{index}.bin"
            file_path = dataset_dir / filename
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content)
            print(f"   📄 已下載: {filename} ({file_size:,} bytes)")
            self.download_stats["total_size_downloaded"] += file_size
            
        except Exception as e:
            raise Exception(f"一般下載失敗: {e}")
    
    def _extract_compressed_file(self, file_path: Path, extract_dir: Path):
        """解壓縮文件"""
        try:
            if file_path.suffix == '.zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"   📂 已解壓: {file_path.name}")
            elif file_path.suffix in ['.tar', '.gz']:
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
                print(f"   📂 已解壓: {file_path.name}")
        except Exception as e:
            print(f"   ⚠️  解壓失敗: {e}")
    
    def create_unified_dataset_index(self):
        """創建統一數據集索引"""
        print("\n📇 === 創建統一數據集索引 ===")
        
        index_data = {
            "metadata": {
                "creation_time": time.time(),
                "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ocr0712_version": "v1.0 - Large Dataset Integration",
                "indexer_version": "1.0"
            },
            "download_statistics": self.download_stats,
            "datasets": {},
            "unified_statistics": {
                "total_files": 0,
                "total_size_bytes": 0,
                "available_datasets": 0,
                "character_coverage": 0,
                "estimated_training_samples": 0
            }
        }
        
        total_files = 0
        total_size = 0
        
        # 掃描每個數據集目錄
        for dataset_id, config in self.dataset_configs.items():
            dataset_dir = self.base_dir / dataset_id
            
            if dataset_dir.exists():
                dataset_info = {
                    "name": config["name"],
                    "description": config["description"],
                    "status": "downloaded",
                    "files": [],
                    "total_size": 0,
                    "file_count": 0
                }
                
                # 掃描文件
                for file_path in dataset_dir.rglob("*"):
                    if file_path.is_file():
                        file_size = file_path.stat().st_size
                        file_info = {
                            "filename": file_path.name,
                            "path": str(file_path.relative_to(dataset_dir)),
                            "size": file_size,
                            "extension": file_path.suffix,
                            "modification_time": file_path.stat().st_mtime
                        }
                        dataset_info["files"].append(file_info)
                        dataset_info["total_size"] += file_size
                        total_size += file_size
                
                dataset_info["file_count"] = len(dataset_info["files"])
                total_files += dataset_info["file_count"]
                
                index_data["datasets"][dataset_id] = dataset_info
                index_data["unified_statistics"]["available_datasets"] += 1
                
                print(f"✅ {config['name']}: {dataset_info['file_count']} 文件, {dataset_info['total_size']/(1024*1024):.1f}MB")
            else:
                # 數據集未下載
                index_data["datasets"][dataset_id] = {
                    "name": config["name"],
                    "description": config["description"],
                    "status": "not_downloaded",
                    "download_urls": config["urls"],
                    "backup_info": config["backup_info"]
                }
                print(f"⚠️  {config['name']}: 未下載")
        
        # 更新統計
        index_data["unified_statistics"]["total_files"] = total_files
        index_data["unified_statistics"]["total_size_bytes"] = total_size
        
        # 保存索引
        index_file = self.base_dir / "unified_dataset_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 統一索引已創建: {index_file}")
        print(f"   總文件數: {total_files}")
        print(f"   總大小: {total_size/(1024*1024):.1f} MB")
        print(f"   可用數據集: {index_data['unified_statistics']['available_datasets']}/{len(self.dataset_configs)}")
        
        return index_data
    
    def generate_integration_report(self, index_data: Dict):
        """生成整合報告"""
        print("\n📊 === 數據集整合報告 ===")
        
        report = {
            "summary": {
                "execution_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_datasets_configured": len(self.dataset_configs),
                "successful_downloads": self.download_stats["successful_downloads"],
                "failed_downloads": self.download_stats["failed_downloads"],
                "download_success_rate": self.download_stats["successful_downloads"] / max(1, self.download_stats["attempted_downloads"]),
                "total_downloaded_mb": self.download_stats["total_size_downloaded"] / (1024*1024)
            },
            "dataset_status": {},
            "recommendations": [],
            "next_steps": []
        }
        
        # 數據集狀態
        for dataset_id, dataset_info in index_data["datasets"].items():
            report["dataset_status"][dataset_id] = {
                "name": dataset_info["name"],
                "status": dataset_info["status"],
                "file_count": dataset_info.get("file_count", 0),
                "size_mb": dataset_info.get("total_size", 0) / (1024*1024)
            }
        
        # 建議
        if report["summary"]["successful_downloads"] < len(self.dataset_configs):
            report["recommendations"].append("考慮手動下載未成功的數據集")
        
        if report["summary"]["total_downloaded_mb"] < 100:
            report["recommendations"].append("當前數據量較小，建議增加更多數據源")
        
        report["recommendations"].extend([
            "實施數據預處理和格式統一",
            "建立數據質量檢查機制",
            "設計增量更新策略"
        ])
        
        # 下一步
        report["next_steps"] = [
            "數據預處理：統一格式和標註",
            "質量評估：檢查數據完整性",
            "整合訓練：與OCR0712訓練流程對接",
            "性能測試：大規模數據訓練效果評估",
            "增量更新：定期獲取新數據"
        ]
        
        # 保存報告
        report_file = self.base_dir / "integration_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 顯示摘要
        print(f"總配置數據集: {report['summary']['total_datasets_configured']}")
        print(f"成功下載: {report['summary']['successful_downloads']}")
        print(f"下載失敗: {report['summary']['failed_downloads']}")
        print(f"成功率: {report['summary']['download_success_rate']:.1%}")
        print(f"下載總量: {report['summary']['total_downloaded_mb']:.1f} MB")
        print(f"\n📄 詳細報告: {report_file}")
        
        return report
    
    def run_large_dataset_integration(self):
        """運行大型數據集整合流程"""
        print("🚀 開始OCR0712大型數據集整合流程...")
        print()
        
        try:
            # 步驟1: 分析需求
            requirements = self.analyze_dataset_requirements()
            
            # 步驟2: 創建下載計劃
            download_plan = self.create_dataset_download_plan()
            
            # 步驟3: 下載可獲取數據集
            self.download_available_datasets(download_plan)
            
            # 步驟4: 創建統一索引
            index_data = self.create_unified_dataset_index()
            
            # 步驟5: 生成整合報告
            report = self.generate_integration_report(index_data)
            
            print(f"\n🎉 === 大型數據集整合完成！ ===")
            print(f"📁 數據目錄: {self.base_dir.absolute()}")
            print(f"📊 統一索引: unified_dataset_index.json")
            print(f"📋 整合報告: integration_report.json")
            
            return True
            
        except Exception as e:
            print(f"❌ 整合過程中出現錯誤: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函數"""
    integrator = LargeDatasetIntegrator()
    success = integrator.run_large_dataset_integration()
    
    if success:
        print("\n🔄 === 下一步建議 ===")
        print("1. 手動獲取需要特殊許可的數據集")
        print("2. 實施數據預處理和格式統一")
        print("3. 與OCR0712訓練系統整合")
        print("4. 執行大規模訓練測試")
    else:
        print("\n💡 故障排除建議:")
        print("1. 檢查網絡連接")
        print("2. 確保有足夠磁盤空間")
        print("3. 手動下載失敗的數據集")

if __name__ == "__main__":
    main()