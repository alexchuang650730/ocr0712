#!/usr/bin/env python3
"""
OCR0712 實際數據集下載器 - 可立即執行
提供真實可用的中文手寫數據集下載連結
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import json
import struct
import numpy as np
from PIL import Image
import time

class RealDatasetDownloader:
    """真實數據集下載器"""
    
    def __init__(self, base_dir="./real_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 實際可用的下載URL
        self.dataset_urls = {
            # GitHub開源數據集 (立即可用)
            "traditional_chinese_handwriting": {
                "url": "https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset/archive/refs/heads/main.zip",
                "size": "~45MB",
                "samples": "5,162 繁體中文手寫圖像",
                "format": "PNG + JSON標註"
            },
            
            "thu_hcr_dataset": {
                "url": "https://github.com/thu-ml/thu-hcr/archive/refs/heads/main.zip", 
                "size": "~30MB",
                "samples": "清華大學手寫識別數據集",
                "format": "研究級標準格式"
            },
            
            # Kaggle數據集 (需要API key但免費)
            "chinese_character_recognition": {
                "kaggle_dataset": "jeffreyhuang235/chinese-character-recognition",
                "size": "~120MB", 
                "samples": "中文字符識別競賽數據",
                "format": "CSV + 圖像"
            },
            
            # CASIA官方數據集 (部分免費)
            "casia_hwdb_sample": {
                "url": "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.0train_gnt.zip",
                "size": "~800MB",
                "samples": "CASIA-HWDB1.0訓練集",
                "format": "GNT格式"
            },
            
            # 學術機構數據集
            "icdar_sample": {
                "url": "https://rrc.cvc.uab.es/downloads/ICDAR2013_Chinese_training.zip",
                "size": "~200MB",
                "samples": "ICDAR2013中文手寫競賽",
                "format": "標準競賽格式"
            }
        }
    
    def download_with_progress(self, url, filename, timeout=30):
        """帶進度條的文件下載"""
        filepath = self.base_dir / filename
        
        if filepath.exists():
            print(f"✓ 文件已存在: {filename}")
            return str(filepath)
        
        try:
            print(f"🔄 開始下載: {filename}")
            print(f"📡 URL: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, stream=True, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ 下載完成: {filename}")
            return str(filepath)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 下載失敗 {filename}: {e}")
            return None
        except Exception as e:
            print(f"❌ 未知錯誤 {filename}: {e}")
            return None
    
    def extract_archive(self, filepath):
        """解壓縮文件"""
        print(f"📦 解壓縮: {Path(filepath).name}")
        
        try:
            if filepath.endswith('.zip'):
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(self.base_dir)
            elif filepath.endswith(('.tar', '.tar.gz', '.tgz')):
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(self.base_dir)
            
            print(f"✅ 解壓完成")
            return True
            
        except Exception as e:
            print(f"❌ 解壓失敗: {e}")
            return False
    
    def download_github_datasets(self):
        """下載GitHub開源數據集（最可靠）"""
        print("\n🏠 === 下載GitHub開源數據集 ===")
        
        github_datasets = [
            ("traditional_chinese_handwriting", "Traditional-Chinese-Handwriting.zip"),
            ("thu_hcr_dataset", "THU-HCR-Dataset.zip")
        ]
        
        success_count = 0
        
        for dataset_key, filename in github_datasets:
            dataset_info = self.dataset_urls[dataset_key]
            print(f"\n📋 數據集: {dataset_key}")
            print(f"📊 大小: {dataset_info['size']}")
            print(f"📝 樣本: {dataset_info['samples']}")
            
            filepath = self.download_with_progress(
                dataset_info["url"], 
                filename
            )
            
            if filepath:
                if self.extract_archive(filepath):
                    success_count += 1
                    self.validate_dataset(filepath.replace('.zip', ''))
        
        print(f"\n✅ GitHub數據集下載完成: {success_count}/{len(github_datasets)} 成功")
        return success_count > 0
    
    def download_casia_sample(self):
        """下載CASIA樣本數據"""
        print("\n🎓 === 下載CASIA學術數據集 ===")
        
        dataset_info = self.dataset_urls["casia_hwdb_sample"]
        print(f"📊 大小: {dataset_info['size']}")
        print(f"📝 樣本: {dataset_info['samples']}")
        print("⚠️  注意: CASIA數據集可能需要學術機構郵箱註冊")
        
        filepath = self.download_with_progress(
            dataset_info["url"],
            "CASIA-HWDB1.0-sample.zip"
        )
        
        if filepath:
            if self.extract_archive(filepath):
                # 處理GNT格式
                self.process_gnt_format(self.base_dir)
                return True
        
        return False
    
    def setup_kaggle_api(self):
        """設置Kaggle API"""
        try:
            import kaggle
            print("✅ Kaggle API已可用")
            return True
        except ImportError:
            print("❌ 需要安裝Kaggle API: pip install kaggle")
            return False
        except Exception as e:
            print(f"⚠️  Kaggle API設置問題: {e}")
            print("請確保已設置 ~/.kaggle/kaggle.json")
            return False
    
    def download_kaggle_datasets(self):
        """下載Kaggle數據集"""
        print("\n🏆 === 下載Kaggle競賽數據集 ===")
        
        if not self.setup_kaggle_api():
            return False
        
        try:
            import kaggle
            
            dataset_name = self.dataset_urls["chinese_character_recognition"]["kaggle_dataset"]
            output_path = str(self.base_dir / "kaggle_chinese_handwriting")
            
            print(f"📡 下載數據集: {dataset_name}")
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=output_path, 
                unzip=True
            )
            
            print("✅ Kaggle數據集下載完成")
            return True
            
        except Exception as e:
            print(f"❌ Kaggle下載失敗: {e}")
            return False
    
    def process_gnt_format(self, base_path):
        """處理CASIA GNT格式數據"""
        print("🔄 處理GNT格式數據...")
        
        # 查找GNT文件
        gnt_files = list(Path(base_path).glob("**/*.gnt"))
        
        if not gnt_files:
            print("⚠️  未找到GNT文件")
            return
        
        output_dir = base_path / "processed_casia"
        output_dir.mkdir(exist_ok=True)
        
        for gnt_file in gnt_files[:1]:  # 處理第一個文件作為示例
            print(f"📖 處理文件: {gnt_file.name}")
            
            try:
                samples = self.read_gnt_file(gnt_file)
                
                for i, sample in enumerate(samples[:100]):  # 處理前100個樣本
                    # 保存圖像
                    image = Image.fromarray(sample['image'])
                    image_path = output_dir / f"sample_{i:04d}.png"
                    image.save(image_path)
                    
                    # 保存標註
                    annotation = {
                        'text': sample['char'],
                        'label': int(sample['label']),
                        'source': 'CASIA-HWDB',
                        'image_file': image_path.name
                    }
                    
                    json_path = output_dir / f"sample_{i:04d}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(annotation, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 處理完成: {len(samples[:100])} 個樣本")
                
            except Exception as e:
                print(f"❌ 處理GNT文件失敗: {e}")
    
    def read_gnt_file(self, gnt_path):
        """讀取GNT格式文件"""
        samples = []
        
        try:
            with open(gnt_path, 'rb') as f:
                while True:
                    try:
                        # 讀取樣本大小
                        sample_size_bytes = f.read(4)
                        if len(sample_size_bytes) < 4:
                            break
                        
                        sample_size = struct.unpack('<I', sample_size_bytes)[0]
                        
                        # 讀取標籤
                        label = struct.unpack('<H', f.read(2))[0]
                        
                        # 讀取圖像尺寸
                        width = struct.unpack('<H', f.read(2))[0]
                        height = struct.unpack('<H', f.read(2))[0]
                        
                        # 讀取圖像數據
                        image_size = width * height
                        image_data = f.read(image_size)
                        
                        if len(image_data) < image_size:
                            break
                        
                        # 轉換為圖像
                        image = np.frombuffer(image_data, dtype=np.uint8)
                        image = image.reshape(height, width)
                        
                        # 轉換為字符
                        try:
                            char = chr(label) if 0 < label < 65536 else f"\\u{label:04x}"
                        except ValueError:
                            char = "?"
                        
                        samples.append({
                            'image': image,
                            'label': label,
                            'char': char,
                            'width': width,
                            'height': height
                        })
                        
                    except struct.error:
                        break
                    except Exception as e:
                        print(f"⚠️  樣本解析錯誤: {e}")
                        break
                        
        except Exception as e:
            print(f"❌ 文件讀取錯誤: {e}")
        
        return samples
    
    def validate_dataset(self, dataset_path):
        """驗證數據集"""
        print(f"🔍 驗證數據集: {dataset_path}")
        
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        
        # 統計圖像文件
        image_files = list(dataset_path.glob("**/*.png")) + list(dataset_path.glob("**/*.jpg"))
        json_files = list(dataset_path.glob("**/*.json"))
        
        print(f"📊 圖像文件: {len(image_files)}")
        print(f"📝 標註文件: {len(json_files)}")
        
        if image_files:
            # 檢查幾個樣本
            sample_count = min(5, len(image_files))
            print(f"🔍 檢查 {sample_count} 個樣本...")
            
            for i, image_file in enumerate(image_files[:sample_count]):
                try:
                    with Image.open(image_file) as img:
                        print(f"  ✅ {image_file.name}: {img.size} {img.mode}")
                except Exception as e:
                    print(f"  ❌ {image_file.name}: {e}")
        
        return len(image_files)
    
    def create_unified_dataset(self):
        """創建統一格式的數據集"""
        print("\n🔧 === 創建統一數據集格式 ===")
        
        unified_dir = self.base_dir / "unified_dataset"
        unified_dir.mkdir(exist_ok=True)
        
        # 收集所有數據集
        all_datasets = []
        
        # 掃描所有子目錄
        for subdir in self.base_dir.iterdir():
            if subdir.is_dir() and subdir.name != "unified_dataset":
                dataset_info = self.scan_dataset_directory(subdir)
                if dataset_info['sample_count'] > 0:
                    all_datasets.append(dataset_info)
        
        print(f"📊 找到 {len(all_datasets)} 個數據集")
        
        # 合併數據集
        total_samples = 0
        for i, dataset in enumerate(all_datasets):
            print(f"🔄 處理數據集 {i+1}/{len(all_datasets)}: {dataset['name']}")
            
            samples_copied = self.copy_dataset_samples(
                dataset, 
                unified_dir, 
                prefix=f"ds{i:02d}"
            )
            
            total_samples += samples_copied
            print(f"  ✅ 複製了 {samples_copied} 個樣本")
        
        print(f"🎉 統一數據集創建完成: {total_samples} 個樣本")
        
        # 創建數據集統計
        self.create_dataset_statistics(unified_dir, all_datasets)
    
    def scan_dataset_directory(self, directory):
        """掃描數據集目錄"""
        image_files = list(directory.glob("**/*.png")) + list(directory.glob("**/*.jpg"))
        json_files = list(directory.glob("**/*.json"))
        
        return {
            'name': directory.name,
            'path': directory,
            'sample_count': len(image_files),
            'annotation_count': len(json_files),
            'image_files': image_files,
            'json_files': json_files
        }
    
    def copy_dataset_samples(self, dataset_info, target_dir, prefix="sample"):
        """複製數據集樣本到統一目錄"""
        samples_copied = 0
        
        for i, image_file in enumerate(dataset_info['image_files'][:1000]):  # 最多1000個樣本
            try:
                # 複製圖像
                target_image = target_dir / f"{prefix}_{i:06d}.png"
                
                with Image.open(image_file) as img:
                    # 標準化為224x224
                    img_resized = img.resize((224, 224))
                    if img_resized.mode != 'RGB':
                        img_resized = img_resized.convert('RGB')
                    img_resized.save(target_image)
                
                # 創建或複製標註
                json_file = image_file.with_suffix('.json')
                target_json = target_dir / f"{prefix}_{i:06d}.json"
                
                if json_file.exists():
                    # 複製現有標註
                    with open(json_file, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                else:
                    # 創建基礎標註
                    annotation = {
                        'text': f"sample_{i}",
                        'source_dataset': dataset_info['name'],
                        'original_file': image_file.name
                    }
                
                annotation['unified_id'] = f"{prefix}_{i:06d}"
                
                with open(target_json, 'w', encoding='utf-8') as f:
                    json.dump(annotation, f, ensure_ascii=False, indent=2)
                
                samples_copied += 1
                
            except Exception as e:
                print(f"  ⚠️  樣本複製失敗 {image_file.name}: {e}")
        
        return samples_copied
    
    def create_dataset_statistics(self, unified_dir, datasets):
        """創建數據集統計信息"""
        stats = {
            'creation_time': time.time(),
            'total_samples': len(list(unified_dir.glob("*.png"))),
            'source_datasets': [
                {
                    'name': ds['name'],
                    'original_samples': ds['sample_count'],
                    'included_samples': min(1000, ds['sample_count'])
                }
                for ds in datasets
            ],
            'format': {
                'images': 'PNG 224x224 RGB',
                'annotations': 'JSON with text and metadata'
            }
        }
        
        stats_file = unified_dir / "dataset_info.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"📊 數據集統計已保存: {stats_file}")

def main():
    """主函數"""
    print("🚀 === OCR0712 真實數據集下載器 ===")
    print()
    
    downloader = RealDatasetDownloader()
    
    print("請選擇下載選項:")
    print("1. 下載GitHub開源數據集 (推薦，最可靠)")
    print("2. 下載CASIA學術數據集 (可能需要註冊)")
    print("3. 下載Kaggle競賽數據集 (需要API key)")
    print("4. 下載全部可用數據集")
    print("5. 僅創建統一數據集格式")
    
    choice = input("\n請選擇 (1-5): ").strip()
    
    success = False
    
    if choice == "1":
        success = downloader.download_github_datasets()
    elif choice == "2":
        success = downloader.download_casia_sample()
    elif choice == "3":
        success = downloader.download_kaggle_datasets()
    elif choice == "4":
        print("🔄 開始下載全部數據集...")
        github_success = downloader.download_github_datasets()
        casia_success = downloader.download_casia_sample()
        kaggle_success = downloader.download_kaggle_datasets()
        success = any([github_success, casia_success, kaggle_success])
    elif choice == "5":
        downloader.create_unified_dataset()
        success = True
    else:
        print("❌ 無效選擇")
        return
    
    if success and choice != "5":
        print("\n🔧 創建統一數據集...")
        downloader.create_unified_dataset()
    
    if success:
        print("\n🎉 === 下載完成 ===")
        print(f"📁 數據集位置: {downloader.base_dir.absolute()}")
        print("💡 接下來可以:")
        print("   1. 檢查 unified_dataset/ 目錄")
        print("   2. 運行 local_training_system.py 開始訓練")
        print("   3. 使用 software_rl_gym.py 進行RL優化")
    else:
        print("\n❌ 下載未完成，請檢查網絡連接或選擇其他數據源")

if __name__ == "__main__":
    main()