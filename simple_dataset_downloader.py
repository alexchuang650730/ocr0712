#!/usr/bin/env python3
"""
OCR0712 精簡版數據集下載器 - 無外部依賴
提供實際可用的下載連結和處理方法
"""

import os
import urllib.request
import zipfile
import json
from pathlib import Path
import time

class SimpleDatasetDownloader:
    """精簡版數據集下載器"""
    
    def __init__(self, base_dir="./real_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 實際可用的下載URL
        self.dataset_urls = {
            "github_traditional_chinese": {
                "url": "https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset/archive/refs/heads/main.zip",
                "filename": "Traditional-Chinese-Handwriting.zip",
                "description": "繁體中文手寫數據集 - 5,162個樣本",
                "size": "約45MB"
            },
            
            "github_thu_hcr": {
                "url": "https://github.com/thu-ml/thu-hcr/archive/refs/heads/main.zip",
                "filename": "THU-HCR-Dataset.zip", 
                "description": "清華大學手寫識別數據集",
                "size": "約30MB"
            },
            
            "sample_chinese_chars": {
                "url": "https://raw.githubusercontent.com/skishore/makemeahanzi/master/graphics.txt",
                "filename": "chinese_characters_sample.txt",
                "description": "中文字符圖形數據樣本",
                "size": "約5MB"
            }
        }
    
    def download_file(self, url, filename):
        """簡單文件下載"""
        filepath = self.base_dir / filename
        
        if filepath.exists():
            print(f"✓ 文件已存在: {filename}")
            return str(filepath)
        
        try:
            print(f"🔄 開始下載: {filename}")
            print(f"📡 URL: {url}")
            
            # 添加User-Agent避免被阻擋
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                
                with open(filepath, 'wb') as f:
                    downloaded = 0
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r📥 下載進度: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
                    
                    print()  # 換行
            
            print(f"✅ 下載完成: {filename}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ 下載失敗 {filename}: {e}")
            return None
    
    def extract_zip(self, zip_path):
        """解壓ZIP文件"""
        print(f"📦 解壓縮: {Path(zip_path).name}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                extract_path = self.base_dir / Path(zip_path).stem
                zip_ref.extractall(extract_path)
            
            print(f"✅ 解壓完成到: {extract_path}")
            return extract_path
            
        except Exception as e:
            print(f"❌ 解壓失敗: {e}")
            return None
    
    def download_github_datasets(self):
        """下載GitHub開源數據集"""
        print("\n🏠 === 下載GitHub開源數據集 ===")
        
        success_count = 0
        total_count = 0
        
        for dataset_key in ["github_traditional_chinese", "github_thu_hcr"]:
            total_count += 1
            dataset_info = self.dataset_urls[dataset_key]
            
            print(f"\n📋 數據集: {dataset_info['description']}")
            print(f"📊 大小: {dataset_info['size']}")
            
            # 下載文件
            filepath = self.download_file(
                dataset_info["url"], 
                dataset_info["filename"]
            )
            
            # 解壓縮
            if filepath and filepath.endswith('.zip'):
                extract_path = self.extract_zip(filepath)
                if extract_path:
                    self.analyze_dataset(extract_path)
                    success_count += 1
        
        print(f"\n✅ GitHub數據集下載: {success_count}/{total_count} 成功")
        return success_count > 0
    
    def download_sample_data(self):
        """下載樣本數據"""
        print("\n📝 === 下載樣本數據 ===")
        
        dataset_info = self.dataset_urls["sample_chinese_chars"]
        print(f"📋 {dataset_info['description']}")
        
        filepath = self.download_file(
            dataset_info["url"],
            dataset_info["filename"]
        )
        
        if filepath:
            self.process_sample_data(filepath)
            return True
        
        return False
    
    def analyze_dataset(self, dataset_path):
        """分析數據集內容"""
        print(f"🔍 分析數據集: {dataset_path.name}")
        
        # 統計文件類型
        file_counts = {}
        total_size = 0
        
        for file_path in dataset_path.rglob('*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                file_counts[suffix] = file_counts.get(suffix, 0) + 1
                total_size += file_path.stat().st_size
        
        print(f"📊 文件統計:")
        for suffix, count in sorted(file_counts.items()):
            print(f"  {suffix or '無後綴'}: {count} 個文件")
        
        print(f"💾 總大小: {total_size / (1024*1024):.1f} MB")
        
        # 查找重要文件
        readme_files = list(dataset_path.rglob('README*'))
        data_files = list(dataset_path.rglob('*.json')) + list(dataset_path.rglob('*.csv'))
        image_files = list(dataset_path.rglob('*.png')) + list(dataset_path.rglob('*.jpg'))
        
        if readme_files:
            print(f"📖 說明文件: {[f.name for f in readme_files]}")
        
        if data_files:
            print(f"📄 數據文件: {len(data_files)} 個")
            
        if image_files:
            print(f"🖼️  圖像文件: {len(image_files)} 個")
            
            # 分析幾個圖像樣本
            sample_count = min(3, len(image_files))
            print(f"🔍 檢查 {sample_count} 個圖像樣本:")
            
            for i, img_file in enumerate(image_files[:sample_count]):
                try:
                    file_size = img_file.stat().st_size
                    print(f"  ✅ {img_file.name}: {file_size:,} bytes")
                except Exception as e:
                    print(f"  ❌ {img_file.name}: {e}")
    
    def process_sample_data(self, text_file):
        """處理樣本文本數據"""
        print(f"🔄 處理樣本數據: {Path(text_file).name}")
        
        try:
            processed_dir = self.base_dir / "processed_samples"
            processed_dir.mkdir(exist_ok=True)
            
            # 讀取文本文件
            with open(text_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"📄 讀取了 {len(lines)} 行數據")
            
            # 處理前100行作為樣本
            samples = []
            for i, line in enumerate(lines[:100]):
                line = line.strip()
                if line:
                    sample = {
                        'id': i,
                        'content': line,
                        'source': 'chinese_characters_sample',
                        'processed_time': time.time()
                    }
                    samples.append(sample)
            
            # 保存處理後的樣本
            output_file = processed_dir / "processed_samples.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 處理完成: {len(samples)} 個樣本保存到 {output_file}")
            
        except Exception as e:
            print(f"❌ 處理失敗: {e}")
    
    def create_download_summary(self):
        """創建下載摘要"""
        print("\n📋 === 創建下載摘要 ===")
        
        summary = {
            'download_time': time.time(),
            'base_directory': str(self.base_dir.absolute()),
            'available_datasets': [],
            'total_files': 0,
            'total_size_mb': 0
        }
        
        # 掃描所有下載的內容
        for item in self.base_dir.iterdir():
            if item.is_file():
                file_size = item.stat().st_size
                summary['total_files'] += 1
                summary['total_size_mb'] += file_size / (1024 * 1024)
                
                summary['available_datasets'].append({
                    'name': item.name,
                    'type': 'file',
                    'size_mb': file_size / (1024 * 1024)
                })
                
            elif item.is_dir():
                # 統計目錄內容
                dir_files = list(item.rglob('*'))
                dir_size = sum(f.stat().st_size for f in dir_files if f.is_file())
                
                summary['available_datasets'].append({
                    'name': item.name,
                    'type': 'directory',
                    'files': len([f for f in dir_files if f.is_file()]),
                    'size_mb': dir_size / (1024 * 1024)
                })
                
                summary['total_files'] += len([f for f in dir_files if f.is_file()])
                summary['total_size_mb'] += dir_size / (1024 * 1024)
        
        # 保存摘要
        summary_file = self.base_dir / "download_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"📊 下載摘要:")
        print(f"  📁 數據集數量: {len(summary['available_datasets'])}")
        print(f"  📄 總文件數: {summary['total_files']}")
        print(f"  💾 總大小: {summary['total_size_mb']:.1f} MB")
        print(f"  📋 摘要文件: {summary_file}")
        
        return summary

def show_dataset_info():
    """顯示數據集信息"""
    print("🎯 === 可用的中文手寫數據集 ===")
    print()
    
    datasets_info = [
        {
            "name": "Traditional Chinese Handwriting Dataset",
            "source": "AI-FREE-Team (GitHub)",
            "samples": "5,162 繁體中文手寫圖像",
            "format": "PNG + JSON標註",
            "license": "MIT",
            "url": "https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset"
        },
        {
            "name": "THU-HCR Dataset", 
            "source": "清華大學 (GitHub)",
            "samples": "學術級手寫識別數據集",
            "format": "多種格式",
            "license": "學術使用",
            "url": "https://github.com/thu-ml/thu-hcr"
        },
        {
            "name": "CASIA-HWDB",
            "source": "中科院自動化所",
            "samples": "3.9M 手寫漢字 (需註冊)",
            "format": "GNT格式",
            "license": "學術使用",
            "url": "http://www.nlpr.ia.ac.cn/databases/handwriting/"
        }
    ]
    
    for i, dataset in enumerate(datasets_info, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   來源: {dataset['source']}")
        print(f"   樣本: {dataset['samples']}")
        print(f"   格式: {dataset['format']}")
        print(f"   授權: {dataset['license']}")
        print(f"   連結: {dataset['url']}")
        print()

def main():
    """主函數"""
    print("🚀 === OCR0712 精簡版數據集下載器 ===")
    print()
    
    show_dataset_info()
    
    downloader = SimpleDatasetDownloader()
    
    print("請選擇下載選項:")
    print("1. 下載GitHub開源數據集 (推薦)")
    print("2. 下載樣本數據")
    print("3. 下載全部可用數據")
    print("4. 僅顯示數據集信息")
    print("5. 創建下載摘要")
    
    try:
        choice = input("\n請選擇 (1-5): ").strip()
    except KeyboardInterrupt:
        print("\n👋 下載已取消")
        return
    
    success = False
    
    if choice == "1":
        success = downloader.download_github_datasets()
    elif choice == "2":
        success = downloader.download_sample_data()
    elif choice == "3":
        print("🔄 下載全部數據...")
        github_success = downloader.download_github_datasets()
        sample_success = downloader.download_sample_data()
        success = github_success or sample_success
    elif choice == "4":
        print("ℹ️  數據集信息已顯示在上方")
        return
    elif choice == "5":
        downloader.create_download_summary()
        return
    else:
        print("❌ 無效選擇")
        return
    
    if success:
        print("\n📋 創建下載摘要...")
        summary = downloader.create_download_summary()
        
        print("\n🎉 === 下載完成 ===")
        print(f"📁 數據集位置: {downloader.base_dir.absolute()}")
        print("\n💡 後續步驟:")
        print("   1. 檢查下載的數據集目錄")
        print("   2. 查看 download_summary.json 了解詳情")
        print("   3. 運行 local_training_system.py 開始訓練")
        print("   4. 使用 software_rl_gym.py 進行RL優化")
        
        print(f"\n📝 快速開始命令:")
        print(f"   cd {downloader.base_dir}")
        print(f"   ls -la")
        print(f"   python3 ../local_training_system.py")
        
    else:
        print("\n❌ 下載未完成")
        print("💡 建議:")
        print("   1. 檢查網絡連接")
        print("   2. 稍後重試")
        print("   3. 嘗試手動下載數據集")

if __name__ == "__main__":
    main()