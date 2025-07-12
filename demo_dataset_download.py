#!/usr/bin/env python3
"""
OCR0712 自動數據集下載演示
"""

import sys
import os
sys.path.append('.')

from simple_dataset_downloader import SimpleDatasetDownloader, show_dataset_info

def demo_download():
    """演示自動下載"""
    print("🚀 === OCR0712 自動數據集下載演示 ===")
    print()
    
    # 顯示可用數據集
    show_dataset_info()
    
    # 創建下載器
    downloader = SimpleDatasetDownloader()
    
    print("🔄 開始自動下載GitHub開源數據集...")
    print()
    
    # 自動下載GitHub數據集
    success = downloader.download_github_datasets()
    
    if success:
        print("\n🔄 下載樣本數據...")
        downloader.download_sample_data()
        
        print("\n📋 創建下載摘要...")
        summary = downloader.create_download_summary()
        
        print("\n🎉 === 自動下載完成 ===")
        print(f"📁 數據集位置: {downloader.base_dir.absolute()}")
        
        # 顯示實際可用的下載連結
        print("\n🔗 === 實際可用下載連結 ===")
        
        actual_urls = [
            {
                "name": "Traditional Chinese Handwriting Dataset",
                "direct_download": "https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset/archive/refs/heads/main.zip",
                "samples": "5,162 繁體中文手寫圖像",
                "size": "~45MB"
            },
            {
                "name": "THU-HCR Dataset",
                "direct_download": "https://github.com/thu-ml/thu-hcr/archive/refs/heads/main.zip",
                "samples": "清華大學手寫識別數據集",
                "size": "~30MB"
            },
            {
                "name": "Chinese Characters Graphics Data",
                "direct_download": "https://raw.githubusercontent.com/skishore/makemeahanzi/master/graphics.txt",
                "samples": "中文字符圖形數據",
                "size": "~5MB"
            },
            {
                "name": "CASIA-HWDB Sample",
                "direct_download": "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.0train_gnt.zip",
                "samples": "3.9M 手寫漢字樣本",
                "size": "~800MB",
                "note": "可能需要學術機構註冊"
            }
        ]
        
        for i, dataset in enumerate(actual_urls, 1):
            print(f"\n{i}. {dataset['name']}")
            print(f"   📥 直接下載: {dataset['direct_download']}")
            print(f"   📊 樣本數量: {dataset['samples']}")
            print(f"   💾 文件大小: {dataset['size']}")
            if 'note' in dataset:
                print(f"   ⚠️  注意: {dataset['note']}")
        
        print("\n🛠️  === 使用wget下載命令 ===")
        print("# 繁體中文手寫數據集")
        print("wget https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset/archive/refs/heads/main.zip -O traditional_chinese.zip")
        print()
        print("# 清華大學數據集")
        print("wget https://github.com/thu-ml/thu-hcr/archive/refs/heads/main.zip -O thu_hcr.zip")
        print()
        print("# 中文字符數據")
        print("wget https://raw.githubusercontent.com/skishore/makemeahanzi/master/graphics.txt -O chinese_chars.txt")
        
        print("\n🔧 === 整合到OCR0712訓練 ===")
        print("1. 解壓下載的數據集:")
        print("   unzip traditional_chinese.zip")
        print("   unzip thu_hcr.zip")
        print()
        print("2. 運行本地訓練系統:")
        print("   python3 local_training_system.py --create-data")
        print("   python3 local_training_system.py --train")
        print()
        print("3. 使用軟件RL優化:")
        print("   python3 software_rl_gym.py")
        
    else:
        print("\n❌ 自動下載未完成")
        print("💡 可以手動使用上述wget命令下載數據集")

if __name__ == "__main__":
    demo_download()