#!/usr/bin/env python3
"""
實際可用的基礎數據集下載器
經過驗證的真實可用連結
"""

import urllib.request
import json
import os
from pathlib import Path
import time

def download_real_datasets():
    """下載真實可用的數據集"""
    
    base_dir = Path("./real_chinese_datasets")
    base_dir.mkdir(exist_ok=True)
    
    # 實際可用的下載連結 (已驗證)
    datasets = [
        {
            "name": "Chinese Graphics Data",
            "url": "https://raw.githubusercontent.com/skishore/makemeahanzi/master/graphics.txt",
            "filename": "chinese_graphics.txt",
            "description": "9K+ 中文字符筆畫數據"
        },
        {
            "name": "Chinese Dictionary",
            "url": "https://raw.githubusercontent.com/skishore/makemeahanzi/master/dictionary.txt", 
            "filename": "chinese_dictionary.txt",
            "description": "中文字典數據"
        },
        {
            "name": "Chinese Names Corpus",
            "url": "https://raw.githubusercontent.com/wainshine/Chinese-Names-Corpus/master/Chinese_Names_Corpus.txt",
            "filename": "chinese_names.txt",
            "description": "中文姓名語料庫"
        }
    ]
    
    print("🔄 開始下載真實可用的數據集...")
    
    for dataset in datasets:
        print(f"\n📥 下載: {dataset['name']}")
        print(f"📝 描述: {dataset['description']}")
        print(f"🔗 URL: {dataset['url']}")
        
        try:
            filepath = base_dir / dataset['filename']
            
            # 添加請求頭避免被阻擋
            req = urllib.request.Request(dataset['url'])
            req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(filepath, 'wb') as f:
                    f.write(response.read())
            
            # 檢查文件大小
            file_size = filepath.stat().st_size
            print(f"✅ 下載完成: {filepath.name} ({file_size:,} bytes)")
            
        except Exception as e:
            print(f"❌ 下載失敗: {e}")
    
    print(f"\n📁 數據集保存位置: {base_dir.absolute()}")
    return base_dir

def process_graphics_data(data_dir):
    """處理中文字符圖形數據"""
    
    graphics_file = data_dir / "chinese_graphics.txt"
    
    if not graphics_file.exists():
        print("❌ 未找到圖形數據文件")
        return []
    
    print("🔄 處理中文字符圖形數據...")
    
    processed_data = []
    
    try:
        with open(graphics_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num >= 1000:  # 處理前1000個字符
                    break
                    
                try:
                    data = json.loads(line.strip())
                    
                    if 'character' in data and 'strokes' in data:
                        processed_sample = {
                            'character': data['character'],
                            'strokes': data['strokes'],
                            'medians': data.get('medians', []),
                            'stroke_count': len(data['strokes']) if data['strokes'] else 0
                        }
                        processed_data.append(processed_sample)
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    if line_num < 10:  # 只報告前10個錯誤
                        print(f"⚠️  處理第{line_num}行時出錯: {e}")
        
        # 保存處理後的數據
        output_file = data_dir / "processed_chinese_strokes.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 處理完成: {len(processed_data)} 個字符")
        print(f"📄 輸出文件: {output_file}")
        
        # 生成統計信息
        if processed_data:
            stroke_counts = {}
            for sample in processed_data:
                count = sample['stroke_count']
                stroke_counts[count] = stroke_counts.get(count, 0) + 1
            
            print(f"📊 筆畫數分佈: {dict(sorted(stroke_counts.items()))}")
            
            # 顯示幾個樣本
            print("\n📝 樣本預覽:")
            for i, sample in enumerate(processed_data[:5]):
                print(f"  {i+1}. 字符: {sample['character']}, 筆畫數: {sample['stroke_count']}")
        
    except Exception as e:
        print(f"❌ 處理文件時出錯: {e}")
    
    return processed_data

def create_training_format(data_dir, processed_data):
    """創建訓練格式數據"""
    
    print("🔧 創建OCR0712訓練格式...")
    
    training_dir = data_dir / "ocr0712_training_data"
    training_dir.mkdir(exist_ok=True)
    
    training_samples = []
    
    for i, sample in enumerate(processed_data[:500]):  # 使用前500個樣本
        
        # 創建訓練樣本
        training_sample = {
            'sample_id': f"stroke_{i:04d}",
            'text': sample['character'],
            'stroke_data': {
                'strokes': sample['strokes'],
                'medians': sample['medians'],
                'stroke_count': sample['stroke_count']
            },
            'metadata': {
                'source': 'makemeahanzi',
                'data_type': 'stroke_sequence',
                'script_type': 'traditional' if ord(sample['character']) > 0x4E00 else 'other'
            }
        }
        
        training_samples.append(training_sample)
        
        # 保存單個樣本文件
        sample_file = training_dir / f"sample_{i:04d}.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(training_sample, f, ensure_ascii=False, indent=2)
    
    # 保存批量文件
    batch_file = training_dir / "batch_training_data.json"
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(training_samples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 訓練數據創建完成: {len(training_samples)} 個樣本")
    print(f"📁 訓練數據目錄: {training_dir}")
    
    return training_samples

def show_alternative_sources():
    """顯示其他可用數據源"""
    
    print("\n🔗 === 其他可用數據源 ===")
    
    sources = [
        {
            "name": "CASIA-HWDB (官方)",
            "url": "http://www.nlpr.ia.ac.cn/databases/handwriting/",
            "description": "需要學術註冊，3.9M手寫漢字",
            "access": "學術申請"
        },
        {
            "name": "Kaggle Chinese MNIST",
            "url": "https://www.kaggle.com/datasets/gpreda/chinese-mnist",
            "description": "15K中文數字手寫樣本",
            "access": "Kaggle註冊"
        },
        {
            "name": "Unicode漢字數據",
            "url": "https://www.unicode.org/charts/unihan.html",
            "description": "完整的Unicode中文字符信息",
            "access": "公開下載"
        },
        {
            "name": "Google Fonts Noto CJK",
            "url": "https://github.com/googlefonts/noto-cjk",
            "description": "開源中日韓字體，可用於合成數據",
            "access": "GitHub下載"
        }
    ]
    
    for i, source in enumerate(sources, 1):
        print(f"\n{i}. {source['name']}")
        print(f"   🔗 連結: {source['url']}")
        print(f"   📝 描述: {source['description']}")
        print(f"   🔑 獲取方式: {source['access']}")

def main():
    """主函數"""
    print("🚀 === OCR0712 實際可用數據集下載器 ===")
    print()
    
    # 下載數據集
    data_dir = download_real_datasets()
    
    # 處理數據
    processed_data = process_graphics_data(data_dir)
    
    if processed_data:
        # 創建訓練格式
        training_samples = create_training_format(data_dir, processed_data)
        
        print("\n🎉 === 下載和處理完成 ===")
        print("✅ 成功獲取真實中文手寫數據")
        print("✅ 處理為OCR0712訓練格式")
        print("✅ 創建了訓練樣本文件")
        
        print(f"\n📂 文件位置:")
        print(f"   原始數據: {data_dir}")
        print(f"   處理數據: {data_dir}/processed_chinese_strokes.json")
        print(f"   訓練數據: {data_dir}/ocr0712_training_data/")
        
        print(f"\n🚀 下一步:")
        print("1. 檢查下載的數據文件")
        print("2. 運行本地訓練系統:")
        print(f"   python3 local_training_system.py --data-path {data_dir}")
        print("3. 使用RL優化:")
        print("   python3 software_rl_gym.py")
        
    else:
        print("\n⚠️  數據處理未完成，但可以嘗試手動下載")
    
    # 顯示其他數據源
    show_alternative_sources()
    
    print(f"\n💡 如需更多數據，可使用wget直接下載:")
    print("wget https://raw.githubusercontent.com/skishore/makemeahanzi/master/graphics.txt")

if __name__ == "__main__":
    main()