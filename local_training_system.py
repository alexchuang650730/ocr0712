#!/usr/bin/env python3
"""
OCR0712 本地訓練系統 - Mac部署版
專門針對中文手寫文件識別的本地訓練框架
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 導入我們的OCR模塊
from sota_ondevice_ocr import ChineseInertialGAN, ScriptType, ScalingRLOptimizer
from hybrid_edge_cloud_ocr import HybridEdgeCloudOCR
from software_rl_gym import OCRGymEnvironment, SoftwareSensorSystem

# 設置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """訓練配置"""
    # 基本設置
    model_name: str = "OCR0712_Handwriting"
    script_type: str = "mixed"  # traditional, simplified, mixed
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    
    # 數據設置
    train_data_dir: str = "./data/train"
    val_data_dir: str = "./data/val"
    test_data_dir: str = "./data/test"
    image_size: Tuple[int, int] = (224, 224)
    
    # 模型設置
    use_pretrained: bool = True
    freeze_backbone: bool = False
    gan_latent_dim: int = 512
    trajectory_points: int = 500
    
    # 訓練設置
    save_every: int = 10
    validate_every: int = 5
    early_stopping_patience: int = 20
    gradient_clip_value: float = 1.0
    
    # Mac優化設置
    use_mps: bool = True  # Mac M1/M2 GPU
    num_workers: int = 2  # Mac適用的worker數量
    pin_memory: bool = True
    
    # 輸出設置
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    tensorboard_dir: str = "./runs"

class HandwritingDataset(Dataset):
    """手寫文件數據集"""
    
    def __init__(self, data_dir: str, config: TrainingConfig, is_training: bool = True):
        self.data_dir = Path(data_dir)
        self.config = config
        self.is_training = is_training
        
        # 數據變換
        self.transform = self._get_transforms()
        
        # 加載數據索引
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
        
    def _get_transforms(self):
        """獲取數據變換"""
        
        if self.is_training:
            # 訓練時的數據增強
            return transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.RandomRotation((-5, 5)),
                transforms.RandomAffine(0, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # 驗證/測試時的基本變換
            return transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _load_samples(self) -> List[Dict]:
        """加載數據樣本"""
        
        samples = []
        
        # 支持的圖像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 遍歷數據目錄
        for image_path in self.data_dir.rglob('*'):
            if image_path.suffix.lower() in image_extensions:
                # 查找對應的標註文件
                annotation_path = image_path.with_suffix('.json')
                
                if annotation_path.exists():
                    try:
                        with open(annotation_path, 'r', encoding='utf-8') as f:
                            annotation = json.load(f)
                        
                        sample = {
                            'image_path': str(image_path),
                            'annotation': annotation,
                            'text': annotation.get('text', ''),
                            'script_type': annotation.get('script_type', 'mixed'),
                            'trajectory': annotation.get('trajectory', []),
                            'bbox': annotation.get('bbox', [])
                        }
                        
                        samples.append(sample)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load annotation for {image_path}: {e}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加載圖像
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.error(f"Failed to load image {sample['image_path']}: {e}")
            # 返回空白圖像
            image = torch.zeros(3, *self.config.image_size)
        
        # 處理標籤
        text = sample['text']
        script_type = sample['script_type']
        trajectory = np.array(sample['trajectory']) if sample['trajectory'] else np.zeros((500, 2))
        
        return {
            'image': image,
            'text': text,
            'script_type': script_type,
            'trajectory': torch.tensor(trajectory, dtype=torch.float32),
            'sample_info': sample
        }

class OCR0712Model(nn.Module):
    """OCR0712主模型"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # GAN組件
        self.gan_traditional = ChineseInertialGAN(ScriptType.TRADITIONAL_CHINESE)
        self.gan_simplified = ChineseInertialGAN(ScriptType.SIMPLIFIED_CHINESE)
        
        # 文本識別器
        self.text_recognizer = self._build_text_recognizer()
        
        # 腳本類型分類器
        self.script_classifier = self._build_script_classifier()
        
        # 軌跡回歸器
        self.trajectory_regressor = self._build_trajectory_regressor()
        
    def _build_text_recognizer(self):
        """構建文本識別器"""
        
        return nn.Sequential(
            # CNN特徵提取
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            
            # 全連接層
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 輸出層 (假設詞彙表大小為10000)
            nn.Linear(512, 10000)
        )
    
    def _build_script_classifier(self):
        """構建腳本類型分類器"""
        
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # traditional, simplified, mixed
        )
    
    def _build_trajectory_regressor(self):
        """構建軌跡回歸器"""
        
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1000)  # 500 points * 2 coordinates
        )
    
    def forward(self, x):
        # 文本識別
        text_logits = self.text_recognizer(x)
        
        # 腳本類型分類
        script_logits = self.script_classifier(x)
        
        # 軌跡回歸
        trajectory_pred = self.trajectory_regressor(x)
        trajectory_pred = trajectory_pred.view(-1, 500, 2)
        
        return {
            'text_logits': text_logits,
            'script_logits': script_logits,
            'trajectory': trajectory_pred
        }

class OCR0712Trainer:
    """OCR0712訓練器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        
        # 創建輸出目錄
        self._create_directories()
        
        # 初始化模型
        self.model = OCR0712Model(config).to(self.device)
        
        # 初始化優化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # 學習率調度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # 損失函數
        self.criterion_text = nn.CrossEntropyLoss()
        self.criterion_script = nn.CrossEntropyLoss()
        self.criterion_trajectory = nn.MSELoss()
        
        # RL優化器和軟件環境
        self.rl_optimizer = ScalingRLOptimizer()
        self.software_rl_gym = OCRGymEnvironment()
        self.software_sensors = SoftwareSensorSystem()
        
        # TensorBoard
        self.writer = SummaryWriter(config.tensorboard_dir)
        
        # 訓練狀態
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
        
    def _setup_device(self):
        """設置計算設備"""
        
        if self.config.use_mps and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Mac MPS (Metal Performance Shaders)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA GPU")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
            
        return device
    
    def _create_directories(self):
        """創建必要的目錄"""
        
        dirs = [
            self.config.output_dir,
            self.config.checkpoint_dir,
            self.config.tensorboard_dir
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def create_sample_data(self):
        """創建示例數據用於測試"""
        
        logger.info("Creating sample training data...")
        
        # 創建數據目錄
        for split in ['train', 'val', 'test']:
            data_dir = Path(f"./data/{split}")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # 創建示例圖像和標註
            for i in range(50 if split == 'train' else 10):
                # 創建示例圖像
                img = np.ones((224, 224, 3), dtype=np.uint8) * 255
                
                # 添加一些隨機文字
                cv2.putText(img, f"Sample {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(img, "手寫文字", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # 保存圖像
                img_path = data_dir / f"sample_{i:03d}.jpg"
                cv2.imwrite(str(img_path), img)
                
                # 創建標註
                annotation = {
                    "text": f"Sample {i} 手寫文字",
                    "script_type": "mixed",
                    "trajectory": np.random.rand(500, 2).tolist(),
                    "bbox": [50, 50, 200, 200]
                }
                
                # 保存標註
                ann_path = data_dir / f"sample_{i:03d}.json"
                with open(ann_path, 'w', encoding='utf-8') as f:
                    json.dump(annotation, f, ensure_ascii=False, indent=2)
        
        logger.info("Sample data created successfully!")
    
    def load_data(self):
        """加載數據"""
        
        # 創建數據集
        train_dataset = HandwritingDataset(
            self.config.train_data_dir, self.config, is_training=True
        )
        val_dataset = HandwritingDataset(
            self.config.val_data_dir, self.config, is_training=False
        )
        
        # 創建數據加載器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        logger.info(f"Loaded {len(train_dataset)} training samples")
        logger.info(f"Loaded {len(val_dataset)} validation samples")
    
    def train_with_software_rl(self, image: torch.Tensor, target_text: str) -> Dict:
        """使用軟件RL環境訓練單個樣本"""
        
        # 重置RL環境
        obs = self.software_rl_gym.reset(image, target_text)
        
        total_rl_reward = 0
        episode_steps = 0
        max_steps = 5  # 限制RL步數
        
        while not self.software_rl_gym.done and episode_steps < max_steps:
            # 隨機選擇動作 (簡化版本，實際應該用訓練好的策略)
            action = self.software_rl_gym.action_space.sample()
            
            # 執行動作
            next_obs, reward, done, info = self.software_rl_gym.step(action)
            
            total_rl_reward += reward
            episode_steps += 1
            
            obs = next_obs
        
        return {
            'total_rl_reward': total_rl_reward,
            'episode_steps': episode_steps,
            'final_prediction': info.get('ocr_result', {}).get('text', ''),
            'sensor_readings': info.get('sensor_readings', {})
        }
    
    def train_epoch(self):
        
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移動數據到設備
            images = batch['image'].to(self.device)
            trajectories = batch['trajectory'].to(self.device)
            
            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 計算損失
            # 這裡簡化處理，實際應用中需要更複雜的損失計算
            trajectory_loss = self.criterion_trajectory(outputs['trajectory'], trajectories)
            
            # 腳本類型損失 (簡化)
            script_targets = torch.randint(0, 3, (images.size(0),)).to(self.device)
            script_loss = self.criterion_script(outputs['script_logits'], script_targets)
            
            # 總損失
            loss = trajectory_loss + 0.1 * script_loss
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
            
            # 更新參數
            self.optimizer.step()
            
            # 統計
            total_loss += loss.item()
            
            # 更新進度條
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # 記錄到TensorBoard
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/Train_Epoch', avg_loss, self.current_epoch)
        
        return avg_loss
    
    def validate(self):
        """驗證模型"""
        
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # 移動數據到設備
                images = batch['image'].to(self.device)
                trajectories = batch['trajectory'].to(self.device)
                
                # 前向傳播
                outputs = self.model(images)
                
                # 計算損失
                trajectory_loss = self.criterion_trajectory(outputs['trajectory'], trajectories)
                script_targets = torch.randint(0, 3, (images.size(0),)).to(self.device)
                script_loss = self.criterion_script(outputs['script_logits'], script_targets)
                
                loss = trajectory_loss + 0.1 * script_loss
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/Val_Epoch', avg_loss, self.current_epoch)
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """保存檢查點"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config)
        }
        
        # 保存最新檢查點
        checkpoint_path = Path(self.config.checkpoint_dir) / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳檢查點
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with val_loss: {self.best_val_loss:.4f}")
        
        # 定期保存
        if self.current_epoch % self.config.save_every == 0:
            epoch_path = Path(self.config.checkpoint_dir) / f'epoch_{self.current_epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加載檢查點"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """主訓練循環"""
        
        logger.info("Starting training...")
        logger.info(f"Config: {self.config}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # 訓練
            train_loss = self.train_epoch()
            
            # 驗證
            if epoch % self.config.validate_every == 0:
                val_loss = self.validate()
                
                # 更新學習率
                self.scheduler.step()
                
                # 檢查是否是最佳模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # 保存檢查點
                self.save_checkpoint(is_best)
                
                # 早停
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, best_val_loss={self.best_val_loss:.4f}"
                )
            else:
                # 只保存檢查點
                self.save_checkpoint()
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        
        self.writer.close()
        logger.info("Training completed!")

def main():
    """主函數"""
    
    parser = argparse.ArgumentParser(description="OCR0712 Local Training System")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--create-data", action="store_true", help="Create sample data")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--script-type", choices=['traditional', 'simplified', 'mixed'], 
                       default='mixed', help="Script type to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    # 創建配置
    config = TrainingConfig(
        script_type=args.script_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # 創建訓練器
    trainer = OCR0712Trainer(config)
    
    # 創建示例數據
    if args.create_data:
        trainer.create_sample_data()
        return
    
    # 加載數據
    try:
        trainer.load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("Creating sample data...")
        trainer.create_sample_data()
        trainer.load_data()
    
    # 恢復訓練
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 開始訓練
    trainer.train()

if __name__ == "__main__":
    main()