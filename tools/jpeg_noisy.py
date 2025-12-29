import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class JpegSimulator(nn.Module):
    """
    模拟JPEG压缩的神经网络
    使用两个卷积层来模拟JPEG的块状压缩效果
    包含YUV420色度子采样步骤
    """
    def __init__(self, channels=3, block_size=8, use_yuv420=True):
        super(JpegSimulator, self).__init__()
        self.channels = channels
        self.block_size = block_size
        self.use_yuv420 = use_yuv420
        
        # RGB到YUV的转换矩阵 (BT.601标准)
        # Y =  0.299*R + 0.587*G + 0.114*B
        # U = -0.169*R - 0.331*G + 0.500*B + 0.5
        # V =  0.500*R - 0.419*G - 0.081*B + 0.5
        self.register_buffer('rgb_to_yuv', torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.169, -0.331, 0.500],
            [0.500, -0.419, -0.081]
        ], dtype=torch.float32))
        
        # YUV到RGB的转换矩阵
        # R = Y + 1.402*(V-0.5)
        # G = Y - 0.344*(U-0.5) - 0.714*(V-0.5)
        # B = Y + 1.772*(U-0.5)
        self.register_buffer('yuv_to_rgb', torch.tensor([
            [1.000, 0.000, 1.402],
            [1.000, -0.344, -0.714],
            [1.000, 1.772, 0.000]
        ], dtype=torch.float32))
        
        # 第一个卷积层: [B,C,H,W] -> [B,C*64,H/8,W/8]
        # 使用 kernel_size=8, stride=8 来分块
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels * block_size * block_size,
            kernel_size=block_size,
            stride=block_size,
            padding=0
        )

        # 使用ReLU引入非线性失真，近似量化带来的信息丢失/噪声
        self.relu = nn.ReLU(inplace=True)
        
        # 第二个卷积层: [B,C*64,H/8,W/8] -> [B,C,H,W]
        # 使用转置卷积来恢复原始尺寸
        self.conv2 = nn.ConvTranspose2d(
            in_channels=channels * block_size * block_size,
            out_channels=channels,
            kernel_size=block_size,
            stride=block_size,
            padding=0
        )
    
    def rgb_to_yuv_transform(self, rgb):
        """
        将RGB图像转换为YUV色彩空间（完全可微）
        rgb: [B, 3, H, W], 范围 [-1, 1]
        return: [B, 3, H, W], Y在[-1,1], U和V在[-1,1]左右
        """
        # 先转换到[0, 1]范围
        rgb_01 = (rgb + 1.0) / 2.0
        
        # [B, 3, H, W] -> [B, H, W, 3]
        rgb_permuted = rgb_01.permute(0, 2, 3, 1)
        
        # 矩阵乘法进行色彩空间转换
        yuv_permuted = torch.matmul(rgb_permuted, self.rgb_to_yuv.T)
        
        # U和V需要加0.5偏移到[0,1]范围（避免原地操作，保持可微）
        y = yuv_permuted[:, :, :, 0:1]
        u = yuv_permuted[:, :, :, 1:2] + 0.5
        v = yuv_permuted[:, :, :, 2:3] + 0.5
        yuv_permuted = torch.cat([y, u, v], dim=-1)
        
        # [B, H, W, 3] -> [B, 3, H, W]
        yuv = yuv_permuted.permute(0, 3, 1, 2)
        
        # 转换回[-1, 1]范围
        yuv = yuv * 2.0 - 1.0
        
        return yuv
    
    def yuv_to_rgb_transform(self, yuv):
        """
        将YUV图像转换为RGB色彩空间（完全可微）
        yuv: [B, 3, H, W], 范围 [-1, 1]
        return: [B, 3, H, W], 范围 [-1, 1]
        """
        # 先转换到[0, 1]范围
        yuv_01 = (yuv + 1.0) / 2.0
        
        # [B, 3, H, W] -> [B, H, W, 3]
        yuv_permuted = yuv_01.permute(0, 2, 3, 1)
        
        # U和V需要减去0.5偏移（避免原地操作，保持可微）
        y = yuv_permuted[:, :, :, 0:1]
        u = yuv_permuted[:, :, :, 1:2] - 0.5
        v = yuv_permuted[:, :, :, 2:3] - 0.5
        yuv_adjusted = torch.cat([y, u, v], dim=-1)
        
        # 矩阵乘法进行色彩空间转换
        rgb_permuted = torch.matmul(yuv_adjusted, self.yuv_to_rgb.T)
        
        # [B, H, W, 3] -> [B, 3, H, W]
        rgb = rgb_permuted.permute(0, 3, 1, 2)
        
        # 转换到[-1, 1]范围（不使用clamp，保持完全可微）
        rgb = rgb * 2.0 - 1.0
        
        return rgb
    
    def yuv420_subsample(self, yuv):
        """
        YUV420色度子采样
        对U和V通道进行2x2下采样，然后上采样恢复原始分辨率
        这会导致色度信息的损失，模拟JPEG的色度子采样
        yuv: [B, 3, H, W]
        return: [B, 3, H, W]
        """
        B, C, H, W = yuv.shape
        
        # 分离Y, U, V通道
        y = yuv[:, 0:1, :, :]  # [B, 1, H, W]
        u = yuv[:, 1:2, :, :]  # [B, 1, H, W]
        v = yuv[:, 2:3, :, :]  # [B, 1, H, W]
        
        # 对U和V通道进行2x2平均池化下采样
        u_down = F.avg_pool2d(u, kernel_size=2, stride=2)  # [B, 1, H/2, W/2]
        v_down = F.avg_pool2d(v, kernel_size=2, stride=2)  # [B, 1, H/2, W/2]
        
        # 使用双线性插值上采样恢复原始分辨率
        u_up = F.interpolate(u_down, size=(H, W), mode='bilinear', align_corners=False)
        v_up = F.interpolate(v_down, size=(H, W), mode='bilinear', align_corners=False)
        
        # 合并通道
        yuv_subsampled = torch.cat([y, u_up, v_up], dim=1)  # [B, 3, H, W]
        
        return yuv_subsampled
        
    def forward(self, x):
        """
        前向传播
        x: [B, C, H, W], RGB图像，范围 [-1, 1]
        return: [B, C, H, W]
        
        处理流程:
        RGB -> YUV -> YUV420子采样 -> 卷积(YUV空间) -> YUV -> RGB
        """
        if self.use_yuv420:
            # 步骤1: RGB -> YUV
            yuv = self.rgb_to_yuv_transform(x)
            
            # 步骤2: YUV420色度子采样
            yuv = self.yuv420_subsample(yuv)
            
            # 步骤3: 卷积模拟DCT和量化（在YUV空间进行）
            # 编码: [B,C,H,W] -> [B,C*64,H/8,W/8]
            encoded = self.relu(self.conv1(yuv))
            # 解码: [B,C*64,H/8,W/8] -> [B,C,H,W]
            yuv_out = self.conv2(encoded)
            
            # 步骤4: YUV -> RGB
            decoded = self.yuv_to_rgb_transform(yuv_out)
        else:
            # 不使用YUV420时，直接在RGB空间进行卷积
            encoded = self.relu(self.conv1(x))
            decoded = self.conv2(encoded)
        
        return decoded


def real_jpeg_compress(img_tensor, quality=60):
    """
    对图像张量进行真实的JPEG压缩
    img_tensor: [B, C, H, W], 范围 [-1, 1]
    quality: JPEG压缩质量 (1-100)
    return: [B, C, H, W], 范围 [-1, 1]
    """
    batch_size = img_tensor.size(0)
    compressed_imgs = []
    
    for i in range(batch_size):
        # 反归一化到 [0, 255]
        img = img_tensor[i].cpu().clone()
        img = (img + 1.0) / 2.0  # [-1,1] -> [0,1]
        img = (img * 255).clamp(0, 255).byte()
        
        # 转换为PIL图像
        img_pil = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')
        
        # JPEG压缩
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # 读取压缩后的图像
        img_compressed = Image.open(buffer).convert('RGB')
        
        # 转回张量并归一化到 [-1, 1]
        img_tensor_compressed = torch.from_numpy(
            __import__('numpy').array(img_compressed)
        ).permute(2, 0, 1).float() / 255.0
        img_tensor_compressed = img_tensor_compressed * 2.0 - 1.0  # [0,1] -> [-1,1]
        
        compressed_imgs.append(img_tensor_compressed)
    
    return torch.stack(compressed_imgs).to(img_tensor.device)


class JpegDataset(Dataset):
    """复用train.py的数据集加载方式"""
    def __init__(self, data_path):
        self.image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomCrop(size=(config.H, config.W)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img


def train_jpeg_simulator():
    """训练JPEG模拟器"""
    # 创建数据集和数据加载器
    train_dataset = JpegDataset(config.train_data_path)
    val_dataset = JpegDataset(config.val_data_path)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    # 创建模型
    model = JpegSimulator(channels=3, block_size=8)
    model = model.to(config.device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 检查点路径
    jpeg_ckpt_path = os.path.join(config.checkpoint_path, "jpeg_simulator.pth")
    
    # 加载检查点（如果存在）
    start_epoch = 0
    if os.path.exists(jpeg_ckpt_path):
        ckpt = torch.load(jpeg_ckpt_path, map_location=config.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print(f"从epoch {start_epoch} 恢复JPEG模拟器训练")
    
    # 训练循环
    num_epochs = config.num_epochs  # JPEG模拟器训练的epoch数
    for epoch in range(start_epoch, num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        
        for batch_idx, img in enumerate(train_loader):
            img = img.to(config.device)
            
            # 真实JPEG压缩
            with torch.no_grad():
                jpeg_img = real_jpeg_compress(img, quality=config.jpeg_quality)
            
            # 模拟JPEG压缩
            simulated_img = model(img)
            
            # 计算损失: 模拟压缩与真实压缩的MSE
            loss = criterion(simulated_img, jpeg_img)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % config.log_interval_batch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}")
        
        train_avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] 训练平均损失: {train_avg_loss:.6f}")
        
        # 验证阶段
        model.eval()
        val_total_loss = 0.0
        first_val_batch = None
        with torch.no_grad():
            for batch_idx, img in enumerate(val_loader):
                img = img.to(config.device)
                jpeg_img = real_jpeg_compress(img, quality=config.jpeg_quality)
                simulated_img = model(img)
                loss = criterion(simulated_img, jpeg_img)
                val_total_loss += loss.item()

                if batch_idx == 0:
                    first_val_batch = (img.detach(), jpeg_img.detach(), simulated_img.detach())
        
        val_avg_loss = val_total_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] 验证平均损失: {val_avg_loss:.6f}")

        # 保存本epoch的验证样例图片（png）
        if first_val_batch is not None:
            out_root = os.path.join("inference_results", "jpeg_simulator", f"epoch_{epoch+1}")
            os.makedirs(out_root, exist_ok=True)

            img, jpeg_img, simulated_img = first_val_batch
            save_n = min(getattr(config, "save_eval_number", 2), img.size(0))
            for i in range(save_n):
                src = (img[i].cpu() + 1.0) / 2.0
                tgt = (jpeg_img[i].cpu() + 1.0) / 2.0
                pred = (simulated_img[i].cpu() + 1.0) / 2.0

                torchvision.utils.save_image(src, os.path.join(out_root, f"src_{i}.png"), normalize=False)
                torchvision.utils.save_image(tgt, os.path.join(out_root, f"jpeg_{i}.png"), normalize=False)
                torchvision.utils.save_image(pred, os.path.join(out_root, f"sim_{i}.png"), normalize=False)
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_avg_loss,
                'val_loss': val_avg_loss,
            }, jpeg_ckpt_path)
            print(f"保存检查点: {jpeg_ckpt_path}")
    
    print("JPEG模拟器训练完成!")
    return model


def load_jpeg_simulator(device=None):
    """加载预训练的JPEG模拟器"""
    if device is None:
        device = config.device
    
    model = JpegSimulator(channels=3, block_size=8)
    jpeg_ckpt_path = os.path.join(config.checkpoint_path, "jpeg_simulator.pth")
    
    if os.path.exists(jpeg_ckpt_path):
        ckpt = torch.load(jpeg_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"加载JPEG模拟器: {jpeg_ckpt_path}")
    else:
        print("警告: 未找到预训练的JPEG模拟器，使用随机初始化")
    
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    train_jpeg_simulator()
