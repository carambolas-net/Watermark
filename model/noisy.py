import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import numpy as np
from PIL import Image
import io

class Noisy(nn.Module):
    def __init__(self, config):
        super(Noisy, self).__init__()
        self.config = config
        self.H = config.H
        self.W = config.W
        self.noisy_sequence = config.noise_sequence
        self.dct_y = config.DCT_Y  # Y通道保留的低频系数
        self.dct_uv = config.DCT_UV  # UV通道保留的低频系数
        
        # 预计算DCT变换核和低频掩码
        self.register_buffer('dct_kernel', self._create_dct_kernel())
        self.register_buffer('idct_kernel', self._create_idct_kernel())
        self.register_buffer('freq_mask_y', self._create_freq_mask(self.dct_y))
        self.register_buffer('freq_mask_uv', self._create_freq_mask(self.dct_uv))
        
        # RGB <-> YUV 转换矩阵
        self.register_buffer('rgb_to_yuv', torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ]).float())
        self.register_buffer('yuv_to_rgb', torch.tensor([
            [1.0, 0.0, 1.13983],
            [1.0, -0.39465, -0.58060],
            [1.0, 2.03211, 0.0]
        ]).float())
    
    def _create_dct_kernel(self):
        """创建8x8 DCT变换卷积核，输出64通道"""
        N = 8
        kernel = torch.zeros(64, 1, 8, 8)
        for u in range(N):
            for v in range(N):
                alpha_u = np.sqrt(1/N) if u == 0 else np.sqrt(2/N)
                alpha_v = np.sqrt(1/N) if v == 0 else np.sqrt(2/N)
                for x in range(N):
                    for y in range(N):
                        kernel[u * N + v, 0, x, y] = alpha_u * alpha_v * \
                            np.cos(np.pi * u * (2*x + 1) / (2*N)) * \
                            np.cos(np.pi * v * (2*y + 1) / (2*N))
        return kernel
    
    def _create_idct_kernel(self):
        """创建IDCT逆变换卷积核"""
        N = 8
        kernel = torch.zeros(1, 64, 8, 8)
        for u in range(N):
            for v in range(N):
                alpha_u = np.sqrt(1/N) if u == 0 else np.sqrt(2/N)
                alpha_v = np.sqrt(1/N) if v == 0 else np.sqrt(2/N)
                for x in range(N):
                    for y in range(N):
                        kernel[0, u * N + v, x, y] = alpha_u * alpha_v * \
                            np.cos(np.pi * u * (2*x + 1) / (2*N)) * \
                            np.cos(np.pi * v * (2*y + 1) / (2*N))
        return kernel
    
    def _create_freq_mask(self, n):
        """创建低频掩码，保留前n个zigzag顺序的系数"""
        # Zigzag顺序索引
        zigzag_order = [
            0,  1,  8, 16,  9,  2,  3, 10,
           17, 24, 32, 25, 18, 11,  4,  5,
           12, 19, 26, 33, 40, 48, 41, 34,
           27, 20, 13,  6,  7, 14, 21, 28,
           35, 42, 49, 56, 57, 50, 43, 36,
           29, 22, 15, 23, 30, 37, 44, 51,
           58, 59, 52, 45, 38, 31, 39, 46,
           53, 60, 61, 54, 47, 55, 62, 63
        ]
        mask = torch.zeros(64)
        for i in range(min(n, 64)):
            mask[zigzag_order[i]] = 1.0
        return mask.view(1, 64, 1, 1)
        
    #@torch.no_grad()
    def forward(self, encoded_image):
        """
        对编码后的图像添加噪声
        Args:
            encoded_image: 形状为 (B, C, H, W) 的图像张量
            noise_sequence: 按顺序应用的噪声类型列表，如 ["gaussian", "jpeg", "cropout"]
        Returns:
            noisy_image: 形状为 (B, C, H', W') 的图像张量
        """
        result = encoded_image
        for noise in self.noisy_sequence:
            if noise == "none":
                continue
            elif noise == "gaussian":
                result = self.add_gaussian_noise(result)
            elif noise == "jpeg":
                result = self.jpeg_compression(result)
            elif noise == "jpeg_real":
                result = self.jpeg_real(result)
            elif noise == "dropout":
                result = self.dropout(result)
            elif noise == "crop":
                result = self.random_crop(result)
            elif noise == "cropout":
                result = self.random_cropout(result)
        
        return result
    
    def add_gaussian_noise(self, image):
        """添加高斯噪声"""
        noise = torch.randn_like(image) * self.config.gaussian_std
        noisy_image = image + noise
        return torch.clamp(noisy_image, -1, 1)
    
    def _rgb_to_yuv(self, rgb):
        """RGB转YUV，输入输出范围[-1,1]"""
        # rgb: (B, 3, H, W) -> (B, H, W, 3)
        rgb_t = rgb.permute(0, 2, 3, 1)
        # 先转到[0,1]范围进行转换
        rgb_01 = (rgb_t + 1) / 2
        yuv = torch.matmul(rgb_01, self.rgb_to_yuv.T)
        # YUV范围: Y[0,1], U[-0.436,0.436], V[-0.615,0.615]
        # 归一化到[-1,1]
        yuv[..., 0] = yuv[..., 0] * 2 - 1  # Y: [0,1] -> [-1,1]
        yuv[..., 1] = yuv[..., 1] / 0.436  # U: [-0.436,0.436] -> [-1,1]
        yuv[..., 2] = yuv[..., 2] / 0.615  # V: [-0.615,0.615] -> [-1,1]
        return yuv.permute(0, 3, 1, 2)  # (B, 3, H, W)
    
    def _yuv_to_rgb(self, yuv):
        """YUV转RGB，输入输出范围[-1,1]"""
        # yuv: (B, 3, H, W) -> (B, H, W, 3)
        yuv_t = yuv.permute(0, 2, 3, 1).clone()
        # 反归一化
        yuv_t[..., 0] = (yuv_t[..., 0] + 1) / 2  # Y: [-1,1] -> [0,1]
        yuv_t[..., 1] = yuv_t[..., 1] * 0.436    # U: [-1,1] -> [-0.436,0.436]
        yuv_t[..., 2] = yuv_t[..., 2] * 0.615    # V: [-1,1] -> [-0.615,0.615]
        rgb_01 = torch.matmul(yuv_t, self.yuv_to_rgb.T)
        # [0,1] -> [-1,1]
        rgb = rgb_01 * 2 - 1
        return rgb.permute(0, 3, 1, 2)  # (B, 3, H, W)

    def jpeg_compression(self, image):
        """
        使用DCT变换模拟JPEG压缩
        1. RGB转YUV
        2. 8x8块DCT变换（使用卷积实现）
        3. Y通道保留DCT_Y个低频系数，UV通道保留DCT_UV个低频系数
        4. IDCT逆变换
        5. YUV转RGB
        """
        B, C, H, W = image.shape
        
        # RGB转YUV
        if C == 3:
            image_yuv = self._rgb_to_yuv(image)
        else:
            image_yuv = image
        
        # 确保尺寸是8的倍数，使用ZeroPad2d
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            zero_pad = nn.ZeroPad2d((0, pad_w, 0, pad_h))
            image_yuv = zero_pad(image_yuv)
        
        _, _, H_pad, W_pad = image_yuv.shape
        
        # 对每个通道分别处理，Y和UV使用不同的频率掩码
        result_channels = []
        for c in range(C):
            channel = image_yuv[:, c:c+1, :, :]  # (B, 1, H, W)
            
            # DCT变换: 使用8x8卷积，步长8
            dct_coef = F.conv2d(channel, self.dct_kernel, stride=8)  # (B, 64, H/8, W/8)
            
            # 根据通道选择频率掩码：Y通道用freq_mask_y，UV通道用freq_mask_uv
            if c == 0:  # Y通道
                dct_coef = dct_coef * self.freq_mask_y
            else:  # U, V通道
                dct_coef = dct_coef * self.freq_mask_uv
            
            # IDCT逆变换: 使用转置卷积
            reconstructed = F.conv_transpose2d(dct_coef, self.idct_kernel.permute(1, 0, 2, 3), stride=8)
            result_channels.append(reconstructed)
        
        result_yuv = torch.cat(result_channels, dim=1)  # (B, C, H, W)
        
        # 裁剪回原尺寸
        if pad_h > 0 or pad_w > 0:
            result_yuv = result_yuv[:, :, :H, :W]
        
        # YUV转RGB
        if C == 3:
            result = self._yuv_to_rgb(result_yuv)
        else:
            result = result_yuv
        
        return torch.clamp(result, -1, 1)
    
    def jpeg_real(self, image):
        """
        使用PIL进行真实的JPEG压缩
        Args:
            image: 形状为 (B, C, H, W) 的图像张量，值范围 [-1, 1]
        Returns:
            压缩后的图像张量
        """
        B, C, H, W = image.shape
        device = image.device
        
        # 获取JPEG质量参数（默认75）
        quality = getattr(self.config, 'jpeg_quality')
        
        # 将值从 [-1, 1] 转换到 [0, 255]
        image_uint8 = ((image + 1) * 127.5).clamp(0, 255)
        
        result_list = []
        for b in range(B):
            img_np = image_uint8[b].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            
            # 转换为PIL图像
            if C == 1:
                pil_img = Image.fromarray(img_np.squeeze(), mode='L')
            else:
                pil_img = Image.fromarray(img_np, mode='RGB')
            
            # JPEG压缩和解压缩
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            pil_img_compressed = Image.open(buffer)
            
            # 转回numpy数组
            img_compressed_np = np.array(pil_img_compressed)
            if C == 1:
                img_compressed_np = img_compressed_np[:, :, np.newaxis]
            
            # 转回tensor，值范围 [0, 255] -> [-1, 1]
            img_tensor = torch.from_numpy(img_compressed_np).permute(2, 0, 1).float()
            img_tensor = (img_tensor / 127.5) - 1
            result_list.append(img_tensor)
        
        result = torch.stack(result_list, dim=0).to(device)
        
        # 使用直通估计器保持梯度
        result = image + (result - image).detach()
        
        return result
    
    # def dropout(self, image):
    #     """随机dropout像素"""
    #     if not self.training:
    #         return image
        
    #     mask = torch.rand_like(image) > self.config.dropout_prob
    #     return image * mask.float()
    
    def random_crop(self, image):
        """
        随机裁剪，不resize回原尺寸
        """
        B, C, H, W = image.shape

        # 随机裁剪比例
        crop_ratio = np.random.uniform(self.config.crop_ratio_min, self.config.crop_ratio_max)
        new_h = int(H * crop_ratio)
        new_w = int(W * crop_ratio)
        cropped = v2.RandomCrop(size=(new_h, new_w))(image)
        
        return cropped
    
    # def random_cropout(self, image):
    #     """
    #     随机遮挡一块区域（填充为0或随机值）
    #     """
    #     B, C, H, W = image.shape
    #     result = image.clone()
        
    #     # 随机遮挡区域大小
    #     cropout_ratio = np.random.uniform(self.config.cropout_ratio_min, self.config.cropout_ratio_max)
    #     cropout_h = int(H * cropout_ratio)
    #     cropout_w = int(W * cropout_ratio)
        
    #     # 随机起始位置
    #     top = np.random.randint(0, H - cropout_h + 1)
    #     left = np.random.randint(0, W - cropout_w + 1)
        
    #     # 遮挡区域填充为0或随机值（范围[-1,1]）
    #     if np.random.rand() > 0.5:
    #         result[:, :, top:top+cropout_h, left:left+cropout_w] = 0
    #     else:
    #         result[:, :, top:top+cropout_h, left:left+cropout_w] = torch.rand(B, C, cropout_h, cropout_w).to(image.device) * 2 - 1
        
    #     return result
