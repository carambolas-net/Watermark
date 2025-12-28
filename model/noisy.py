import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import numpy as np
from PIL import Image
import io
import math

class Noisy(nn.Module):
    def __init__(self, config):
        super(Noisy, self).__init__()
        self.config = config
        self.H = config.H
        self.W = config.W
        self.noisy_sequence = config.noise_sequence
        self.quality = config.jpeg_quality
     # 1. DCT Filters
        self.size = 8
        u = torch.arange(self.size).float().unsqueeze(1)
        x = torch.arange(self.size).float().unsqueeze(0)
        mat = torch.cos((2 * x + 1) * u * math.pi / (2 * self.size)) * math.sqrt(2 / self.size)
        mat[0] *= 1 / math.sqrt(2)
        self.register_buffer('filters', torch.einsum('ax,by->abxy', mat, mat).reshape(-1, 1, self.size, self.size))
        
        # 2. Quantization Tables
        self.luma_q_base = torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ]).float()

        self.chroma_q_base = torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ]).float()

        scale = 5000 / self.quality if self.quality < 50 else 200 - 2 * self.quality
        def get_q_table(base_table):
            q = torch.floor((base_table * scale + 50) / 100)
            q[q <= 0] = 1
            q[q > 255] = 255
            return q.reshape(-1)
            
        self.register_buffer('luma_q', get_q_table(self.luma_q_base))
        self.register_buffer('chroma_q', get_q_table(self.chroma_q_base))

    def rgb_to_ycbcr(self, x):
        y  = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        cb = -0.1687 * x[:, 0] - 0.3313 * x[:, 1] + 0.5 * x[:, 2] + 128
        cr = 0.5 * x[:, 0] - 0.4187 * x[:, 1] - 0.0813 * x[:, 2] + 128
        return y.unsqueeze(1), cb.unsqueeze(1), cr.unsqueeze(1)

    def ycbcr_to_rgb(self, y, cb, cr):
        cb = cb - 128
        cr = cr - 128
        r = y + 1.402 * cr
        g = y - 0.34414 * cb - 0.71414 * cr
        b = y + 1.772 * cb
        return torch.cat([r, g, b], dim=1).clamp(0, 255)

    def process_channel(self, x, q_table):
        y = F.conv2d(x, self.filters, stride=self.size)
        q = q_table.view(1, 64, 1, 1)
        y_quantized = y + (torch.round(y / q) * q - y).detach()
        out = F.conv_transpose2d(y_quantized, self.filters, stride=self.size)
        return out

    def jpeg_compression(self, x):
        x = ((x + 1)/2) * 255
        y, cb, cr = self.rgb_to_ycbcr(x)
        
        y_rec = self.process_channel(y, self.luma_q)
        cb_rec = self.process_channel(cb, self.chroma_q)
        cr_rec = self.process_channel(cr, self.chroma_q)
        
        out = self.ycbcr_to_rgb(y_rec, cb_rec, cr_rec)
        return (out / 255.0)*2 -1
        
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
        # result = image + (result - image).detach()
        
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
