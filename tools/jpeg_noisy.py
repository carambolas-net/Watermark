import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DifferentiableJPEG(nn.Module):
    """
    可微分的JPEG模拟器
    流程: RGB -> YUV -> DCT -> Quantization -> IDCT -> YUV -> RGB
    全程可微分，使用卷积实现DCT/IDCT
    """
    
    def __init__(self, block_size=8):
        super(DifferentiableJPEG, self).__init__()
        self.block_size = block_size
        
        # 初始化DCT卷积核
        self.register_buffer('dct_kernel', self._create_dct_kernel())
        # 初始化IDCT卷积核
        self.register_buffer('idct_kernel', self._create_idct_kernel())
        
        # RGB to YUV 转换矩阵
        # Y = 0.299*R + 0.587*G + 0.114*B
        # U = -0.14713*R - 0.28886*G + 0.436*B + 0.5
        # V = 0.615*R - 0.51499*G - 0.10001*B + 0.5
        rgb_to_yuv_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ], dtype=torch.float32)
        self.register_buffer('rgb_to_yuv_matrix', rgb_to_yuv_matrix)
        
        # YUV to RGB 转换矩阵 (逆矩阵)
        yuv_to_rgb_matrix = torch.tensor([
            [1.0, 0.0, 1.13983],
            [1.0, -0.39465, -0.58060],
            [1.0, 2.03211, 0.0]
        ], dtype=torch.float32)
        self.register_buffer('yuv_to_rgb_matrix', yuv_to_rgb_matrix)
        
    def _create_dct_kernel(self):
        """
        创建DCT卷积核
        DCT-II 公式: X[k] = sum_{n=0}^{N-1} x[n] * cos(pi/N * (n + 0.5) * k)
        """
        block_size = self.block_size
        # 创建1D DCT基函数
        dct_basis = torch.zeros(block_size, block_size)
        for k in range(block_size):
            for n in range(block_size):
                if k == 0:
                    dct_basis[k, n] = 1.0 / np.sqrt(block_size)
                else:
                    dct_basis[k, n] = np.sqrt(2.0 / block_size) * np.cos(
                        np.pi * (2 * n + 1) * k / (2 * block_size)
                    )
        
        # 创建2D DCT核 [64, 1, 8, 8]
        dct_kernel = torch.zeros(block_size * block_size, 1, block_size, block_size)
        for i in range(block_size):
            for j in range(block_size):
                # 2D DCT基 = 外积 of 1D DCT基
                basis_2d = torch.outer(dct_basis[i], dct_basis[j])
                dct_kernel[i * block_size + j, 0] = basis_2d
                
        return dct_kernel
    
    def _create_idct_kernel(self):
        """
        创建IDCT卷积核
        IDCT是DCT的转置
        """
        block_size = self.block_size
        # 创建1D IDCT基函数 (DCT的转置)
        idct_basis = torch.zeros(block_size, block_size)
        for n in range(block_size):
            for k in range(block_size):
                if k == 0:
                    idct_basis[n, k] = 1.0 / np.sqrt(block_size)
                else:
                    idct_basis[n, k] = np.sqrt(2.0 / block_size) * np.cos(
                        np.pi * (2 * n + 1) * k / (2 * block_size)
                    )
        
        # 创建2D IDCT核 [64, 1, 8, 8] for transpose conv
        idct_kernel = torch.zeros(block_size * block_size, 1, block_size, block_size)
        for i in range(block_size):
            for j in range(block_size):
                basis_2d = torch.outer(idct_basis[:, i], idct_basis[:, j])
                idct_kernel[i * block_size + j, 0] = basis_2d
                
        return idct_kernel
    
    def rgb_to_yuv(self, rgb):
        """
        RGB转YUV
        输入: [B, 3, H, W] RGB图像，范围[-1, 1]
        输出: [B, 3, H, W] YUV图像，范围[-1, 1]
        """
        B, C, H, W = rgb.shape
        # 重排为 [B, H, W, 3]
        rgb_hwc = rgb.permute(0, 2, 3, 1)
        # 矩阵乘法转换
        yuv_hwc = torch.matmul(rgb_hwc, self.rgb_to_yuv_matrix.T)
        # 重排回 [B, 3, H, W]
        yuv = yuv_hwc.permute(0, 3, 1, 2)
        return yuv
    
    def yuv_to_rgb(self, yuv):
        """
        YUV转RGB
        输入: [B, 3, H, W] YUV图像，范围[-1, 1]
        输出: [B, 3, H, W] RGB图像，范围[-1, 1]
        """
        B, C, H, W = yuv.shape
        # 重排为 [B, H, W, 3]
        yuv_hwc = yuv.permute(0, 2, 3, 1)
        # 矩阵乘法转换
        rgb_hwc = torch.matmul(yuv_hwc, self.yuv_to_rgb_matrix.T)
        # 重排回 [B, 3, H, W]
        rgb = rgb_hwc.permute(0, 3, 1, 2)
        return rgb
    
    def dct_2d(self, x):
        """
        使用卷积实现2D DCT
        输入: [B, C, H, W] 
        输出: [B, C, H, W] DCT系数 (按8x8块组织)
        """
        B, C, H, W = x.shape
        block_size = self.block_size
        
        # 确保尺寸是block_size的倍数
        assert H % block_size == 0 and W % block_size == 0, \
            f"Image dimensions must be multiples of {block_size}"
        
        # 分离通道处理
        outputs = []
        for c in range(C):
            # 取单通道 [B, 1, H, W]
            x_c = x[:, c:c+1, :, :]
            
            # 使用卷积计算DCT，stride=8实现分块
            # 输出: [B, 64, H//8, W//8]
            dct_coeffs = F.conv2d(x_c, self.dct_kernel, stride=block_size)
            
            # 重排DCT系数到空间域 [B, 64, H//8, W//8] -> [B, 1, H, W]
            B_out, _, H_blocks, W_blocks = dct_coeffs.shape
            # reshape to [B, 8, 8, H//8, W//8]
            dct_coeffs = dct_coeffs.view(B_out, block_size, block_size, H_blocks, W_blocks)
            # permute to [B, H//8, 8, W//8, 8]
            dct_coeffs = dct_coeffs.permute(0, 3, 1, 4, 2)
            # reshape to [B, 1, H, W]
            dct_coeffs = dct_coeffs.reshape(B_out, 1, H, W)
            
            outputs.append(dct_coeffs)
        
        # 合并通道 [B, C, H, W]
        return torch.cat(outputs, dim=1)
    
    def idct_2d(self, x):
        """
        使用卷积实现2D IDCT
        输入: [B, C, H, W] DCT系数
        输出: [B, C, H, W] 空间域图像
        """
        B, C, H, W = x.shape
        block_size = self.block_size
        
        outputs = []
        for c in range(C):
            x_c = x[:, c:c+1, :, :]
            
            H_blocks = H // block_size
            W_blocks = W // block_size
            
            # 重排系数 [B, 1, H, W] -> [B, 64, H//8, W//8]
            # reshape to [B, H//8, 8, W//8, 8]
            x_reshaped = x_c.view(B, H_blocks, block_size, W_blocks, block_size)
            # permute to [B, 8, 8, H//8, W//8]
            x_reshaped = x_reshaped.permute(0, 2, 4, 1, 3)
            # reshape to [B, 64, H//8, W//8]
            x_reshaped = x_reshaped.reshape(B, block_size * block_size, H_blocks, W_blocks)
            
            # 使用转置卷积实现IDCT
            # idct_kernel: [64, 1, 8, 8]
            spatial = F.conv_transpose2d(x_reshaped, self.idct_kernel, stride=block_size)
            
            outputs.append(spatial)
        
        return torch.cat(outputs, dim=1)
    
    def apply_quantization(self, dct_coeffs, q_table):
        """
        应用量化表 (用户自定义接口)
        输入: 
            dct_coeffs: [B, C, H, W] DCT系数
            q_table: 用户提供的量化表，可以是:
                - [8, 8] 单个量化表，应用于所有通道
                - [C, 8, 8] 每个通道一个量化表
                - [B, C, 8, 8] 每个样本每个通道一个量化表
                - 或者任何可微分的函数/模块
        输出: [B, C, H, W] 量化后的系数
        """
        B, C, H, W = dct_coeffs.shape
        block_size = self.block_size
        H_blocks = H // block_size
        W_blocks = W // block_size
        
        # 如果q_table是张量，进行除法操作
        if isinstance(q_table, torch.Tensor):
            # 扩展q_table到匹配的形状
            if q_table.dim() == 2:
                # [8, 8] -> [1, 1, 8, 8]
                q_table = q_table.unsqueeze(0).unsqueeze(0)
            elif q_table.dim() == 3:
                # [C, 8, 8] -> [1, C, 8, 8]
                q_table = q_table.unsqueeze(0)
            
            # 将q_table平铺到整个图像
            # [B, C, 8, 8] -> [B, C, H, W]
            q_table_tiled = q_table.repeat(1, 1, H_blocks, W_blocks)
            
            # 量化操作: 除以量化表
            quantized = dct_coeffs / (q_table_tiled + 1e-10)
            
            return quantized
        else:
            # 如果是可调用对象(如nn.Module)，直接调用
            return q_table(dct_coeffs)
    
    def apply_dequantization(self, quantized_coeffs, q_table):
        """
        应用反量化 (用户自定义接口)
        输入:
            quantized_coeffs: [B, C, H, W] 量化后的系数
            q_table: 量化表
        输出: [B, C, H, W] 反量化后的DCT系数
        """
        B, C, H, W = quantized_coeffs.shape
        block_size = self.block_size
        H_blocks = H // block_size
        W_blocks = W // block_size
        
        if isinstance(q_table, torch.Tensor):
            if q_table.dim() == 2:
                q_table = q_table.unsqueeze(0).unsqueeze(0)
            elif q_table.dim() == 3:
                q_table = q_table.unsqueeze(0)
            
            q_table_tiled = q_table.repeat(1, 1, H_blocks, W_blocks)
            
            # 反量化: 乘以量化表
            dequantized = quantized_coeffs * q_table_tiled
            
            return dequantized
        else:
            return q_table(quantized_coeffs)
    
    def forward(self, rgb_image, q_table):
        """
        完整的可微分JPEG流程
        输入:
            rgb_image: [B, 3, H, W] RGB图像，范围[-1, 1]
            q_table: 量化表，用户提供
        输出:
            rgb_output: [B, 3, H, W] 处理后的RGB图像，范围[-1, 1]
        """
        # Step 1: RGB -> YUV
        yuv = self.rgb_to_yuv(rgb_image)
        
        # Step 2: DCT (使用卷积实现)
        A = self.dct_2d(yuv)
        
        # Step 3: 应用量化表 (用户接口)
        B = self.apply_quantization(A, q_table)
        
        # Step 4: 应用反量化
        C_dct = self.apply_dequantization(B, q_table)
        
        # Step 5: IDCT (使用卷积实现)
        C = self.idct_2d(C_dct)
        
        # Step 6: YUV -> RGB
        rgb_output = self.yuv_to_rgb(C)
        
        return rgb_output
    
    def forward_with_intermediate(self, rgb_image, q_table):
        """
        返回中间结果的forward，用于调试
        """
        yuv = self.rgb_to_yuv(rgb_image)
        A = self.dct_2d(yuv)
        B = self.apply_quantization(A, q_table)
        C_dct = self.apply_dequantization(B, q_table)
        C = self.idct_2d(C_dct)
        rgb_output = self.yuv_to_rgb(C)
        
        return {
            'yuv': yuv,
            'A': A,  # DCT系数
            'B': B,  # 量化后
            'C_dct': C_dct,  # 反量化后
            'C': C,  # IDCT后的YUV
            'rgb_output': rgb_output
        }


class LearnableQuantizationTable(nn.Module):
    """
    可学习的量化表示例
    用户可以参考此实现自定义q_table
    """
    def __init__(self, num_channels=3, init_quality=50):
        super(LearnableQuantizationTable, self).__init__()
        
        # 标准JPEG亮度量化表
        standard_luma_table = torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float32)
        
        # 标准JPEG色度量化表
        standard_chroma_table = torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float32)
        
        # 根据质量调整
        if init_quality < 50:
            scale = 5000 / init_quality
        else:
            scale = 200 - 2 * init_quality
        scale = scale / 100.0
        
        # 初始化可学习参数 [3, 8, 8]
        init_tables = torch.stack([
            standard_luma_table * scale,    # Y通道
            standard_chroma_table * scale,  # U通道
            standard_chroma_table * scale   # V通道
        ])
        
        # 使用log空间参数化确保正值
        self.log_q_table = nn.Parameter(torch.log(init_tables + 1e-10))
    
    def forward(self):
        """返回量化表"""
        return torch.exp(self.log_q_table)


# 使用示例
if __name__ == "__main__":
    # 创建模型
    jpeg_layer = DifferentiableJPEG(block_size=8)
    
    # 创建测试输入 [B, C, H, W]，范围[-1, 1]
    batch_size = 2
    height, width = 64, 64  # 必须是8的倍数
    test_input = torch.rand(batch_size, 3, height, width) * 2 - 1  # 范围[-1, 1]
    
    # 方法1: 使用固定量化表
    q_table = torch.ones(8, 8) * 10.0  # 简单的均匀量化表
    q_table.requires_grad = True  # 如果需要梯度
    
    output = jpeg_layer(test_input, q_table)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # 验证可微分性
    loss = output.mean()
    loss.backward()
    print(f"q_table gradient shape: {q_table.grad.shape}")
    print(f"Gradient exists: {q_table.grad is not None}")
    
    # 方法2: 使用可学习量化表
    learnable_q = LearnableQuantizationTable(num_channels=3, init_quality=75)
    q_table_learned = learnable_q()
    
    output2 = jpeg_layer(test_input, q_table_learned)
    loss2 = output2.mean()
    loss2.backward()
    print(f"\nLearnable q_table gradient exists: {learnable_q.log_q_table.grad is not None}")
    
    # 方法3: 获取中间结果
    results = jpeg_layer.forward_with_intermediate(test_input, q_table)
    
    print("\n中间结果形状:")
    for key, value in results.items():
        print(f"  {key}: {value.shape}")
