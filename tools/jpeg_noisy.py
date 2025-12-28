import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import io
import os
import numpy as np
from PIL import Image
from torchvision import transforms

class RealJPEG(nn.Module):
    def __init__(self, size=8, quality=50, subsampling='4:2:0'):
        super().__init__()
        self.size = size
        self.subsampling = subsampling
        
        # 1. DCT Filters
        u = torch.arange(size).float().unsqueeze(1)
        x = torch.arange(size).float().unsqueeze(0)
        mat = torch.cos((2 * x + 1) * u * math.pi / (2 * size)) * math.sqrt(2 / size)
        mat[0] *= 1 / math.sqrt(2)
        self.register_buffer('filters', torch.einsum('ax,by->abxy', mat, mat).reshape(-1, 1, size, size))
        
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

        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
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

    def forward(self, x):
        x = x * 255.0
        y, cb, cr = self.rgb_to_ycbcr(x)
        
        y_rec = self.process_channel(y, self.luma_q)
        
        if self.subsampling == '4:2:0':
            cb_sub = F.avg_pool2d(cb, kernel_size=2, stride=2)
            cr_sub = F.avg_pool2d(cr, kernel_size=2, stride=2)
            cb_rec_sub = self.process_channel(cb_sub, self.chroma_q)
            cr_rec_sub = self.process_channel(cr_sub, self.chroma_q)
            cb_rec = F.interpolate(cb_rec_sub, size=y.shape[2:], mode='bilinear', align_corners=False)
            cr_rec = F.interpolate(cr_rec_sub, size=y.shape[2:], mode='bilinear', align_corners=False)
        else:
            cb_rec = self.process_channel(cb, self.chroma_q)
            cr_rec = self.process_channel(cr, self.chroma_q)
        
        out = self.ycbcr_to_rgb(y_rec, cb_rec, cr_rec)
        return out / 255.0

def real_jpeg_process(x, quality=50, subsampling_mode='4:2:0'):
    pil_sub = 0 if subsampling_mode == '4:4:4' else 2
    out = []
    for t in x.cpu():
        buf = io.BytesIO()
        img_pil = transforms.ToPILImage()(t.clamp(0, 1))
        img_pil.save(buf, format='JPEG', quality=quality, subsampling=pil_sub)
        out.append(transforms.ToTensor()(Image.open(buf)))
    return torch.stack(out).to(x.device)

if __name__ == "__main__":
    IMG_PATH = "img.jpg"
    QUALITY = 50
    SUBSAMPLING = '4:2:0'
    
    if not os.path.exists(IMG_PATH):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            arr[i, :, 0] = i 
            arr[:, i, 1] = 255 - i
        Image.fromarray(arr).save(IMG_PATH)
        
    print(f"Processing: {IMG_PATH} | Q={QUALITY} | Sub={SUBSAMPLING}")
    img = Image.open(IMG_PATH).convert('RGB')
    align = 16 if SUBSAMPLING == '4:2:0' else 8
    W, H = (s // align * align for s in img.size)
    x = transforms.ToTensor()(img.resize((W, H))).unsqueeze(0)
    
    # Run
    model = RealJPEG(quality=QUALITY, subsampling=SUBSAMPLING)
    model.eval()
    with torch.no_grad():
        recon = model(x)
        real = real_jpeg_process(x, quality=QUALITY, subsampling_mode=SUBSAMPLING)
    
    # ---------------------------------------------------------
    # 计算三个指标
    # ---------------------------------------------------------
    # 1. Recon Error (模拟器误差): 模拟图 - 原图
    recon_err = (x - recon).abs().mean().item()
    
    # 2. Real Error (真实误差): 真实JPEG - 原图 【你需要的】
    real_err  = (x - real).abs().mean().item()
    
    # 3. Diff (逼近误差): 真实JPEG - 模拟图
    diff      = (real - recon).abs().mean().item()
    
    print("-" * 40)
    print(f"Recon Error (Model vs Orig): {recon_err:.6f}")
    print(f"Real Error  (Real  vs Orig): {real_err:.6f}") 
    print(f"Diff        (Real  vs Model):{diff:.6f}")
    print("-" * 40)
    
    # 简单的分析
    if diff < real_err * 0.5:
        print(">> 状态优秀：Diff 远小于压缩误差本身，模拟非常精准。")
    elif abs(recon_err - real_err) < 0.001:
        print(">> 状态良好：模拟出的失真程度(Error)与真实JPEG一致。")
        
    transforms.ToPILImage()(recon.squeeze(0).clamp(0, 1)).save("recon_sim.png")
    transforms.ToPILImage()(real.squeeze(0).clamp(0, 1)).save("real_jpeg.png")
    transforms.ToPILImage()((real - recon).abs().squeeze(0) * 10).save("diff_x10.png")
    print("Images saved.")