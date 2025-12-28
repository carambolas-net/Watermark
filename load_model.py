import os
import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image
from model.encoder_decoder import EncoderDecoder
from config import config_test as config
import sys
import numpy as np

class WatermarkInference:
    """水印模型推理类"""
    
    def __init__(self, checkpoint_epoch=None):
        """
        初始化推理模型
        Args:
            checkpoint_epoch: 要加载的checkpoint的epoch，默认使用config中的resume_epoch
        """
        self.device = config.device
        self.config = config
        
        # 图像预处理
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.CenterCrop((config.H, config.W)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # 加载模型
        self.model = self._load_model(checkpoint_epoch)
    
    def _load_model(self, checkpoint_epoch=None):
        """加载模型和权重"""
        epoch = checkpoint_epoch if checkpoint_epoch is not None else config.resume_epoch
        ckpt_path = f"{config.checkpoint_path}checkpoint_epoch_{epoch}.pth"
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint不存在: {ckpt_path}")
        
        # 实例化模型
        model = EncoderDecoder(config)
        model = torch.nn.DataParallel(model)
        model = model.to(self.device)
        
        # 加载权重
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model.module.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        print(f"成功加载模型 checkpoint_epoch_{epoch}.pth")
        print(f"训练损失: {ckpt.get('train_loss', 'N/A'):.4f}, 验证损失: {ckpt.get('val_loss', 'N/A'):.4f}")
        
        return model
    
    def encode(self, image_path, message=None, output_path=None):
        """
        对图片进行水印编码
        Args:
            image_path: 输入图片路径
            message: 32bit消息，如果为None则随机生成
            output_path: 输出图片路径，如果为None则自动生成
        Returns:
            encoded_image: 编码后的图片tensor
            message: 使用的消息
        """
        # 加载并处理图片
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 生成或使用消息
        if message is None:
            message = torch.randint(0, 2, (1, config.message_length), dtype=torch.float32)
        elif isinstance(message, list):
            message = torch.tensor(message, dtype=torch.float32).unsqueeze(0)
        message = message.to(self.device)
        
        # 推理（只编码，不加噪声）
        with torch.no_grad():
            encoded_image = self.model.module.encoder(img_tensor, message)
        
        # 保存结果
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"encoded_{base_name}.png"
        
        # 反归一化并保存
        img_to_save = encoded_image[0].cpu().clone()
        img_to_save = (img_to_save + 1.0) / 2.0
        torchvision.utils.save_image(img_to_save, output_path, normalize=False)
        
        print(f"编码完成，保存到: {output_path}")
        print(f"嵌入消息: {' '.join([str(int(x)) for x in message[0].cpu().numpy()])}")
        
        return encoded_image, message
    
    def decode(self, image_path):
        """
        从图片中解码水印消息
        Args:
            image_path: 包含水印的图片路径
        Returns:
            decoded_message: 解码后的消息（二值化）
            raw_message: 解码后的原始值
        """
        # 加载并处理图片
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 推理解码
        with torch.no_grad():
            decoded_message = self.model.module.decoder(img_tensor)
        
        raw_message = decoded_message[0].cpu().numpy()
        binary_message = (raw_message > 0.5).astype(int)
        
        print(f"解码消息: {' '.join([str(x) for x in binary_message])}")
        print(f"原始值: {' '.join([f'{x:.4f}' for x in raw_message])}")
        
        return binary_message, raw_message
    
    def decode_with_ber(self, image_path, original_message):
        """
        从图片中解码水印消息并计算误码率
        Args:
            image_path: 包含水印的图片路径
            original_message: 原始消息(列表或tensor)
        Returns:
            结果字典，包含解码消息、原始值和误码率
        """
        # 转换原始消息格式
        if isinstance(original_message, list):
            original_msg = np.array(original_message, dtype=int)
        elif isinstance(original_message, torch.Tensor):
            original_msg = original_message.cpu().numpy().astype(int).squeeze()
        else:
            original_msg = original_message
        
        # 加载并处理图片
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 推理解码
        with torch.no_grad():
            decoded_message = self.model.module.decoder(img_tensor)
        
        raw_message = decoded_message[0].cpu().numpy()
        binary_message = (raw_message > 0.5).astype(int)
        
        # 计算误码率
        err_bits = (original_msg != binary_message).sum()
        ber = err_bits / config.message_length
        
        print(f"原始消息: {' '.join([str(int(x)) for x in original_msg])}")
        print(f"解码消息: {' '.join([str(x) for x in binary_message])}")
        print(f"原始值: {' '.join([f'{x:.4f}' for x in raw_message])}")
        print(f"误码率(BER): {ber:.6f} ({err_bits}/{config.message_length})")
        
        return {
            'original_message': original_msg,
            'decoded_message': binary_message,
            'raw_message': raw_message,
            'ber': ber,
            'err_bits': err_bits
        }
    
    def encode_and_decode(self, image_path, message=None, output_dir=None):
        """
        完整的编码解码流程（用于测试）
        Args:
            image_path: 输入图片路径
            message: 32bit消息
            output_dir: 输出目录
        Returns:
            结果字典
        """
        if output_dir is None:
            output_dir = "inference_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载图片
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 生成消息
        if message is None:
            message = torch.randint(0, 2, (1, config.message_length), dtype=torch.float32)
        elif isinstance(message, list):
            message = torch.tensor(message, dtype=torch.float32).unsqueeze(0)
        message = message.to(self.device)
        
        # 完整前向传播
        with torch.no_grad():
            encoded_image, noisy_image, decoded_message = self.model(img_tensor, message)
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存编码后图片
        encoded_path = os.path.join(output_dir, f"{base_name}_encoded.png")
        img_to_save = encoded_image[0].cpu().clone()
        img_to_save = (img_to_save + 1.0) / 2.0
        torchvision.utils.save_image(img_to_save, encoded_path, normalize=False)
        
        # 保存加噪后图片
        noisy_path = os.path.join(output_dir, f"{base_name}_noisy.png")
        noisy_to_save = noisy_image[0].cpu().clone()
        noisy_to_save = (noisy_to_save + 1.0) / 2.0
        torchvision.utils.save_image(noisy_to_save, noisy_path, normalize=False)
        
        # 计算误码率
        msg_np = message[0].cpu().numpy()
        decoded_np = decoded_message[0].cpu().numpy()
        binary_decoded = (decoded_np > 0.5).astype(int)
        err_bits = (msg_np != binary_decoded).sum()
        ber = err_bits / config.message_length
        
        # 保存消息结果
        msg_path = os.path.join(output_dir, f"{base_name}_message.txt")
        with open(msg_path, 'w', encoding='utf-8') as f:
            f.write("原始消息:\n")
            f.write(' '.join([str(int(x)) for x in msg_np]))
            f.write("\n\n解码消息:\n")
            f.write(' '.join([str(x) for x in binary_decoded]))
            f.write(f"\n\n误码率(BER): {ber:.6f}")
            f.write(f"\n错误比特数: {err_bits}/{config.message_length}")
            f.write("\n\n原始解码值:\n")
            f.write(' '.join([f"{x:.6f}" for x in decoded_np]))
        
        print(f"\n推理结果:")
        print(f"  编码图片: {encoded_path}")
        print(f"  加噪图片: {noisy_path}")
        print(f"  消息文件: {msg_path}")
        print(f"  误码率(BER): {ber:.6f} ({err_bits}/{config.message_length})")
        
        return {
            'encoded_image': encoded_image,
            'noisy_image': noisy_image,
            'decoded_message': binary_decoded,
            'raw_message': decoded_np,
            'original_message': msg_np,
            'ber': ber
        }


def main():
    """测试推理功能"""
    # 初始化推理器
    inference = WatermarkInference(checkpoint_epoch=config.resume_epoch)
    
    # 检查是否提供了图片路径参数
    if len(sys.argv) > 1:
        # 使用指定的图片进行解码
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return
        
        # 检查是否提供了原始消息参数（以逗号分隔的二进制数字）
        if len(sys.argv) > 2:
            msg_str = sys.argv[2]
            try:
                original_message = [int(x) for x in msg_str.replace(',', ' ').split()]
                print(f"解码指定图片: {image_path}")
                print(f"原始消息: {' '.join([str(x) for x in original_message])}\n")
                result = inference.decode_with_ber(image_path, original_message)
            except ValueError:
                print(f"消息格式错误，应为逗号或空格分隔的二进制数字")
        else:
            print(f"解码指定图片: {image_path}")
            print("(未提供原始消息，无法计算误码率)\n")
            binary_msg, raw_msg = inference.decode(image_path)
        return
    
    # 使用验证集的前N张图片测试
    N = 2  # 可根据需要修改
    val_images = os.listdir(config.val_data_path)
    if val_images:
        test_images = val_images[:N]
        for idx, img_name in enumerate(test_images, 1):
            test_image = os.path.join(config.val_data_path, img_name)
            print(f"\n[{idx}/{len(test_images)}] 使用测试图片: {test_image}")
            # 完整编码解码测试
            result = inference.encode_and_decode(test_image)
        print(f"\n共测试{len(test_images)}张图片，测试完成!")
    else:
        print("验证集为空，请检查val_data_path配置")


if __name__ == "__main__":
    main()
