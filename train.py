import os
import torch
import time

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision
from PIL import Image
from model.encoder_decoder import EncoderDecoder
from config import config, config_test
from loss_function import LossFunction




class WatermarkDataset(Dataset):
    """水印数据集：加载图片并生成随机消息"""
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
        # 加载图片并转换
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        # 生成随机32bit消息
        msg = torch.randint(0, 2, (config.message_length,), dtype=torch.float32)
        return img, msg


def log(message, every=None, step=None):
    if every is None or step is None or (step + 1) % every == 0:
        print(message)


def compute_step(model, loss_fn, img, msg, optimizer=None, device=None):
    is_train = optimizer is not None
    if device is None:
        device = config.device
    img = img.to(device)
    msg = msg.to(device)

    with torch.set_grad_enabled(is_train):
        if is_train:
            optimizer.zero_grad()

        out_img, noisy_img, out_msg = model(img, msg)

        # 打印最大最小值用于调试
        # print(f"out_img max: {out_img.max().item():.4f}, min: {out_img.min().item():.4f}")
        img_loss = loss_fn.img_loss(out_img, img)
        msg_loss = loss_fn.msg_loss(out_msg, msg)
        total_loss = img_loss * 0.7 + msg_loss

        if is_train:
            total_loss.backward()
            optimizer.step()

    return out_img, noisy_img, out_msg, img_loss, msg_loss, total_loss


def load_checkpoint(model, optimizer):
    start_epoch = 0
    if config.resume_epoch > 0:
        ckpt_path = f"{config.checkpoint_path}checkpoint_epoch_{config.resume_epoch}.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=config.device)
            model.module.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch']
            log(f"从epoch {start_epoch} 恢复训练")
        else:
            log(f"警告: checkpoint {ckpt_path} 不存在，从头开始训练")
    return start_epoch


def train_one_epoch(epoch, model, train_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    num_epochs = config.num_epochs
    prev_batch_end = time.time()

    for batch_idx, (img, msg) in enumerate(train_loader):
        if batch_idx == config.data_time_batch_idx:
            data_time = time.time() - prev_batch_end
            log(
                f"Epoch [{epoch+1}/{num_epochs}] 第{batch_idx+1}批数据获取耗时: {data_time:.2f}s"
            )

        _, _, _, img_loss, msg_loss, loss = compute_step(model, loss_fn, img, msg, optimizer=optimizer)
        total_loss += loss.item()

        log(
            f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
            f"Loss: {loss.item():.4f} (img: {img_loss.item():.4f}, msg: {msg_loss.item():.4f})",
            every=config.log_interval_batch,
            step=batch_idx,
        )

        prev_batch_end = time.time()

    train_avg_loss = total_loss / len(train_loader)
    log(f"Epoch [{epoch+1}/{num_epochs}] 训练平均损失: {train_avg_loss:.4f}")
    return train_avg_loss


def validate_one_epoch(epoch, model, val_loader, loss_fn):
    num_epochs = config.num_epochs
    val_device = "cuda" if torch.cuda.is_available() else config.device
    if val_device != "cuda":
        log("警告: 未检测到CUDA，验证仍在CPU上进行")

    # 创建验证用模型（使用config_test的噪声配置）
    val_model = EncoderDecoder(config_test)
    val_model = torch.nn.DataParallel(val_model)
    val_model = val_model.to(val_device)
    val_model.module.load_state_dict(model.module.state_dict())
    val_model.eval()

    val_total_loss = 0.0
    last_batch = None
    last_img_loss = None
    last_msg_loss = None

    for _, (img, msg) in enumerate(val_loader):
        out_img, noisy_img, out_msg, img_loss, msg_loss, _ = compute_step(
            val_model, loss_fn, img, msg, device=val_device
        )
        val_total_loss += (img_loss + msg_loss).item()
        last_batch = (out_img, noisy_img, out_msg, msg)
        last_img_loss = img_loss
        last_msg_loss = msg_loss

    val_avg_loss = val_total_loss / len(val_loader)
    log(
        f"Epoch [{epoch+1}/{num_epochs}] 验证平均损失: {val_avg_loss:.4f} "
        f"img: {last_img_loss.item():.4f}, msg: {last_msg_loss.item():.4f}"
    )
    return val_avg_loss, last_batch


def save_eval_outputs(epoch, batch_outputs):
    out_img, noisy_img, out_msg, msg = batch_outputs
    epoch_dir = os.path.join(config.eval_data_path, f"epoch{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    save_n = min(config.save_eval_number, out_img.size(0))
    for i in range(save_n):
        img_to_save = out_img[i].cpu().clone()
        img_to_save = (img_to_save + 1.0) / 2.0  # 反归一化到[0,1]
        torchvision.utils.save_image(img_to_save, os.path.join(epoch_dir, f"img{i}.png"), normalize=False)
        noisy_img_to_save = noisy_img[i].cpu().clone()
        noisy_img_to_save = (noisy_img_to_save + 1.0) / 2.0  # 反归一化到[0,1]
        torchvision.utils.save_image(
            noisy_img_to_save, os.path.join(epoch_dir, f"noisy_img{i}.png"), normalize=False
        )
        with open(os.path.join(epoch_dir, f"msg{i}.txt"), 'w', encoding='utf-8') as f:
            f.write("原始消息:\n")
            f.write(' '.join([str(int(x)) for x in msg[i].cpu().numpy()]))
            f.write("\n解码消息:\n")
            # 美化解码消息：用0/1表示，空格分隔
            f.write(' '.join([str(int(x)) for x in (out_msg[i].cpu().numpy() > 0.5)]))
            f.write("\n误码率(BER): ")
            err_bits = (msg[i].cpu().numpy() != (out_msg[i].cpu().numpy() > 0.5)).sum()
            ber = err_bits / config.message_length
            f.write(f"{ber:.6f}\n")
            f.write("\n原始解码消息:\n")
            f.write(' '.join([f"{x:.6f}" for x in out_msg[i].cpu().numpy()]))
            f.write("\n")


def save_checkpoint(epoch, model, optimizer, train_loss, val_loss):
    if (epoch + 1) % config.save_interval_epoch != 0:
        return
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, f"{config.checkpoint_path}checkpoint_epoch_{epoch+1}.pth")


def run_training(model, train_loader, val_loader, loss_fn, optimizer):
    start_epoch = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.time()

        train_start_time = time.time()
        train_avg_loss = train_one_epoch(epoch, model, train_loader, loss_fn, optimizer)
        train_time = time.time() - train_start_time

        val_avg_loss = None
        if config.enable_validation:
            val_start_time = time.time()
            val_avg_loss, val_batch = validate_one_epoch(epoch, model, val_loader, loss_fn)
            val_time = time.time() - val_start_time
            log(f"Epoch [{epoch+1}/{config.num_epochs}] 验证耗时: {val_time:.2f}s")

            if config.enable_save_eval and val_batch is not None:
                save_start_time = time.time()
                save_eval_outputs(epoch, val_batch)
                save_time = time.time() - save_start_time
                log(f"Epoch [{epoch+1}/{config.num_epochs}] 保存验证输出耗时: {save_time:.2f}s")

        ckpt_start_time = time.time()
        save_checkpoint(
            epoch,
            model,
            optimizer,
            train_avg_loss,
            val_avg_loss,
        )
        ckpt_time = time.time() - ckpt_start_time

        epoch_time = time.time() - epoch_start_time
        log(f"Epoch [{epoch+1}/{config.num_epochs}] 训练耗时: {train_time:.2f}s")
        log(f"Epoch [{epoch+1}/{config.num_epochs}] 保存检查点耗时: {ckpt_time:.2f}s")
        log(f"Epoch [{epoch+1}/{config.num_epochs}] 总耗时: {epoch_time:.2f}s")

    log("训练完成!")


def train():
    # 创建数据集和数据加载器
    train_dataset = WatermarkDataset(config.train_data_path)
    val_dataset = WatermarkDataset(config.val_data_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    # 实例化模型，支持多卡训练
    model = EncoderDecoder(config)
    model = torch.nn.DataParallel(model)
    model = model.to(config.device)

    # 实例化损失函数和优化器
    loss_fn = LossFunction(device=config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    run_training(model, train_loader, val_loader, loss_fn, optimizer)


__name__ == "__main__" and train()
