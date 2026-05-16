import os
import sys
import signal
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision
from PIL import Image
from model.encoder_decoder import EncoderDecoder
from model.discriminator import Discriminator
from config import config, config_test
from loss_function import LossFunction


# 全局日志文件句柄
_log_file = None
# 中断标志（由信号处理器设置）
_interrupted = False


def _signal_handler(signum, frame):
    """捕获 SIGINT/SIGTERM，设置中断标志"""
    global _interrupted
    _interrupted = True
    # 首次 Ctrl+C 只设标志，第二次直接退出
    if signum == signal.SIGINT:
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(1))


def set_log_file(log_dir="logs"):
    """初始化日志文件，返回日志文件路径"""
    global _log_file
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_{timestamp}.log")
    _log_file = open(log_path, 'w', encoding='utf-8')
    return log_path


def close_log_file():
    """关闭日志文件"""
    global _log_file
    if _log_file is not None:
        _log_file.flush()
        _log_file.close()
        _log_file = None


def is_interrupted():
    """检查是否收到中断信号"""
    global _interrupted
    return _interrupted




class WatermarkDataset(Dataset):
    """水印数据集：加载图片并生成随机消息"""
    def __init__(self, data_path):
        self.image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(config.H, config.W), antialias=True),
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
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    if every is None or step is None or (step + 1) % every == 0:
        print(message)
        sys.stdout.flush()
        global _log_file
        if _log_file is not None:
            _log_file.write(message + '\n')
            _log_file.flush()


def compute_psnr(img1, img2):
    """计算两张图片之间的 PSNR (峰值信噪比)
    img1, img2: 归一化到 [-1, 1] 的图片张量, shape (B, C, H, W) 或 (C, H, W)
    返回: 平均 PSNR (dB)
    """
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])  # 每个样本的 MSE
    # 像素值范围为 [-1, 1], MAX = 2.0
    psnr = 10 * torch.log10(4.0 / (mse + 1e-10))
    return psnr.mean().item()


def compute_step(encoder_decoder, discriminator, loss_fn, img, msg,
                 optimizer_enc_dec=None, optimizer_discrim=None, device=None):
    """单步训练/推理：判别器 + 编码器-解码器 联合优化"""
    is_train = optimizer_enc_dec is not None and optimizer_discrim is not None
    if device is None:
        device = config.device
    img = img.to(device)
    msg = msg.to(device)
    batch_size = img.shape[0]

    with torch.set_grad_enabled(is_train):
        if is_train:
            optimizer_discrim.zero_grad()

        # ---------- 训练判别器: on cover (真实图) ----------
        d_on_cover = discriminator(img)
        d_target_cover = torch.full((batch_size, 1), config.cover_label, device=device, dtype=torch.float32)
        d_loss_cover = loss_fn.adv_loss(d_on_cover, d_target_cover)
        if is_train:
            d_loss_cover.backward()

        # ---------- 编码器-解码器前向传播 ----------
        out_img, noisy_img, out_msg = encoder_decoder(img, msg)

        # ---------- 训练判别器: on encoded (编码图, detach) ----------
        d_on_encoded = discriminator(out_img.detach())
        d_target_encoded = torch.full((batch_size, 1), config.encoded_label, device=device, dtype=torch.float32)
        d_loss_encoded = loss_fn.adv_loss(d_on_encoded, d_target_encoded)
        if is_train:
            d_loss_encoded.backward()
            # # === 抢修策略4: 限制判别器更新频率 ===
            # # 如果判别器的总损失已经很低（例如小于0.4），说明它已经能轻易识破生成器，跳过本次参数更新
            # total_d_loss = d_loss_cover.item() + d_loss_encoded.item()
            # if total_d_loss > 0.2:
            #     # === 抢修策略2: 梯度裁剪 (Discriminator) ===
            #     # 防止梯度爆炸导致决策边界被严重破坏
            #     torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            #     optimizer_discrim.step()
            # else:
            #     # 清空已计算的梯度，放弃更新，让生成器追赶
            #     optimizer_discrim.zero_grad()
            optimizer_discrim.step()

        # ---------- 训练生成器 (encoder-decoder) ----------
        if is_train:
            optimizer_enc_dec.zero_grad()

        # 对抗损失：希望判别器把编码图判为 cover（真实）
        d_on_encoded_for_gen = discriminator(out_img)
        g_target_cover = torch.full((batch_size, 1), config.cover_label, device=device, dtype=torch.float32)
        g_loss_adv = loss_fn.adv_loss(d_on_encoded_for_gen, g_target_cover)

        # 图片损失 (VGG perceptual)，消息损失
        g_loss_enc = loss_fn.img_loss(out_img, img)
        g_loss_dec = loss_fn.msg_loss(out_msg, msg)

        # 生成器总损失
        g_loss = (config.adversarial_loss_weight * g_loss_adv
                  + config.encoder_loss_weight * g_loss_enc
                  + config.decoder_loss_weight * g_loss_dec)

        if is_train:
            g_loss.backward()
            #torch.nn.utils.clip_grad_norm_(encoder_decoder.parameters(), max_norm=1.0)
            optimizer_enc_dec.step()

    # 计算编码图与原始图的 PSNR
    psnr = compute_psnr(out_img, img)

    return (out_img, noisy_img, out_msg,
        g_loss_enc, g_loss_dec, g_loss_adv,
        d_loss_cover, d_loss_encoded, g_loss, psnr,
        d_on_cover.detach(), d_on_encoded.detach())


def load_checkpoint(model, discriminator, optimizer_enc_dec, optimizer_discrim):
    start_epoch = 0
    if config.resume_epoch > 0:
        ckpt_path = f"{config.checkpoint_path}checkpoint_epoch_{config.resume_epoch}.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=config.device)
            # DDP 模型通过 .module 访问底层模型
            underlying_model = model.module if hasattr(model, 'module') else model
            underlying_disc = discriminator.module if hasattr(discriminator, 'module') else discriminator
            underlying_model.load_state_dict(ckpt['model_state_dict'])
            underlying_disc.load_state_dict(ckpt['discriminator_state_dict'])
            optimizer_enc_dec.load_state_dict(ckpt['optimizer_enc_dec_state_dict'])
            optimizer_discrim.load_state_dict(ckpt['optimizer_discrim_state_dict'])
            start_epoch = ckpt['epoch']
            log(f"从epoch {start_epoch} 恢复训练")
        else:
            log(f"警告: checkpoint {ckpt_path} 不存在，从头开始训练")
    return start_epoch


def train_one_epoch(epoch, model, discriminator, train_loader, loss_fn,
                    optimizer_enc_dec, optimizer_discrim, train_sampler=None):
    model.train()
    discriminator.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    total_loss = 0.0
    num_epochs = config.num_epochs
    prev_batch_end = time.time()

    for batch_idx, (img, msg) in enumerate(train_loader):
        if batch_idx == config.data_time_batch_idx:
            data_time = time.time() - prev_batch_end
            log(
                f"Epoch [{epoch+1}/{num_epochs}] 第{batch_idx+1}批数据获取耗时: {data_time:.2f}s"
            )

        _, _, _, g_loss_enc, g_loss_dec, g_loss_adv, d_loss_cover, d_loss_encoded, g_loss, psnr, _, _ = \
            compute_step(model, discriminator, loss_fn, img, msg,
                         optimizer_enc_dec=optimizer_enc_dec,
                         optimizer_discrim=optimizer_discrim)
        total_loss += g_loss.item()

        log(
            f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
            f"G: {g_loss.item():.4f} (enc: {g_loss_enc.item():.4f}, dec: {g_loss_dec.item():.4f}, "
            f"adv: {g_loss_adv.item():.4f}) "
            f"D: (cover: {d_loss_cover.item():.4f}, enc: {d_loss_encoded.item():.4f}) "
            f"PSNR: {psnr:.2f}dB",
            every=config.log_interval_batch,
            step=batch_idx,
        )

        prev_batch_end = time.time()

    train_avg_loss = total_loss / len(train_loader)
    log(f"Epoch [{epoch+1}/{num_epochs}] 训练平均损失: {train_avg_loss:.4f}")
    return train_avg_loss


def validate_one_epoch(epoch, model, discriminator, val_loader, loss_fn):
    num_epochs = config.num_epochs
    is_distributed = dist.is_initialized()
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if is_distributed else 0
    val_device = torch.device(f"cuda:{local_rank}") if is_distributed else config.device

    # 创建验证用模型（使用config_test的噪声配置），不需要 DDP 包装
    val_model = EncoderDecoder(config_test)
    val_model = val_model.to(val_device)
    # DDP 模型通过 .module 访问底层模型
    state_dict = model.module.state_dict() if is_distributed else model.state_dict()
    val_model.load_state_dict(state_dict)
    val_model.eval()

    # 验证用判别器（eval 模式）
    val_discriminator = Discriminator(config_test)
    val_discriminator = val_discriminator.to(val_device)
    disc_state_dict = discriminator.module.state_dict() if is_distributed else discriminator.state_dict()
    val_discriminator.load_state_dict(disc_state_dict)
    val_discriminator.eval()

    val_total_loss = 0.0
    last_batch = None
    last_g_loss_enc = None
    last_g_loss_dec = None
    last_psnr = None

    all_d_on_cover = []
    all_d_on_encoded = []

    for _, (img, msg) in enumerate(val_loader):
        img = img.to(val_device)
        msg = msg.to(val_device)
        
        # 【修改这里】：接收 d_cover 和 d_encoded
        out_img, noisy_img, out_msg, g_loss_enc, g_loss_dec, _, _, _, g_loss, psnr, d_cover, d_encoded = compute_step(
            val_model, val_discriminator, loss_fn, img, msg, device=val_device
        )
        val_total_loss += g_loss.item()
        last_batch = (out_img, noisy_img, out_msg, msg, img, psnr)
        last_g_loss_enc = g_loss_enc
        last_g_loss_dec = g_loss_dec
        last_psnr = psnr
        
        # 【新增】：展平并加入列表
        all_d_on_cover.append(d_cover.view(-1))
        all_d_on_encoded.append(d_encoded.view(-1))

    # 【新增】：拼接所有的预测值并计算中位数
    all_d_on_cover = torch.cat(all_d_on_cover)
    all_d_on_encoded = torch.cat(all_d_on_encoded)
    
    median_d_cover = torch.median(all_d_on_cover).item()
    median_d_encoded = torch.median(all_d_on_encoded).item()

    val_avg_loss = val_total_loss / len(val_loader)
    
    # 【修改这里】：在日志中增加输出中位数信息
    log(
        f"Epoch [{epoch+1}/{num_epochs}] 验证平均损失: {val_avg_loss:.4f} "
        f"enc: {last_g_loss_enc.item():.4f}, dec: {last_g_loss_dec.item():.4f} "
        f"PSNR: {last_psnr:.2f}dB | "
        f"D_Median(Cover): {median_d_cover:.4f}, D_Median(Encoded): {median_d_encoded:.4f}"
    )
    return val_avg_loss, last_batch


def save_eval_outputs(epoch, batch_outputs):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    out_img, noisy_img, out_msg, msg, orig_img, batch_psnr = batch_outputs
    epoch_dir = os.path.join(config.eval_data_path, f"epoch{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    save_n = min(config.save_eval_number, out_img.size(0))
    for i in range(save_n):
        # 保存编码后的图片
        img_to_save = out_img[i].cpu().clone()
        img_to_save = (img_to_save + 1.0) / 2.0  # 反归一化到[0,1]
        torchvision.utils.save_image(img_to_save, os.path.join(epoch_dir, f"img{i}.png"), normalize=False)
        # 保存噪声图片
        noisy_img_to_save = noisy_img[i].cpu().clone()
        noisy_img_to_save = (noisy_img_to_save + 1.0) / 2.0  # 反归一化到[0,1]
        torchvision.utils.save_image(
            noisy_img_to_save, os.path.join(epoch_dir, f"noisy_img{i}.png"), normalize=False
        )
        # 保存原始裁切图片
        orig_img_to_save = orig_img[i].cpu().clone()
        orig_img_to_save = (orig_img_to_save + 1.0) / 2.0  # 反归一化到[0,1]
        torchvision.utils.save_image(
            orig_img_to_save, os.path.join(epoch_dir, f"orig_img{i}.png"), normalize=False
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
            # 计算单张图片的 PSNR
            mse = torch.mean((out_img[i] - orig_img[i]) ** 2).item()
            single_psnr = 10 * math.log10(4.0 / (mse + 1e-10))
            f.write(f"\nPSNR (编码图 vs 原图): {single_psnr:.2f} dB\n")
            f.write(f"批次平均 PSNR: {batch_psnr:.2f} dB\n")


def save_checkpoint(epoch, model, discriminator, optimizer_enc_dec, optimizer_discrim, train_loss, val_loss):
    if (epoch + 1) % config.save_interval_epoch != 0:
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': (model.module if hasattr(model, 'module') else model).state_dict(),
        'discriminator_state_dict': (discriminator.module if hasattr(discriminator, 'module') else discriminator).state_dict(),
        'optimizer_enc_dec_state_dict': optimizer_enc_dec.state_dict(),
        'optimizer_discrim_state_dict': optimizer_discrim.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, f"{config.checkpoint_path}checkpoint_epoch_{epoch+1}.pth")


def run_training(model, discriminator, train_loader, val_loader, loss_fn,
                 optimizer_enc_dec, optimizer_discrim, train_sampler=None):
    start_epoch = load_checkpoint(model, discriminator, optimizer_enc_dec, optimizer_discrim)

    for epoch in range(start_epoch, config.num_epochs):
        if is_interrupted():
            log(f"检测到中断信号，在 epoch {epoch} 前停止训练")
            break

        epoch_start_time = time.time()

        train_start_time = time.time()
        train_avg_loss = train_one_epoch(epoch, model, discriminator, train_loader, loss_fn,
                                         optimizer_enc_dec, optimizer_discrim, train_sampler)
        train_time = time.time() - train_start_time

        # 训练后检查中断
        if is_interrupted():
            log(f"Epoch [{epoch+1}/{config.num_epochs}] 训练被中断，保存紧急检查点...")
            save_checkpoint(
                epoch, model, discriminator,
                optimizer_enc_dec, optimizer_discrim,
                train_avg_loss, None,
            )
            log(f"紧急检查点已保存，正在退出...")
            break

        val_avg_loss = None
        if config.enable_validation:
            val_start_time = time.time()
            val_avg_loss, val_batch = validate_one_epoch(epoch, model, discriminator, val_loader, loss_fn)
            val_time = time.time() - val_start_time
            log(f"Epoch [{epoch+1}/{config.num_epochs}] 验证耗时: {val_time:.2f}s")

            if config.enable_save_eval and val_batch is not None:
                save_start_time = time.time()
                save_eval_outputs(epoch, val_batch)
                save_time = time.time() - save_start_time
                log(f"Epoch [{epoch+1}/{config.num_epochs}] 保存验证输出耗时: {save_time:.2f}s")

        ckpt_start_time = time.time()
        save_checkpoint(
            epoch, model, discriminator,
            optimizer_enc_dec, optimizer_discrim,
            train_avg_loss, val_avg_loss,
        )
        ckpt_time = time.time() - ckpt_start_time

        epoch_time = time.time() - epoch_start_time
        log(f"Epoch [{epoch+1}/{config.num_epochs}] 训练耗时: {train_time:.2f}s")
        log(f"Epoch [{epoch+1}/{config.num_epochs}] 保存检查点耗时: {ckpt_time:.2f}s")
        log(f"Epoch [{epoch+1}/{config.num_epochs}] 总耗时: {epoch_time:.2f}s")

    if not is_interrupted():
        log("训练完成!")
    else:
        log("训练因用户中断而提前结束")


def init_ddp():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        log("未检测到分布式环境变量，使用单卡训练")
        return -1, 1

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train():
    # 注册信号处理器
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        # DDP 初始化
        rank, world_size = init_ddp()
        is_distributed = dist.is_initialized()

        # 初始化日志文件
        log_path = set_log_file("logs")

        if is_distributed:
            log(f"分布式训练: rank={rank}, world_size={world_size}, "
                f"local_rank={os.environ.get('LOCAL_RANK', 0)}")
        log(f"日志文件: {log_path}")

        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}") if is_distributed else config.device

        # 创建数据集
        train_dataset = WatermarkDataset(config.train_data_path)
        val_dataset = WatermarkDataset(config.val_data_path)

        # 分布式采样器
        train_sampler = DistributedSampler(train_dataset) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
        )

        # 实例化编码器-解码器模型
        model = EncoderDecoder(config)
        model = model.to(device)
        if is_distributed:
            model = DDP(model, device_ids=[device], output_device=device)

        # 实例化判别器模型
        discriminator = Discriminator(config)
        discriminator = discriminator.to(device)
        if is_distributed:
            discriminator = DDP(discriminator, device_ids=[device], output_device=device)

        # 实例化损失函数和两个优化器
        loss_fn = LossFunction(device=device)
        optimizer_enc_dec = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        optimizer_discrim = torch.optim.Adam(discriminator.parameters(), lr=config.discriminator_learning_rate)

        run_training(model, discriminator, train_loader, val_loader, loss_fn,
                     optimizer_enc_dec, optimizer_discrim, train_sampler)

    except KeyboardInterrupt:
        log("捕获 KeyboardInterrupt，正在清理资源...")
    except Exception as e:
        log(f"训练异常退出: {e}")
        import traceback
        traceback.print_exc()
        # 也要写入日志文件
        global _log_file
        if _log_file is not None:
            traceback.print_exc(file=_log_file)
            _log_file.flush()
        raise
    finally:
        cleanup_ddp()
        close_log_file()
        log("日志文件已关闭")  # 这行只输出到终端（文件已关）


if __name__ == "__main__":
    train()
