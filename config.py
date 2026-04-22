
import torch
import os

class Config:
    def __init__(self, mode="train"):
        is_train = mode == "train"

        train_paths = ("../data_set/train/", "../data_set/val/")
        test_paths = ("data_set/train/", "data_set/val/")
        linux_paths = ("/root/watermark/img/train/class1", "/root/watermark/img/val/class1")

        if os.name == "nt":  # Windows
            self.train_data_path, self.val_data_path = train_paths if is_train else test_paths
        else:  # Linux/Unix
            self.train_data_path, self.val_data_path = linux_paths

        self.eval_data_path = "data_set/eval_val/"
        self.checkpoint_path = "checkpoints/"

        # model
        self.conv_channels = 64
        self.encoder_block = 4
        self.decoder_block = 7
        self.message_length = 32
        self.H = 100
        self.W = 100

        # training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 8 if is_train else 16
        self.batch_size = 24 if is_train else 100
        self.learning_rate = 1e-3
        self.num_epochs = 400
        self.log_interval_batch = 3
        self.save_eval_number = 4 if is_train else 2
        self.save_interval_epoch = 3
        self.resume_epoch = 0
        self.enable_validation = True
        self.enable_save_eval = True
        self.data_time_batch_idx = 0

        # noisy
        self.jpeg_quality = 60
        self.gaussian_std = 0.03  # 高斯噪声标准差
        self.dropout_prob = 0.3  # dropout概率
        self.crop_ratio_min = 0.7  # 裁剪最小比例
        self.crop_ratio_max = 1.0  # 裁剪最大比例
        self.cropout_ratio_min = 0.1  # 遮挡最小比例
        self.cropout_ratio_max = 0.3  # 遮挡最大比例

        # 混合噪声配置（按顺序应用）
        # 示例: ["gaussian", "jpeg", "cropout"] 会依次应用这三种噪声
        # 设置为 None 或空列表则使用单一随机噪声
        self.noise_sequence = ["I"] if is_train else ["I"]


config = Config("train")
config_test = Config("test")