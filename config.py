import torch

class Config:
    # data paths
    train_data_path = "data_set/train/"
    #train_data_path = "/root/watermark/img/train/class1"
    val_data_path = "data_set/val/"
    #val_data_path = "/root/watermark/img/val/class1"
    eval_data_path = "data_set/eval_val/"
    checkpoint_path = "checkpoints/"
    
    # model
    conv_channels = 64
    encoder_block = 4
    decoder_block = 7
    message_length = 32
    
    H = 128
    W = 128


    # training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 16
    batch_size = 2
    learning_rate = 1e-5
    num_epochs = 400
    log_interval_batch = 3
    save_eval_number = 2
    save_interval_epoch = 3
    ##137效果不错
    #0-115 0.2ssim0.8 0.7img_loss 1e-3
    #116-144 0.2ssim0.8 0.7img_loss 1e-4
    #145-263 0.3ssim0.7 0.5img_loss 1e-4
    #264-345 0.4ssim0.5 0.5img_loss 1e-4
    #346- 0.4ssim0.5 0.5img_loss 1e-5
    resume_epoch = 10
    
    # noisy
    DCT_Y=25
    DCT_UV=10
    gaussian_std = 0.03  # 高斯噪声标准差
    dropout_prob = 0.3  # dropout概率
    crop_ratio_min = 0.7  # 裁剪最小比例
    crop_ratio_max = 1.0  # 裁剪最大比例
    cropout_ratio_min = 0.1  # 遮挡最小比例
    cropout_ratio_max = 0.3  # 遮挡最大比例
    
    # 混合噪声配置（按顺序应用）
    # 示例: ["gaussian", "jpeg", "cropout"] 会依次应用这三种噪声
    # 设置为 None 或空列表则使用单一随机噪声
    noise_sequence = ["crop","gaussian","jpeg"]  # 或者例如: ["gaussian", "jpeg"]
    
config = Config()

class Config_test:
    # data paths
    train_data_path = "/root/watermark/img/train/class1"
    val_data_path = "data_set/val/"
    #val_data_path = "/root/watermark/img/val/class1"
    eval_data_path = "data_set/eval_val/"
    checkpoint_path = "checkpoints/"
    
    # model
    conv_channels = 64
    encoder_block = 4
    decoder_block = 7
    message_length = 32
    
    H = 128
    W = 128


    # training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 16
    batch_size = 128
    learning_rate = 1e-5
    num_epochs = 400
    log_interval_batch = 3
    save_eval_number = 2
    ##137效果不错
    #0-115 0.2ssim0.8 0.7img_loss 1e-3
    #116-144 0.2ssim0.8 0.7img_loss 1e-4
    #145-263 0.3ssim0.7 0.5img_loss 1e-4
    #264-345 0.4ssim0.5 0.5img_loss 1e-4
    #346- 0.4ssim0.5 0.5img_loss 1e-5
    resume_epoch = 0
    
    # noisy
    DCT_Y=25
    DCT_UV=10
    jpeg_quality=75
    gaussian_std = 0.03  # 高斯噪声标准差
    dropout_prob = 0.3  # dropout概率
    crop_ratio_min = 0.7  # 裁剪最小比例
    crop_ratio_max = 1.0  # 裁剪最大比例
    cropout_ratio_min = 0.1  # 遮挡最小比例
    cropout_ratio_max = 0.3  # 遮挡最大比例
    
    # 混合噪声配置（按顺序应用）
    # 示例: ["gaussian", "jpeg", "cropout"] 会依次应用这三种噪声
    # 设置为 None 或空列表则使用单一随机噪声
    noise_sequence = ["crop","gaussian","jpeg_real"]  # 或者例如: ["gaussian", "jpeg"]
    
config_test = Config_test()