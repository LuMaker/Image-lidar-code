import torch
from utils.data_processing import get_data_loaders
from model.autoencoder import ShapeIntensityAutoencoder
from utils.unsupervised_train_eval import (
    train_autoencoder,
    extract_features,
    find_optimal_k,
    cluster_features,
    detect_outliers,
    visualize_clusters,
    save_clustering_results
)
import os
import argparse

# 解析命令行参数
# parser = argparse.ArgumentParser(description='无监督学习图像分类')
# parser.add_argument('--use_pretrained', action='store_true', help='使用已训练的自编码器模型')
# parser.add_argument('--model_path', type=str, default='autoencoder_model.pth', help='预训练模型路径')
# parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
# parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
# parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
# parser.add_argument('--min_k', type=int, default=2, help='最小聚类数量')
# parser.add_argument('--max_k', type=int, default=10, help='最大聚类数量')
# parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128], help='图像尺寸')
# parser.add_argument('--center_crop', action='store_true', help='启用中心裁剪')
# parser.add_argument('--outlier_threshold', type=float, default=1.5, help='异常检测阈值')
# args = parser.parse_args()

# 配置参数
# data_dir = args.data_dir
# batch_size = args.batch_size
# num_epochs = args.num_epochs
# learning_rate = args.learning_rate
# min_k = args.min_k
# max_k = args.max_k
# img_size = tuple(args.img_size)
# center_crop = args.center_crop
# outlier_threshold = args.outlier_threshold
# use_pretrained = args.use_pretrained
# model_path = args.model_path

# 配置参数
data_dir = 'data0826_near'
batch_size = 16
max_epochs = 100
learning_rate = 0.001
min_k = 2
max_k = 10
img_size = (128, 128)
center_crop = True
outlier_threshold = 2.0
use_pretrained = True
model_path = 'autoencoder_model.pth'

# 检查数据目录是否存在
if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
    print(f"数据目录 {data_dir} 不存在或为空。请将图像放入该目录。")
    exit(1)

# 加载数据（无监督模式）
data_loader, _, _ = get_data_loaders(
    data_dir, 
    batch_size=batch_size, 
    supervised=False, 
    img_size=img_size, 
    center_crop=center_crop
)
print(f"数据加载完成，共 {len(data_loader.dataset)} 张图像")
print(f"图像尺寸: {img_size}, 中心裁剪: {center_crop}")

# 初始化或加载自编码器模型
if use_pretrained:
    if os.path.exists(model_path):
        print(f"加载预训练模型: {model_path}")
        model = ShapeIntensityAutoencoder(input_size=img_size)
        model.load_state_dict(torch.load(model_path))
        print("预训练模型加载完成")
    else:
        print(f"错误: 预训练模型文件 {model_path} 不存在")
        print("将训练新的自编码器模型")
        use_pretrained = False
        model = ShapeIntensityAutoencoder(input_size=img_size)
        print("自编码器模型初始化完成")
else:
    model = ShapeIntensityAutoencoder(input_size=img_size)
    print("自编码器模型初始化完成")

# 训练自编码器（如果不使用预训练模型）
if not use_pretrained:
    print("开始训练自编码器...")
    trained_model = train_autoencoder(model, data_loader, max_epochs=max_epochs, learning_rate=learning_rate)
    print("自编码器训练完成")
    
    # 保存模型
    torch.save(trained_model.state_dict(), model_path)
    print(f"自编码器模型已保存至 {model_path}")
else:
    trained_model = model

# 提取特征
print("开始提取图像特征...")
features, image_paths = extract_features(trained_model, data_loader)
print(f"特征提取完成，特征维度: {features.shape}")

# 自动确定最佳聚类数量
print("自动确定最佳聚类数量...")
optimal_k, _ = find_optimal_k(features, min_k=min_k, max_k=max_k)
print(f"最佳聚类数量确定为: {optimal_k}")

optimal_k = 2
# 聚类特征
print("开始聚类特征...")
clusters, kmeans = cluster_features(features, n_clusters=optimal_k)
print("特征聚类完成")

# 检测异常数据
print("检测异常数据...")
outliers, inliers = detect_outliers(features, clusters, kmeans, threshold=outlier_threshold)
print("异常检测完成")

# 可视化聚类结果（包含异常标记）
print("可视化聚类结果...")
visualize_clusters(features, clusters, outliers)
print("聚类结果可视化完成")

# 保存聚类结果（包含异常样本）
print("保存聚类结果...")
save_clustering_results(image_paths, clusters, outliers=outliers)
print("聚类结果保存完成")

print("\n无监督学习流程完成！")
print("聚类结果已保存到 'clustering_results' 目录中")
print("每个簇的图像已分别存放在对应的子目录中")
if len(outliers) > 0:
    print(f"异常样本已保存到 'clustering_results/outliers' 目录中")
