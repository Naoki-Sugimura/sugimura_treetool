import pclpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import open3d as o3d
import treetool.seg_tree as seg_tree
import treetool.utils as utils
import treetool.tree_tool as tree_tool

# 点群データの読み込み
def load_point_cloud(file_path):
    PointCloud = pclpy.pcl.PointCloud.PointXYZ()
    pclpy.pcl.io.loadPCDFile(file_path, PointCloud)
    return PointCloud

# 点群をボクセル化する関数
def voxelize_point_cloud(PointCloud, voxel_size):
    voxelized = seg_tree.voxelize(PointCloud.xyz, voxel_size)
    
    # voxelizedがnumpy配列として返されることを期待する
    print(f"Voxelized shape: {voxelized.shape}")
    return voxelized

# pclpyのPointCloudをNumPy配列に変換する関数
def pcl_to_numpy(pcl_point_cloud):
    # 点群データがすでにNumPy配列であればそのまま返す
    if isinstance(pcl_point_cloud, np.ndarray):
        return pcl_point_cloud
    # pclpyのPointCloudからNumPy配列を取得
    np_points = np.array(pcl_point_cloud.xyz)
    
    if np_points.shape[1] != 3:
        raise ValueError(f"Expected point cloud data with shape (N, 3), but got shape {np_points.shape}")
    
    return np_points

# pclpyのPointCloudをOpen3DのPointCloudに変換する関数
def pcl_to_o3d(np_points):
    # Open3DのPointCloudに変換
    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(np_points)  # Open3DのPointCloudに変換
    return o3d_points

# 点群可視化の関数（Open3D）
def visualize_point_cloud_o3d(PointCloud_list, voxel_size):
    # PointCloud_listがリストであることを想定
    o3d_point_clouds = []
    
    for PointCloud in PointCloud_list:
        np_points = pcl_to_numpy(PointCloud)  # pclpyのPointCloudをNumPy配列に変換
        
        # np_pointsの形状を確認して、デバッグ
        print(f"PointCloud shape before visualization: {np_points.shape}")
        
        # Open3Dに変換
        o3d_point_cloud = pcl_to_o3d(np_points)
        o3d_point_clouds.append(o3d_point_cloud)
    
    # Open3Dで可視化
    o3d.visualization.draw_geometries(o3d_point_clouds)

# 樹木検出処理を行う関数
def process_trees(PointCloudV):
    My_treetool = tree_tool.treetool(PointCloudV)
    
    # ステップ1: 地面除去
    My_treetool.step_1_remove_floor()
    
    # デバッグ: non_ground_cloud と ground_cloud の確認
    print(f"non_ground_cloud: {getattr(My_treetool, 'non_ground_cloud', 'Not Set')}")
    print(f"ground_cloud: {getattr(My_treetool, 'ground_cloud', 'Not Set')}")
    
    if hasattr(My_treetool, 'non_ground_cloud') and hasattr(My_treetool, 'ground_cloud'):
        # 両方がセットされていれば可視化
        visualize_point_cloud_o3d([My_treetool.non_ground_cloud, My_treetool.ground_cloud], voxel_size=0.3)
    else:
        print("non_ground_cloud or ground_cloud is not set correctly.")
    
    # ステップ2: 法線フィルタリング
    My_treetool.step_2_normal_filtering(verticality_threshold=0.06, curvature_threshold=0.1)
    visualize_point_cloud_o3d([My_treetool.filtered_points.xyz, 
                               My_treetool.filtered_points.xyz + My_treetool.filtered_normals * 0.05], voxel_size=0.3)
    
    # ステップ3: ユークリッドクラスタリング
    My_treetool.step_3_euclidean_clustering(tolerance=0.1, min_cluster_size=40, max_cluster_size=6000000)
    visualize_point_cloud_o3d(My_treetool.cluster_list, voxel_size=0.3)
    
    # ステップ4: 幹のグループ化
    My_treetool.step_4_group_stems(max_distance=0.4)
    visualize_point_cloud_o3d(My_treetool.complete_Stems, voxel_size=0.2)
    
    # ステップ5: 樹木の高さ調整
    My_treetool.step_5_get_ground_level_trees(lowstems_height=5, cutstems_height=5)
    visualize_point_cloud_o3d(My_treetool.low_stems, voxel_size=0.2)
    
    # ステップ6: 円筒モデルによる樹木モデリング
    My_treetool.step_6_get_cylinder_tree_models(search_radius=0.1)
    visualize_point_cloud_o3d([i['tree'] for i in My_treetool.finalstems] + My_treetool.visualization_cylinders, voxel_size=0.2)
    
    # ステップ7: 楕円フィット
    My_treetool.step_7_ellipse_fit()
    
    # 結果をCSVに保存
    My_treetool.save_results(save_location='results/myresults.csv')
    print("Results saved to 'results/myresults.csv'")
    
    return My_treetool

def main():
    # 点群データの読み込み
    file_directory = 'data/downsampledlesscloudEURO2.pcd'  # このパスを実際のPCDファイルのパスに変更してください
    PointCloud = load_point_cloud(file_directory)
    
    # 点群データのボクセル化
    PointCloudV = voxelize_point_cloud(PointCloud, voxel_size=0.04)
    print(f"PointCloudV shape after voxelization: {PointCloudV.shape}")
    
    # 点群データをOpen3Dで可視化
    visualize_point_cloud_o3d([PointCloudV], voxel_size=0.3)
    
    # 樹木検出処理
    My_treetool = process_trees(PointCloudV)
    
    # 結果の確認
    print("Detected tree data and DBH saved to CSV.")
    
if __name__ == "__main__":
    main()
