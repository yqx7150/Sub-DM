import scipy.io
# 加载.mat文件
data= scipy.io.loadmat('./input_data/contract_data_8h/brain_8ch_ori.mat')
#访问数据
print(data.keys())
# 
# import os
# import random
# import numpy as np
# import scipy.io as sio
# 
# def multiply_random_mask_k_space(input_array):
#     folder_path = "/home/who/桌面/zkl-gy/score_sde_pytorch-SIAT_MRIRec_noise1_multichannel6_SAKE(复件)/input_data4000/mask"  
# 
#     # 获取所有子文件夹路径
#     subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
# 
#     if len(subfolders) == 0:
#         raise ValueError("文件夹中没有子文件夹")
# 
#     # 随机选择一个子文件夹
#     random_subfolder = random.choice(subfolders)
# 
#     # 子文件夹路径
#     subfolder_path = os.path.join(folder_path, random_subfolder)
# 
#     # 获取子文件夹中所有的mask.mat文件路径
#     mask_files = [file for file in os.listdir(subfolder_path) if file.endswith(".mat")]
# 
#     if len(mask_files) == 0:
#         raise ValueError("子文件夹中没有mask.mat文件")
# 
#     random_mask_file = random.choice(mask_files)
# 
#     # 读取选中的mask.mat文件
#     mask_data = sio.loadmat(os.path.join(subfolder_path, random_mask_file))
# 
#     mask = None
#     for key in mask_data.keys():
#         if isinstance(mask_data[key], np.ndarray) and mask_data[key].shape == input_array.shape:
#             mask = mask_data[key]
#         break
# 
#     if mask is None:
#         raise ValueError("找不到合适的mask")
# 
#     # 将输入数据和mask相乘
#     result = np.multiply(input_array, mask)
# 
#     return result
# 
# 
# x_idwt_sos = np.sqrt(np.sum(np.square(np.abs(x_idwt)), axis=0))
# x_idwt_sos = x_idwt_sos / np.max(np.abs(x_idwt_sos))
# plt.imshow(abs(x_idwt_sos), cmap='gray')
# plt.show()
# plt.savefig('1.png', dpi=300)
# sys.exit()
# 
# xLL_real = xLL_real.cpu().numpy()
# xLL_imag = xLL_imag.cpu().numpy()
# xLH_real = xHL_real.cpu().numpy()
# xLH_imag = xHL_imag.cpu().numpy()
# xHL_real = xHL_real.cpu().numpy()
# xHL_imag = xHL_imag.cpu().numpy()
# xHH_real = xHH_real.cpu().numpy()
# xHH_imag = xHH_imag.cpu().numpy()
# io.savemat(osp.join('./result/dwt', 'siat_complex.mat'), {'siat_complex': siat_complex})
# io.savemat(osp.join('./result/dwt', 'siat_kdata.mat'), {'siat_kdata': siat_kdata})
# io.savemat(osp.join('./result/dwt', 'xLL_real.mat'), {'xLL_real': xLL_real})
# io.savemat(osp.join('./result/dwt', 'xLL_imag.mat'), {'xLL_imag': xLL_imag})
# io.savemat(osp.join('./result/dwt', 'xLH_real.mat'), {'xLH_real': xLH_real})
# io.savemat(osp.join('./result/dwt', 'xLH_imag.mat'), {'xLH_imag': xLH_imag})
# io.savemat(osp.join('./result/dwt', 'xHL_real.mat'), {'xHL_real': xHL_real})
# io.savemat(osp.join('./result/dwt', 'xHL_imag.mat'), {'xHL_imag': xHL_imag})
# io.savemat(osp.join('./result/dwt', 'xHH_real.mat'), {'xHH_real': xHH_real})
# io.savemat(osp.join('./result/dwt', 'xHH_imag.mat'), {'xHH_imag': xHH_imag})
# sys.exit()

