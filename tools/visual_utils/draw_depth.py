import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

depth_path = '/home/ipl-pc/VirConv/data/kitti/training/depth_2/000316.png'
image = cv2.imread(depth_path)
print(image/255.0)
save_path="/home/ipl-pc/cmkd/output/depth1.png"
plt.imsave(save_path, image, cmap="viridis")
        # import matplotlib.pyplot as plt
        # print(self.forward_ret_dict["depth_maps"][0,:].shape)
        # plt.imsave(save_path, self.forward_ret_dict["depth_maps"][0,:].cpu().detach())