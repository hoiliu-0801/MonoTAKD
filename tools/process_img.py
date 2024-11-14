# from PIL import Image
# import numpy as np
# image1 = Image.open('/home/ipl-pc/cmkd/output/camera_img.png')
# image1=image1.convert('L')
# image2 = Image.open('/home/ipl-pc/cmkd/output/LiDAR_img.png')
# image2=image2.convert('L')
import cv2
import numpy as np

img = cv2.imread('/home/ipl-pc/cmkd/output/vis_result/000113_bev_img.png')
ret, mask = cv2.threshold(img[:, :,2], 120, 255, cv2.THRESH_BINARY)

mask3 = np.zeros_like(img)
mask3[:, :, 0] = mask
mask3[:, :, 1] = mask
mask3[:, :, 2] = mask

# extracting `orange` region using `bitewise_and`
orange = cv2.bitwise_and(img, mask3)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# extracting non-orange region
gray = cv2.bitwise_and(img, 255 - mask3)

# orange masked output
out = gray + orange

cv2.imwrite('/home/ipl-pc/cmkd/output/diff_feat_113.png', orange)
# cv2.imwrite('gray.png', gray)
# cv2.imwrite("output.png", out)

# diff_image.save('/home/ipl-pc/cmkd/output/diff_feat.png')
