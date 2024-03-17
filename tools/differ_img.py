from PIL import Image
import numpy as np
image1 = Image.open('/home/ipl-pc/cmkd/output/diff_feat.png')
image1=image1.convert('L')
# image2 = Image.open('/home/ipl-pc/cmkd/output/LiDAR_img.png')
# image2=image2.convert('L')

array1 = np.array(image1)
# print(array1.max())
# array2 = np.array(image2)

# diff_array = np.abs(array1 - array2)
for i in range(array1.shape[0]):
    for j in range(array1.shape[1]):
        if array1[i][j]==0:
            array1[i][j]=255
        else:
            array1[i][j]=0
        # diff_array[i][j]=255-diff_array[i][j]
# print(array1[-1][0])
# print(array2[-1][0])
# print(diff_array[-1][0])
diff_image = Image.fromarray(array1.astype(np.uint8))

diff_image.save('/home/ipl-pc/cmkd/output/diff_feat_2.png')
