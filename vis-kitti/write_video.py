from logging import root
import os
import os.path as osp
import imageio
from tqdm import tqdm


def write_video(image_path, save_dir, save_name='out.mp4', fps=8):
    writer = imageio.get_writer(os.path.join(save_dir, save_name), fps=fps)
    for image in tqdm(sorted(os.listdir(image_path))):
        if image.endswith(('.jpg', '.png')):
            img = imageio.imread(os.path.join(image_path, image))
            writer.append_data(img)
    writer.close()

if __name__ == '__main__':
    # Set image path, save directory, and video name
    image_path = '/mnt/disk2/christine/viz_kitti/seq2/BEV'
    save_dir = '/mnt/disk2/christine/viz_kitti/seq2'
    video_name = '2011_09_26_drive_0056_sync_BEV'

    write_video(image_path, save_dir, video_name + '.mp4')


