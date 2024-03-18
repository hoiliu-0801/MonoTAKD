"""
Written by Heng Fan
Mod@2023 by Christine Wu

The KITTI class
"""
import os
import os.path as osp
import glob
import cv2 as cv
import mayavi.mlab as mlab
import numpy as np
from tqdm import tqdm
from utility import *


class KITTI(object):
    """
    Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite,
    Andreas Geiger, Philip Lenz, and Raquel Urtasun,
    CVPR, 2012.
    """
    def __init__(self, dataset_path, label_path=None):
        '''
        :param dataset_path: path to the KITTI dataset
        '''
        super(KITTI, self).__init__()
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.split = 'test' # val, test, seq
        self.split_path = '/mnt/disk2/Data/KITTI/KITTI3D/ImageSets/{}.txt'.format(self.split)
        self.categories = ['Car', 'Pedestrian', 'Cyclist']
        self.colors = custom_colors()
        self._get_sequence_list()
        

    def _get_sequence_list(self):
        """
        :return: the sequence list
        """
        self.sequence_list = []

        # Get all frame names
        with open(self.split_path, 'r') as f:
            self.frame_names = f.readlines()
        f.close()
        self.sequence_num = len(self.frame_names) #7481

        # Store information of a sequence
        sequence = dict()
        sequence['name'] = self.split
        
        # Get image list
        img_list = []
        pcloud_list = []
        for frame in self.frame_names:
            frame_num = '{:06d}'.format(int(frame[:-1]))
            frame_img_path = os.path.join(self.dataset_path, 'image_2',  frame_num + '.png')
            img_list.append(frame_img_path)
            
            frame_pc_path = os.path.join(self.dataset_path, 'velodyne',  frame_num + '.bin')
            pcloud_list.append(frame_pc_path)
            
        sequence['img_list'] = img_list
        sequence['img_size'] = self.get_sequence_img_size(sequence['img_list'][0])
        sequence['pcloud_list'] = pcloud_list
        sequence['label_list'] = self.get_sequence_labels('gt')   # get lables in this sequence
        sequence['calib_list'] = self.get_sequence_calib(sequence['label_list'])
        
        # Visualize label or not
        if self.label_path is not None:
            sequence['pred_label_list'] = self.get_sequence_labels('pred')
        else: 
            sequence['pred_label_list'] = []

        self.sequence_list.append(sequence)


    def get_sequence_img_size(self, initial_img_path):
        """
        get the size of image in the sequence
        :return: image size
        """
        img = cv.imread(initial_img_path) 

        img_size = dict()

        img_size['height'] = img.shape[0]
        img_size['width'] = img.shape[1]

        return img_size #{'height': 375, 'width': 1242}

    def get_sequence_calib(self, label_list):
        """
        get the calib parameters
        :param sequence_name: sequence name
        :return: calib
        """
        calib_list = []
        for label in tqdm(self.frame_names, desc='Load Calib', leave=False):
            # Load data
            label_name = '{:06d}'.format(int(label[:-1])) + '.txt'
            sequence_calib_path = osp.join(self.dataset_path, 'calib', label_name)
            
            with open(sequence_calib_path, 'r') as f:
                calib_lines = f.readlines()

            calib = dict()
            calib['P0'] = np.array(calib_lines[0].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            calib['P1'] = np.array(calib_lines[1].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            calib['P2'] = np.array(calib_lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            calib['P3'] = np.array(calib_lines[3].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            calib['Rect'] = np.array(calib_lines[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)
            calib['Tr_velo_cam'] = np.array(calib_lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            calib['Tr_imu_velo'] = np.array(calib_lines[6].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            
            calib_list.append(calib)

        return calib_list

    def get_sequence_labels(self, label_type):
        """
        get labels for all frames in the sequence
        :param label_type: type of label (ground truth or prediction)
        :return: the labels of a sequence
        """
        if label_type == 'gt':
            label_path = '/home/ipl-pc/VirConv/output/models/kitti/VirConv-S-test/default/eval/epoch_2/test/default/final_result/data/'
        else:
            label_path = self.label_path

        with open(self.split_path, 'r') as f:
            frame_names = f.readlines()
        f.close()
        
        frame_id_list = []
        object_list = []    
        sequence_label = []
        for frame in tqdm(frame_names, desc=f'Load {label_type}', leave=False):
            frame_id = int(frame[:-1])
            frame = '{:06d}'.format(frame_id)
            
            # Open current label
            with open(os.path.join(label_path, (frame + '.txt')), 'r') as f:
                objects = f.readlines()
            f.close()
            
            obj_id = 0
            object_list = [] 
            for obj in objects:
                obj = obj.split()
                if len(obj) == 15:
                    object_type, truncat, occ, alpha, l, t, r, b, height, width, lenght, x, y, z, rotation = obj
                    score = 1.0
                else:
                    object_type, truncat, occ, alpha, l, t, r, b, height, width, lenght, x, y, z, rotation, score = obj
                
                # Map string to int or float
                truncat, occ = map(int, [float(truncat), occ])
                alpha, l, t, r, b, height, width, lenght, x, y, z, rotation, score = map(float, [alpha, l, t, r, b, height, width, lenght, x, y, z, rotation, score])
                
                if object_type in self.categories:
                    object = dict()
                    object['id'] = obj_id
                    object['object_type'] = object_type
                    object['truncat'] = truncat
                    object['occ'] = occ
                    object['alpha'] = alpha
                    object['bbox'] = [l, t, r, b]
                    object['dimension'] = [height, width, lenght]
                    object['location'] = [x, y, z]
                    object['rotation'] = rotation
                    object['score'] = score
                    object['frame'] = frame[:-1]

                    obj_id += 1
                    
                    object_list.append(object)
                    frame_id_list.append(frame_id)
            sequence_label.append(object_list)

        return sequence_label

    def show_sequence_rgb(self, vid_id, vis_2dbox=False, vis_3dbox=False, save_img=False, save_path=None, wait_time=30):
        """
        visualize the sequence in RGB
        :param vid_id: id of the sequence, starting from 0
        :return: none
        """

        assert vid_id>=0 and vid_id<len(self.sequence_list), \
            'The id of the sequence should be in the range [0, {}]'.format(str(self.sequence_num-1))

        sequence = self.sequence_list[vid_id]
        sequence_name = sequence['name']
        img_list = sequence['img_list']     # get the image list of this sequence
        labels = sequence['label_list']
        preds = sequence['pred_label_list']
        calibs = sequence['calib_list']
        
        if len(preds) > 0:
            draw_pred = True
        else:
            draw_pred = False

        assert len(img_list) == len(labels), 'The number of image and number of labels do NOT match!'
        assert not(vis_2dbox == True and vis_3dbox == True), 'It is NOT good to visualize both 2D and 3D boxes simultaneously!'

        # create folder to save image if not existing
        if save_img:
            if save_path is None:
                if vis_2dbox:
                    save_path = os.path.join('./seq_camera_vis', sequence_name+'_2D_box')
                elif vis_3dbox:
                    save_path = os.path.join('./seq_camera_vis', sequence_name+'_3D_box')
                else:
                    save_path = os.path.join('./seq_camera_vis', sequence_name+'_no_box')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
        # show the sequence
        for i, (img_name, img_label) in enumerate(tqdm(zip(img_list, labels), total=len(labels))):
            img = cv.imread(img_name)   # BGR image format
            thickness = 1

            # visualize 2d boxes in the image
            if vis_2dbox:
                #### DRAW Ground Truth ### 
                for object in img_label:
                    object_type = object['object_type']
                    bbox = object['bbox']
                    bbox = [int(tmp) for tmp in bbox]
                    bbox_color = [0, 255, 0] #Green   #self.colors[self.categories.index(object_type)]
                    bbox_color = (bbox_color[0], bbox_color[1], bbox_color[2])
                    cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=bbox_color, thickness=thickness)

                    # cv.putText(img, text=object_type, org=(bbox[0], bbox[1] - 5),
                    #            fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=bbox_color, thickness=thickness)

                #### DRAW Prediction ###
                if draw_pred: 
                    for object in preds[i]:
                        object_type = object['object_type']
                        bbox = object['bbox']
                        bbox = [int(tmp) for tmp in bbox]
                        bbox_color = self.colors[self.categories.index(object_type)]
                        bbox_color = (bbox_color[0], bbox_color[1], bbox_color[2])
                        cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=bbox_color, thickness=thickness)

                        # cv.putText(img, text=object_type + '-ID: ' + str(object['id']), org=(bbox[0], bbox[1] - 5),
                        #         fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=bbox_color, thickness=thickness)
                    

            # Visualize 3d boxes in the image
            if vis_3dbox:
                calib = calibs[i]
                #### DRAW Ground Truth ### 
                for object in img_label:
                    object_type = object['object_type']
                    bbox_color = [0, 255, 0]
                    bbox_color = (bbox_color[0], bbox_color[1], bbox_color[2])

                    corners_3d_img = transform_3dbox_to_image(object['dimension'], object['location'], object['rotation'], calib)

                    if corners_3d_img is None:
                        # None means object is behind the camera, and ignore this object.
                        continue
                    else:
                        corners_3d_img = corners_3d_img.astype(int)

                        ### Draw lines in the image
                        # p10-p1, p1-p2, p2-p3, p3-p0
                        cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                            (corners_3d_img[1, 0], corners_3d_img[1, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                            (corners_3d_img[2, 0], corners_3d_img[2, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                            (corners_3d_img[3, 0], corners_3d_img[3, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                            (corners_3d_img[0, 0], corners_3d_img[0, 1]), color=bbox_color, thickness=thickness)

                        # p4-p5, p5-p6, p6-p7, p7-p0
                        cv.line(img, (corners_3d_img[4, 0], corners_3d_img[4, 1]),
                            (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[5, 0], corners_3d_img[5, 1]),
                            (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[6, 0], corners_3d_img[6, 1]),
                            (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[7, 0], corners_3d_img[7, 1]),
                            (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)

                        # p0-p4, p1-p5, p2-p6, p3-p7
                        cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                            (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                            (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                            (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                            (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)

                        # draw front lines
                        cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                                (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                        cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                                (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)

                        cv.putText(img, text=object_type, org=(corners_3d_img[4, 0], corners_3d_img[4, 1]-5),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=bbox_color, thickness=thickness)

                
                #### DRAW Preds ### 
                if draw_pred:
                    for object in preds[i]:
                        object_type = object['object_type']
                        bbox_color = self.colors[self.categories.index(object_type)]
                        bbox_color = (bbox_color[0], bbox_color[1], bbox_color[2])

                        corners_3d_img = transform_3dbox_to_image(object['dimension'], object['location'], object['rotation'], calib)

                        if corners_3d_img is None:
                            # None means object is behind the camera, and ignore this object.
                            continue
                        else:
                            corners_3d_img = corners_3d_img.astype(int)

                            # draw lines in the image
                            # p10-p1, p1-p2, p2-p3, p3-p0
                            cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                                (corners_3d_img[1, 0], corners_3d_img[1, 1]), color=bbox_color, thickness=thickness)
                            cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                                (corners_3d_img[2, 0], corners_3d_img[2, 1]), color=bbox_color, thickness=thickness)
                            cv.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                                (corners_3d_img[3, 0], corners_3d_img[3, 1]), color=bbox_color, thickness=thickness)
                            cv.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                                (corners_3d_img[0, 0], corners_3d_img[0, 1]), color=bbox_color, thickness=thickness)

                            # p4-p5, p5-p6, p6-p7, p7-p0
                            cv.line(img, (corners_3d_img[4, 0], corners_3d_img[4, 1]),
                                (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                            cv.line(img, (corners_3d_img[5, 0], corners_3d_img[5, 1]),
                                (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
                            cv.line(img, (corners_3d_img[6, 0], corners_3d_img[6, 1]),
                                (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)
                            cv.line(img, (corners_3d_img[7, 0], corners_3d_img[7, 1]),
                                (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)

                            # p0-p4, p1-p5, p2-p6, p3-p7
                            cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                                (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)
                            cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                                (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                            cv.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                                (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
                            cv.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                                (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)

                            # draw front lines
                            cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                                    (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                            cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                                    (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)

            #cv.imshow('Play {}'.format(sequence['name']), img)
            # Save visualization image if you want
            if save_img:
                cv.imwrite(os.path.join(save_path, img_name.split('/')[-1].split('.')[0] + '.png'), img)
            cv.waitKey(wait_time)

        cv.destroyAllWindows()

    def show_sequence_pointcloud(self, vid_id, img_region=False, vis_box=False, save_img=False, save_path=None):
        """
        visualize the sequence in point cloud
        :param vid_id: id of the sequence, starting from 0
        :param img_region: only show point clouds in RGB image
        :param vis_box: show 3D boxes or not
        :return: none
        """

        assert 0 <= vid_id < len(self.sequence_list), 'The sequence id should be in [0, {}]'.format(str(self.sequence_num - 1))
        sequence = self.sequence_list[vid_id]
        sequence_name = sequence['name']
        pcloud_list = sequence['pcloud_list']
        labels = sequence['label_list']
        preds = sequence['pred_label_list']
        img_size = sequence['img_size']
        calibs = sequence['calib_list']
        
        draw_pred = False
        if len(preds) > 0:
            draw_pred = True

        # create folder to save image if not existing
        if save_img:
            if save_path is None:
                if vis_box:
                    save_path = os.path.join('./seq_pointcloud_vis', sequence_name+'_3D_box')
                else:
                    save_path = os.path.join('./seq_pointcloud_vis', sequence_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # # load point cloud
        # pcloud = np.fromfile(pcloud_list[0], dtype=np.float32).reshape(-1, 4)
        #
        # pcloud_fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
        # plt = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], mode='point', figure=pcloud_fig)
        # # another way is to use animate function in mlab to play cloud
        # # but somehow, it sometimes works, but sometimes fails
        #
        # @mlab.animate(delay=100)
        # def anim():
        #     for i in range(1, len(pcloud_list)):
        #         pcloud_name = pcloud_list[i]
        #         print(pcloud_name)
        #         # load point cloud
        #         pcloud = np.fromfile(pcloud_name, dtype=np.float32).reshape(-1, 4)
        #         plt.mlab_source.reset(x=pcloud[:, 0], y=pcloud[:, 1], z=pcloud[:, 2])
        #         mlab.savefig(filename='temp_img2/' + str(i) + '.png')
        #         yield
        #
        # anim()
        # mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=50.0)
        # mlab.show()

        # visualization
        pcloud_fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
        for i, pcloud_name in enumerate(tqdm(pcloud_list, total=len(pcloud_list))):
            # clear
            mlab.clf()
            calib = calibs[i]
            # BE CAREFUL!
            # the reason why doing so is because there are bin files missing in some sequences (e.g., sequence 0001)
            # e.g., in label file, the seuqnece is: 000001, 000002, 000003, 000004, 000005
            # but in bin file, the sequence is:     000001, 000004, 000005
            img_label = labels[int(pcloud_name.split('/')[-1].split('.')[0])]

            # load point cloud
            # point[:, 0]: x; point[:, 1]: y; point[:, 2]: z; point[:, 3]: reflectance information
            pcloud = np.fromfile(pcloud_name, dtype=np.float32).reshape(-1, 4)

            # remove point clouds not in RBG image
            if img_region:
                # velodyne coordinate to camera 0 coordinate
                pcloud_C2_depth, pcloud_C2 = velodyne_to_camera_2(pcloud, calib)

                # remove points out of image
                pcloud_in_img = remove_cloudpoints_out_of_image(pcloud_C2_depth, pcloud_C2, pcloud, img_size)
                pcloud = pcloud_in_img

            # show point cloud
            plot = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], np.arange(len(pcloud)), mode='point', figure=pcloud_fig)

            # load and show 3d boxes
            if vis_box:
                #### DRAW Ground Truth ### 
                for object in img_label:
                    object_type = object['object_type']
                    # bbox_color = self.colors[self.categories.index(object_type)]
                    bbox_color = [0, 255, 0]
                    bbox_color = (bbox_color[2]/255, bbox_color[1]/255, bbox_color[0]/255)
                    corners_3d = transform_3dbox_to_pointcloud(object['dimension'], object['location'], object['rotation'])

                    # draw lines
                    # a utility function to draw a line
                    def draw_line_3d(p1, p2, line_color=(0, 0, 0), fig=None):
                        mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=line_color, tube_radius=None, line_width=3, figure=fig)

                    # draw the bootom lines
                    draw_line_3d(corners_3d[0], corners_3d[1], bbox_color)
                    draw_line_3d(corners_3d[1], corners_3d[2], bbox_color)
                    draw_line_3d(corners_3d[2], corners_3d[3], bbox_color)
                    draw_line_3d(corners_3d[3], corners_3d[0], bbox_color)

                    # draw the up lines
                    draw_line_3d(corners_3d[4], corners_3d[5], bbox_color)
                    draw_line_3d(corners_3d[5], corners_3d[6], bbox_color)
                    draw_line_3d(corners_3d[6], corners_3d[7], bbox_color)
                    draw_line_3d(corners_3d[7], corners_3d[4], bbox_color)

                    # draw the vertical lines
                    draw_line_3d(corners_3d[4], corners_3d[0], bbox_color)
                    draw_line_3d(corners_3d[5], corners_3d[1], bbox_color)
                    draw_line_3d(corners_3d[6], corners_3d[2], bbox_color)
                    draw_line_3d(corners_3d[7], corners_3d[3], bbox_color)

                    # draw front lines
                    draw_line_3d(corners_3d[4], corners_3d[1], bbox_color)
                    draw_line_3d(corners_3d[5], corners_3d[0], bbox_color)

                    # mlab.text3d(x=corners_3d[5, 0], y=corners_3d[5, 1], z=corners_3d[5, 2], \
                    #             text=object_type+'-ID: '+str(object['id']), color=bbox_color, scale=0.35)
                
                #### DRAW Preds ###
                if draw_pred:
                    for object in preds[i]:
                        bbox_color = self.colors[self.categories.index(object_type)]
                        bbox_color = (bbox_color[2]/255, bbox_color[1]/255, bbox_color[0]/255)
                        corners_3d = transform_3dbox_to_pointcloud(object['dimension'], object['location'], object['rotation'])

                        # draw lines
                        # a utility function to draw a line
                        def draw_line_3d(p1, p2, line_color=(0, 0, 0), fig=None):
                            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=line_color, tube_radius=None, line_width=3, figure=fig)

                        # draw the bootom lines
                        draw_line_3d(corners_3d[0], corners_3d[1], bbox_color)
                        draw_line_3d(corners_3d[1], corners_3d[2], bbox_color)
                        draw_line_3d(corners_3d[2], corners_3d[3], bbox_color)
                        draw_line_3d(corners_3d[3], corners_3d[0], bbox_color)

                        # draw the up lines
                        draw_line_3d(corners_3d[4], corners_3d[5], bbox_color)
                        draw_line_3d(corners_3d[5], corners_3d[6], bbox_color)
                        draw_line_3d(corners_3d[6], corners_3d[7], bbox_color)
                        draw_line_3d(corners_3d[7], corners_3d[4], bbox_color)

                        # draw the vertical lines
                        draw_line_3d(corners_3d[4], corners_3d[0], bbox_color)
                        draw_line_3d(corners_3d[5], corners_3d[1], bbox_color)
                        draw_line_3d(corners_3d[6], corners_3d[2], bbox_color)
                        draw_line_3d(corners_3d[7], corners_3d[3], bbox_color)

                        # draw front lines
                        draw_line_3d(corners_3d[4], corners_3d[1], bbox_color)
                        draw_line_3d(corners_3d[5], corners_3d[0], bbox_color)

            # fix the view of the camera
            mlab.view(azimuth=180, distance=30, elevation=60, focalpoint=np.mean(pcloud, axis=0)[:-1])
            if save_img:
                mlab.savefig(filename=os.path.join(save_path, pcloud_name.split('/')[-1].split('.')[0] + '.png'))
            else:
                mlab.savefig(filename='temp_img.png')  # save the visualization image (this line is necessary for visualization)

        # mlab.show()   # do NOT use this line, as it will get the focus and pause the code
        mlab.close(all=True)
        if not save_img:
            os.remove(path='temp_img.png')  # remove temp image file

    def show_sequence_BEV(self, vid_id, img_region=False, vis_box=False, save_img=False, save_path=None):
        """
        visualize the sequence in bird's eye view
        :param vid_id: id of the sequence, starting from 0
        :param img_region: only show point clouds in RGB image
        :param vis_3dbox: show 3D boxes or not
        :return: none
        """

        assert 0 <= vid_id < len(self.sequence_list), 'The sequence id should be in [0, {}]'.format(str(self.sequence_num - 1))
        sequence = self.sequence_list[vid_id]
        sequence_name = sequence['name']
        pcloud_list = sequence['pcloud_list']
        labels = sequence['label_list']
        preds = sequence['pred_label_list']
        img_size = sequence['img_size']
        calibs = sequence['calib_list']
        
        draw_pred = False
        if len(preds) > 0:
            draw_pred = True

        # create folder to save image if not existing
        if save_img:
            if save_path is None:
                if vis_box:
                    save_path = os.path.join('./seq_BEV_vis', sequence_name + '_BEV_box')
                else:
                    save_path = os.path.join('./seq_BEV_vis', sequence_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # visualization
        pcloud_fig = mlab.figure(bgcolor=(1, 1, 1), size=(1280, 720), fgcolor=None)
        for i, pcloud_name in enumerate(tqdm(pcloud_list, total=len(pcloud_list))):
            # clear
            mlab.clf()
            calib = calibs[i]
            # BE CAREFUL!
            # the reason why doing so is because there are bin files missing in some sequences (e.g., sequence 0001)
            # e.g., in label file, the seuqnece is: 000001, 000002, 000003, 000004, 000005
            # but in bin file, the sequence is:     000001, 000004, 000005
            # img_label = labels[int(pcloud_name.split('/')[-1].split('.')[0])]
            img_label = labels[i]

            # load point cloud
            # point[:, 0]: x; point[:, 1]: y; point[:, 2]: z; point[:, 3]: reflectance information
            pcloud = np.fromfile(pcloud_name, dtype=np.float32).reshape(-1, 4)

            # remove point clouds not in RBG image
            if img_region:
                # velodyne coordinate to camera 0 coordinate
                pcloud_C2_depth, pcloud_C2 = velodyne_to_camera_2(pcloud, calib)

                # remove points out of image
                pcloud_in_img = remove_cloudpoints_out_of_image(pcloud_C2_depth, pcloud_C2, pcloud, img_size)
                pcloud = pcloud_in_img

            # show point cloud  colormap="Greys"
            plot = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], np.arange(len(pcloud)), color=(0, 0, 0), mode='point', scale_factor=0.5, figure=pcloud_fig)
            # plot.glyph.scale_mode = 'scale_by_vector'
            # plot = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], mode='point', figure=pcloud_fig)
            # plot = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], np.arange(len(pcloud)), mode='point', figure=pcloud_fig)

            # load and show 3d boxes
            if vis_box:
                for object in img_label:
                    object_type = object['object_type']
                    # bbox_color = self.colors[self.categories.index(object_type)]
                    bbox_color = [0, 255, 0]
                    bbox_color = (bbox_color[2]/255, bbox_color[1]/255, bbox_color[0]/255)
                    corners_3d = transform_3dbox_to_pointcloud(object['dimension'], object['location'], object['rotation'])

                    # draw lines
                    # a utility function to draw a line
                    def draw_line_3d(p1, p2, line_color=(0, 0, 0), fig=None):
                        mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=line_color, tube_radius=None, line_width=2, figure=fig)

                    # draw the lines in X-Y space
                    draw_line_3d(corners_3d[4], corners_3d[5], bbox_color)
                    draw_line_3d(corners_3d[5], corners_3d[6], bbox_color)
                    draw_line_3d(corners_3d[6], corners_3d[7], bbox_color)
                    draw_line_3d(corners_3d[7], corners_3d[4], bbox_color)

                    # mlab.text3d(x=corners_3d[7, 0], y=corners_3d[7, 1]-0.5, z=corners_3d[7, 2], \
                    #             text=object_type + '-ID: ' + str(object['id']), color=bbox_color, scale=0.7)
                    # mlab.text3d(x=corners_3d[7, 0], y=corners_3d[7, 1]-0.5, z=corners_3d[7, 2], \
                    #             text=object_type, color=bbox_color, scale=0.7)
                
                #### DRAW Preds ###
                if draw_pred:
                    for object in preds[i]:
                        object_type = object['object_type']
                        bbox_color = self.colors[self.categories.index(object_type)]
                        bbox_color = (bbox_color[2]/255, bbox_color[1]/255, bbox_color[0]/255)
                        corners_3d = transform_3dbox_to_pointcloud(object['dimension'], object['location'], object['rotation'])

                        # draw lines
                        # a utility function to draw a line
                        def draw_line_3d(p1, p2, line_color=(0, 0, 0), fig=None):
                            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=line_color, tube_radius=None, line_width=2, figure=fig)

                        # draw the lines in X-Y space
                        draw_line_3d(corners_3d[4], corners_3d[5], bbox_color)
                        draw_line_3d(corners_3d[5], corners_3d[6], bbox_color)
                        draw_line_3d(corners_3d[6], corners_3d[7], bbox_color)
                        draw_line_3d(corners_3d[7], corners_3d[4], bbox_color)

            # fix the view of the camera
            mlab.view(azimuth=180, distance=100, elevation=0, focalpoint=np.mean(pcloud, axis=0)[:-1])
            if save_img:
                mlab.savefig(filename=os.path.join(save_path, pcloud_name.split('/')[-1].split('.')[0] + '.png'))
            else:
                mlab.savefig(filename='temp_img.png')  # save the visualization image (this line is necessary for visualization)

        # mlab.show()   # do NOT use this line, as it will get the focus and pause the code
        mlab.close(all=True)
        if not save_img:
            os.remove(path='temp_img.png')  # remove temp image file
