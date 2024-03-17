import numpy as np
import json
import pathlib
from tqdm import tqdm


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        P2 = np.array(json.load(f)['intrinsic'])

    return {'P2': P2.reshape(3, 4)}
#######################################################################################
def get_objects_from_label(obj_list, calib):
    objects = []
    for obj in obj_list:
        #in lidar coord: obj_id, cls_name, x, y, z, theta, l, w, h
        obj_split = obj.split(', ')
        if (obj_split[0] != '*' or 'Sedan' not in obj_split):
            continue
        if len(obj_split) == 10:
            objects.append(Object3D_(obj_split[1:], calib))
        else:
            objects.append(Object3D_(obj_split[2:], calib))
    return objects

class Object3D_(object):
    def __init__(self, obj, calib):
        obj_id, cls_name, x, y, z, theta, l, w, h = obj
        ### Convert from string to float
        x, y, z = float(x), float(y), float(z)
        theta = float(theta)*np.pi/180.
        l, w, h = 2*float(l), 2*float(w), 2*float(h)
        ### Define camera calibration parameters
        intrinsic = [600.0, 570.0, 640.0, 360.0]
        extrinsic = [-0.8, 0.9, 0.6, 0.5, -0.2, 0]
        ### Transform x, y, z, from lidar to camera coordinate
        rot, tra = get_rotation_and_translation_from_extrinsic(extrinsic)
        ldr_pt = np.array([[x, y, z]]) # expand shape dims for lib func input
        cam_pt = get_pointcloud_with_rotation_and_translation(ldr_pt, rot, tra)
        # xc, yc, zc = cam_pt[0][2], cam_pt[0][1], cam_pt[0][0] # re-order for kitti: Xl=Zc, -Yl=Xc, -Zl=Yc
        xc, yc, zc = -y, -z, x
        ### Generate 8 corner points for 3D B-box in Cam Coord
        self.cam_corners = get_corners_in_camera_coordinate(l, w, h, theta, cam_pt[0])
        self.pix_corners = get_pixel_from_point_cloud_in_camera_coordinate(self.cam_corners, intrinsic) # (8, 2)
        ### Check if 3D B-box is FULLY in image
        bbox_trunked = 0 # false for having the whole bbox
        if np.any(self.pix_corners < 0):
            bbox_trunked = 1
        ### Get 2D bounding box information
        x_min, y_min, x_max, y_max = self.pseudo_2d(self.pix_corners)
        W, H = x_max-x_min, y_max-y_min
        # print(W, H)
        self.box2d = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        self.center_2d = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
        ### Get 3D B-box center & l, r, t, b
        # print("cam_pt:",cam_pt)
        self.pix_center_3d = get_pixel_from_point_cloud_in_camera_coordinate(cam_pt, intrinsic)[0]  # [x, y]
        left2d, right2d, top2d, bottom2d = self.pix_center_3d[0] - x_min, \
                                           x_max - self.pix_center_3d[0], \
                                           self.pix_center_3d[1] - y_min, \
                                           y_max - self.pix_center_3d[1]
        self.lrtb = np.array((float(left2d), float(right2d), float(top2d), float(bottom2d)), dtype=np.float32)

        ### Populate the Object fields specifically for KITTI
        self.cls_type = cls_name.replace(" ", "")
        self.trucation = float(0)
        self.occlusion = float(0)  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.h = h
        self.w = l
        self.l = w
        self.pos = np.array((float(xc), float(yc + h/2), float(zc)), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = theta # radians
        self.alpha = float(calib.ry2alpha(self.ry, self.center_2d[0])) # not sure
        self.score = float(0.0) # not sure
        self.level_str = None
        # self.level_str = 'Moderate'
        self.level = 1

        ### Save object summary in KITTI format
        self.src = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.trucation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.pos[0], self.pos[1], self.pos[2],
                       self.ry)
    # Helper to get 2D B-box: min_x, min_y, max_x, max_y
    def pseudo_2d(self, box_2d_ori):
        assert box_2d_ori.shape[-1]==2
        return [np.min(box_2d_ori,0)[0],np.min(box_2d_ori,0)[1],np.max(box_2d_ori,0)[0],np.max(box_2d_ori,0)[1]]

def get_corners_in_camera_coordinate(l, w, h, rad_rot, cam_pt): #lwh, theta directly from kradar labels
    corners_x = np.array([l, l, l, l, -l, -l, -l, -l]) / 2
    corners_y = np.array([w, w, -w, -w, w, w, -w, -w]) / 2
    corners_z = np.array([h, -h, h, -h, h, -h, h, -h]) / 2

    corners = np.row_stack((corners_x, corners_y, corners_z))

    rotation_matrix = np.array([
        [np.cos(rad_rot), -np.sin(rad_rot), 0.0],
        [np.sin(rad_rot),  np.cos(rad_rot), 0.0],
        [0.0, 0.0, 1.0]])

    return rotation_matrix.dot(corners).T + cam_pt

def get_pixel_from_point_cloud_in_camera_coordinate(point_cloud_xyz, intrinsic):
    '''
    * in : pointcloud in np array (nx3)
    * out: projected pixel in np array (nx2)
    '''

    process_pc = point_cloud_xyz
    if (np.shape(point_cloud_xyz) == 1):
        num_points = 0
    else:
        #Temporary fix for when shape = (0.)
        try:
            num_points, _ = np.shape(point_cloud_xyz)
        except:
            num_points = 0
    fx, fy, px, py = intrinsic

    pixels = []
    for i in range(num_points):
        xc, yc, zc = process_pc[i,:]
        y_pix = py - fy*zc/xc
        x_pix = px - fx*yc/xc

        pixels.append([x_pix, y_pix])
    pixels = np.array(pixels)

    return pixels

def get_pointcloud_with_rotation_and_translation(point_cloud_xyz, rot, tra):
    pc_xyz = point_cloud_xyz.copy()
    num_points = len(pc_xyz)

    for i in range(num_points):
        point_temp = pc_xyz[i,:]
        point_temp = np.reshape(point_temp, (3,1))

        point_processed = np.dot(rot, point_temp) + tra
        point_processed = np.reshape(point_processed, (3,))

        pc_xyz[i,:] = point_processed

    return pc_xyz

def get_rotation_and_translation_from_extrinsic(extrinsic, is_deg = True):
    # ext_copy = [(-0.8*np.pi/180), (0.9*np.pi/180), (0.6*np.pi/180), 0.5, -0.2, 0] #
    ext_copy = extrinsic.copy() # if not copy, will change the parameters permanently

    if is_deg:
        ext_copy[:3] = list(map(lambda x: x*np.pi/180., extrinsic[:3]))

    roll, pitch, yaw = ext_copy[:3]
    x, y, z = ext_copy[3:]

    ### Roll-Pitch-Yaw Convention
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    c_r = np.cos(roll)
    s_r = np.sin(roll)

    R_yaw = np.array([[c_y, -s_y, 0.],[s_y, c_y, 0.],[0., 0., 1.]])
    R_pitch = np.array([[c_p, 0., s_p],[0., 1., 0.],[-s_p, 0., c_p]])
    R_roll = np.array([[1., 0., 0.],[0., c_r, -s_r],[0., s_r, c_r]])

    R = np.dot(np.dot(R_yaw, R_pitch), R_roll)
    trans = np.array([[x],[y],[z]])

    return R, trans



########################################################################################

class Calibration(object):
    # def __init__(self, calib_file):
    #     if isinstance(calib_file, str):
    #         calib = get_calib_from_file(calib_file)
    #     else:
    #         calib = calib_file
    def __init__(self):
        intrinsics = [600.0, 570.0, 640.0, 360.0]
        fx, fy, px, py = intrinsics
        ### Create Intrinsic Matrix
        P2 = np.array([
            [fx, 0.0, px],
            [0.0, fy, py],
            [0.0, 0.0, 1.0]
        ])
        #[[600.   0. 640.   0.]
        # [  0. 570. 360.   0.]
        # [  0.   0.   1.   0.]]
        self.P2 = np.hstack((P2, np.array([0., 0., 0.]).reshape(3, -1)))  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack(
            (pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - \
            self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate(
            (corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :,
                                          2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate(
            (x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate(
            (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu ** 2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d ** 2 - x ** 2 - y ** 2)
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return pts_rect

    def inverse_rigid_trans(self, Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def alpha2ry(self, alpha, u):
        """
        Get rotation_y by alpha + theta - 180
        alpha : Observation angle of object, ranging [-pi..pi]
        x : Object center x to the camera center (x-W/2), in pixels
        rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """
        ry = alpha + np.arctan2(u - self.cu, self.fu)

        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi

        return ry

    def ry2alpha(self, ry, u):
        alpha = ry - np.arctan2(u - self.cu, self.fu)

        if alpha > np.pi:
            alpha -= 2 * np.pi
        if alpha < -np.pi:
            alpha += 2 * np.pi

        return alpha

    def flip(self, img_size):
        wsize = 4
        hsize = 2
        p2ds = (np.concatenate(
            [np.expand_dims(np.tile(np.expand_dims(np.linspace(0, img_size[0], wsize), 0), [hsize, 1]), -1),
             np.expand_dims(np.tile(np.expand_dims(np.linspace(
                 0, img_size[1], hsize), 1), [1, wsize]), -1),
             np.linspace(2, 78, wsize * hsize).reshape(hsize, wsize, 1)], -1)).reshape(-1, 3)
        p3ds = self.img_to_rect(p2ds[:, 0:1], p2ds[:, 1:2], p2ds[:, 2:3])
        p3ds[:, 0] *= -1
        p2ds[:, 0] = img_size[0] - p2ds[:, 0]

        # self.P2[0,3] *= -1
        cos_matrix = np.zeros([wsize * hsize, 2, 7])
        cos_matrix[:, 0, 0] = p3ds[:, 0]
        cos_matrix[:, 0, 1] = cos_matrix[:, 1, 2] = p3ds[:, 2]
        cos_matrix[:, 1, 0] = p3ds[:, 1]
        cos_matrix[:, 0, 3] = cos_matrix[:, 1, 4] = 1
        cos_matrix[:, :, -2] = -p2ds[:, :2]
        cos_matrix[:, :, -1] = (-p2ds[:, :2] * p3ds[:, 2:3])
        new_calib = np.linalg.svd(cos_matrix.reshape(-1, 7))[-1][-1]
        new_calib /= new_calib[-1]

        new_calib_matrix = np.zeros([4, 3]).astype(np.float32)
        new_calib_matrix[0, 0] = new_calib_matrix[1, 1] = new_calib[0]
        new_calib_matrix[2, 0:2] = new_calib[1:3]
        new_calib_matrix[3, :] = new_calib[3:6]
        new_calib_matrix[-1, -1] = self.P2[-1, -1]
        self.P2 = new_calib_matrix.T
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)


def get_label_annos_from_txt(list_path_label):
    list_objects = []
    # print("list_path_label:", list_path_label)
    for path_label in tqdm(list_path_label):
        list_objects.append(get_label_anno(path_label))
    return list_objects


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()[1:]
    if len(lines) == 0:
        # print("No GT")
        return annotations

    offset = 0
    if(len(lines[0].strip().split(','))) == 11:
        # print('* Exception error (Dataset): length of values is 11')
        offset = 1
    else:
        print('* Exception error (Dataset): length of values is 10')
        #print(path_label)

    content = [line.strip().split(',') for line in lines]
    annotations['name'] = np.array([x[2+offset][1:] for x in content])
    annotations['truncated'] = np.array([0 for x in content])
    annotations['occluded'] = np.array([0 for x in content])
    annotations['alpha'] = np.array([float(x[6+offset]) for x in content])
    annotations['bbox'] = np.array(
        [[0, 0, 50 ,50] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[7+offset:10+offset]] for x in content]).reshape(
        -1, 3)[:, [0, 2, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[3+offset:6+offset]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[6+offset]) for x in content])
    annotations['score'] = np.ones([len(annotations['bbox'])])
    return annotations


def get_label_anno_pred(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    # print(label_path)
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0 or len(lines[0]) < 15:
        return None

    content = [line.strip().split(' ') for line in lines]
    annotations['name'] = np.array([x[0] for x in content])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
        -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros([len(annotations['bbox'])])
    return annotations


def get_label_annos_from_txt_pred(result_folder, image_ids=None):
    if image_ids is None:
        filepaths = pathlib.Path(result_folder).glob('*.txt')
        image_ids = [p.stem for p in filepaths]
        image_ids = sorted(image_ids)
    annos = []
    result_folder = pathlib.Path(result_folder)
    for image_id in image_ids:
        result_filename = result_folder / (image_id + '.txt')
        temp_anno = get_label_anno_pred(result_filename)
        if temp_anno is not None:
            annos.append(temp_anno)
    return annos
# import numpy as np
def get_pointcloud_with_rotation_and_translation(point_cloud_xyz, rot, tra):
    pc_xyz = point_cloud_xyz.copy()
    num_points = len(pc_xyz)
    for i in range(num_points):
        point_temp = pc_xyz[i,:]
        point_temp = np.reshape(point_temp, (3,1))
        point_processed = np.dot(rot, point_temp) + tra
        point_processed = np.reshape(point_processed, (3,))
        pc_xyz[i,:] = point_processed
    return pc_xyz

def get_rotation_and_translation_from_extrinsic(extrinsic, is_deg = True):
    ext_copy = extrinsic.copy() # if not copy, will change the parameters permanently
    if is_deg:
        ext_copy[:3] = list(map(lambda x: x*np.pi/180., extrinsic[:3]))

    roll, pitch, yaw = ext_copy[:3]
    x, y, z = ext_copy[3:]

    ### Roll-Pitch-Yaw Convention
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    c_r = np.cos(roll)
    s_r = np.sin(roll)

    R_yaw = np.array([[c_y, -s_y, 0.],[s_y, c_y, 0.],[0., 0., 1.]])
    R_pitch = np.array([[c_p, 0., s_p],[0., 1., 0.],[-s_p, 0., c_p]])
    R_roll = np.array([[1., 0., 0.],[0., c_r, -s_r],[0., s_r, c_r]])

    R = np.dot(np.dot(R_yaw, R_pitch), R_roll)
    trans = np.array([[x],[y],[z]])

    return R, trans

def get_pixel_from_point_cloud_in_camera_coordinate(point_cloud_xyz, intrinsic):
    '''
    * in : pointcloud in np array (nx3)
    * out: projected pixel in np array (nx2)
    '''

    process_pc = point_cloud_xyz.copy()
    if (np.shape(point_cloud_xyz) == 1):
        num_points = 0
    else:
        #Temporary fix for when shape = (0.)
        try:
            num_points, _ = np.shape(point_cloud_xyz)
        except:
            num_points = 0
    fx, fy, px, py = intrinsic

    pixels = []
    for i in range(num_points):
        xc, yc, zc = process_pc[i,:]
        y_pix = py - fy*zc/xc
        x_pix = px - fx*yc/xc

        pixels.append([x_pix, y_pix])
    pixels = np.array(pixels)

    return pixels



def convert2kitti(tuple_obj):
    """
    * ([type], [truncated], [occluded], [alpha], [2D bbox l, t, r, b], [h ,w ,l], [x, y, z], [theta], [conf score])
    ('*, 0, 0, Sedan', x, y, z, theta, l, w, h)
    (cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj) 2 kitti
    """
    cls_name, idx_cls, [xc,yc,zc,rot_rad,xl,yl,zl], idx_obj = tuple_obj
    # np.transpose(a,(0,))
    list_value=[]
    list_value.append(cls_name) # type
    list_value.append(0) # None
    list_value.append(0) # None
    list_value.append(rot_rad) # alpha
    list_value.append(0) # 2D l
    list_value.append(0) # 2D t
    list_value.append(50) # 2D r
    list_value.append(50) # 2D b
    list_value.append(zl) # 3D h
    list_value.append(yl) # 3D w
    list_value.append(xl) # 3D l
    list_value.append(xc) # 3D x
    list_value.append(yc) # 3D y
    list_value.append(zc) # 3D z
    list_value.append(rot_rad) # theta
    list_value.append(1) # conf score
    return(list_value)

class Object3d_(object):
    def __init__(self, obj, calib):
        cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = obj
        # print(obj)
        self.cls_name = cls_name
        self.cls_type = cls_name
        self.trucation = float(0)
        # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.occlusion = float(0)
        # self.alpha = float(label[3])
        self.box2d = [0,0,0,0]
        self.center_2d = [0,0]
        # print(self.center_2d)
        self.h = float(w)
        self.w = float(h)
        self.l = float(l)
        # center convertion
        # x, y, z = y, z-self.h / 2, x
        self.pos = np.array((float(x), float(
            y + self.h / 2), float(z)), dtype=np.float32)
        # print("self.pos:",self.pos)
        # self.pos: [-3.5917563  2.0425503  8.5544615]
        # self.pos: [-3.1146202  1.7594954 17.33369  ]
        # self.pos: [-2.463287   1.4045959 44.09929  ]
        # self.pos: [-1.4102565   0.73712903 53.60175   ]
        # self.pos = np.array((float(y), float(
        #     z + self.h / 2), float(x)), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(theta)
        self.alpha = 0 #calib.ry2alpha(self.ry, self.center_2d[0])
        # print(self.alpha)
        self.score = -1.0
        self.level_str = None
        self.level = 2 # for moderate?

### %Christine
class Object3d(object):
    def __init__(self, obj, calib):
        cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj = obj
        # print(obj)
        self.cls_type = cls_name
        self.trucation = float(0)
        # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.occlusion = float(0)
        # self.alpha = float(label[3])
        self.box2d = None
        self.center_2d = None
        # print(self.center_2d)
        self.h = float(w)
        self.w = float(h)
        self.l = float(l)
        self.pos = np.array((float(x), float(
            y + self.h / 2), float(z)), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(theta)
        self.alpha = 0 #calib.ry2alpha(self.ry, self.center_2d[0])
        # print(self.alpha)
        self.score = -1.0
        self.level_str = None
        self.level = 2 # for moderate?


    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -
                     l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2,
                     w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]
                            ) / voxel_size).astype(np.int32)
            box2d[:, 1] = Object3d.BEV_SHAPE[0] - 1 - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(
                np.int32)
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor(
                (self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            cv = Object3d.BEV_SHAPE[0] - 1 - \
                ((self.pos[2] - Object3d.MIN_XZ[1]) /
                 voxel_size).astype(np.int32)
            half_l, half_w = int(self.l / voxel_size /
                                 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                    % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                       self.pos, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.trucation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.pos[0], self.pos[1], self.pos[2],
                       self.ry)
        return kitti_str

#######################################################################################
### K-RADAR CAMERA TO LIDAR COORDINATE LIB FUNCTIONS
#######################################################################################
def get_pointcloud_with_rotation_and_translation(point_cloud_xyz, rot, tra):
    pc_xyz = point_cloud_xyz.copy()
    num_points = len(pc_xyz)
    for i in range(num_points):
        point_temp = pc_xyz[i,:]
        point_temp = np.reshape(point_temp, (3,1))
        point_processed = np.dot(rot, point_temp) + tra
        point_processed = np.reshape(point_processed, (3,))
        pc_xyz[i,:] = point_processed
    return pc_xyz

def get_rotation_and_translation_from_extrinsic(extrinsic, is_deg = True):
    ext_copy = extrinsic.copy() # if not copy, will change the parameters permanently
    if is_deg:
        ext_copy[:3] = list(map(lambda x: x*np.pi/180., extrinsic[:3]))

    roll, pitch, yaw = ext_copy[:3]
    x, y, z = ext_copy[3:]

    ### Roll-Pitch-Yaw Convention
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    c_r = np.cos(roll)
    s_r = np.sin(roll)

    R_yaw = np.array([[c_y, -s_y, 0.],[s_y, c_y, 0.],[0., 0., 1.]])
    R_pitch = np.array([[c_p, 0., s_p],[0., 1., 0.],[-s_p, 0., c_p]])
    R_roll = np.array([[1., 0., 0.],[0., c_r, -s_r],[0., s_r, c_r]])

    R = np.dot(np.dot(R_yaw, R_pitch), R_roll)
    trans = np.array([[x],[y],[z]])

    return R, trans

def get_pixel_from_point_cloud_in_camera_coordinate(point_cloud_xyz, intrinsic):
    '''
    * in : pointcloud in np array (nx3)
    * out: projected pixel in np array (nx2)
    '''

    process_pc = point_cloud_xyz.copy()
    if (np.shape(point_cloud_xyz) == 1):
        num_points = 0
    else:
        #Temporary fix for when shape = (0.)
        try:
            num_points, _ = np.shape(point_cloud_xyz)
        except:
            num_points = 0
    fx, fy, px, py = intrinsic

    pixels = []
    for i in range(num_points):
        xc, yc, zc = process_pc[i,:]
        y_pix = py - fy*zc/xc
        x_pix = px - fx*yc/xc
        pixels.append([x_pix, y_pix])
    pixels = np.array(pixels)

    return pixels
