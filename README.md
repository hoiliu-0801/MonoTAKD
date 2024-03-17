# MonoTAKD
Paper : MonoTAKD: Teaching assistant knowledge distillation for monocular 3D object detection.

Train
python train_TAKD.py --cfg cfgs/kitti_models/TAKD/TAKD-scd/kitti_R50_scd_TAKD.yaml --pretrained_lidar_model ../checkpoints/scd-teacher-kitti.pth --pretrained_img_model ../checkpoints/cmkd-scd-2161.pth

Test
python test_TAKD.py --cfg cfgs/kitti_models/TAKD/TAKD-scd/kitti_R50_scd_TAKD.yaml --ckpt ../checkpoints/twcc_8015_checkpoint_epoch_10.pth

Train Teacher:
python train.py --cfg cfgs/kitti_models/second_teacher.yaml --ckpt ../checkpoints/scd-teacher-kitti.pth --pretrained_model ../checkpoints/scd-teacher-kitti.pth 
python train.py --cfg cfgs/kitti_models/second.yaml  (w/o pretrained)

Test Teacher:
python test.py --cfg cfgs/kitti_models/second_teacher.yaml --ckpt ../checkpoints/scd-teacher-kitti.pth --save_to_file

Tensorboard:
tensorboard --logdir ../output/kitti_models/TAKD/TAKD-scd/kitti_R50_scd_TAKD/
