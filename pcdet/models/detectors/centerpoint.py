import time
from pcdet.utils import common_utils
from .detector3d_template import Detector3DTemplate


class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
                
        # for measure time only
        self.module_time_meter = common_utils.DictAverageMeter()

    def forward(self, batch_dict, record_time=None):
        for cur_module in self.module_list:
            if record_time:
                end = time.time()
            batch_dict = cur_module(batch_dict)
            if record_time:
                module_name = str(type(cur_module)).split('.')[-1][:-2]
                self.module_time_meter.update(module_name, (time.time() - end)*1000)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if record_time:
                end = time.time()
                
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            
            if record_time:
                self.module_time_meter.update('post_processing', (time.time() - end)*1000)
            
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
