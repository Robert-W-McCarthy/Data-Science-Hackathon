from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.catalog import MetadataCatalog,DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
import cv2

register_coco_instances("card", {}, "./midv500_coco.json", "./dunzip")

card_metadata = MetadataCatalog.get("card")
dataset_dicts = DatasetCatalog.get("card")

#print (dataset_dicts[0])


for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d['file_name'])
    visualizer = Visualizer(img[:, :, ::-1], metadata=card_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite('img.png',vis.get_image()[:, :, ::-1])


cfg = get_cfg()
cfg.merge_from_file(
    "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ("card",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.02
# cfg.SOLVER.MAX_ITER = (
#     300
# )  # 300 iterations seems good enough, but you can certainly train longer
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
#     128
# )  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("card", )
predictor = DefaultPredictor(cfg)


from detectron2.utils.visualizer import ColorMode

#for d in random.sample(dataset_dicts, 3):
path_test=''
image_lis=os.listdir(path)
for i in image_lis:    
    im = cv2.imread(path+'/'+i)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=card_metadata, 
                    scale=0.8, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    dir='/mnt/summerpod2/Hyperverge/detectron2/midv500/data/predictions/'
    cv2.imwrite(dir+'pred'+i+'.png',v.get_image()[:, :, ::-1])
