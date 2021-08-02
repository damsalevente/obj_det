# My plan 

main idea: i have a picture, and i kinda have to generate a new picture with the instance segmented mask right 
this is like an encoder where i predict image to image, or like a translation where I could use a visual transformer( later ) 

these are existing solutions: 

GAN Mask R-CNN MIT thesis explores this idea, where they use the ground truth mask and the mask-rcnn output as a discriminant/generator, and the critic checks the difference between them ( basic GAN structure) 

I have a good Yolov4 model on the UAVDT dataset, but there are no instance segmented ground truth data for it 

I could use the coco or bdd100k dataset models, with mask-rcnn, or i could use the my existing yolov4 model and add the extra net to predict the masks too

This also exists: YolACT++ yolact etc. 
 
