import easypyxl
import os





result_dir="results/"
target_models=['vgg16','ResNet18', 'ResNet50', 'DenseNet121', 'inception_v3', 'inception_v4_timm', 'mobilenet_v2','inception_resnet_v2',  'adv_inception_v3', 'ens_adv_inception_resnet_v2']
source_model_names=['ResNet50','VGG16', 'DenseNet121', 'inception_v3']

# Table 1, RN-50
cfg_idx=[(51,'DI-MI-TI'),(76,'ATTA-MI-TI'),(92,'RDI-MI-TI'),(97,'RDI-MI-TI-SI'),(96,'RDI-MI-TI-VT'),(101,'ODI-MI-TI'),(102,'ODI-MI-TI-VT')]
display_idx=[0,1,3,4,5,6,7,8,9]

# VGG16
# cfg_idx=[(51,'DI-MI-TI'),(76,'ATTA-MI-TI'),(92,'RDI-MI-TI'),(97,'RDI-MI-TI-SI'),(96,'RDI-MI-TI-VT'),(101,'ODI-MI-TI'),(102,'ODI-MI-TI-VT')]
#display_idx=[1,2,3,4,5,6,7,8,9]

# DN-121
# cfg_idx=[(51,'DI-MI-TI'),(76,'ATTA-MI-TI'),(92,'RDI-MI-TI'),(97,'RDI-MI-TI-SI'),(96,'RDI-MI-TI-VT'),(101,'ODI-MI-TI'),(102,'ODI-MI-TI-VT')]
#display_idx=[0,1,2,4,5,6,7,8,9]

# Inc-v3
# cfg_idx=[(51,'DI-MI-TI'),(54,'MI-TI'),(76,'ATTA-MI-TI'),(92,'RDI-MI-TI'),(97,'RDI-MI-TI-SI'),(96,'RDI-MI-TI-VT'),(101,'ODI-MI-TI'),(102,'ODI-MI-TI-VT')]
#display_idx=[0,1,2,3,5,6,7,8,9]

# Table 2
#cfg_idx=[(58,'Package'),(64,'Cup'),(65,'Pillow'),(66,'T-shirt'),(68,'Ball'),(67,'Book'),(101,'3Model'),(100,'All')]
#display_idx=[0,1,2,3,4,5,6,7,8,9]

# Table 3
# cfg_idx=[(83,'5'),(82,'15'),(81,'25'),(65,'35'),(84,'45'),
# (86,'No Rand Dist'),(87,'Decreased Rand Dist'),(108,'1X'),
# (65,'random_pixel'),(88,'random_solid'),(91,'Blurred img'),(105,'No background'),
# (68,'1 ball'),(69,'2 balls'),(70,'3 balls'),(71,'4 balls')]
# display_idx=[0,1,2,3,4,5,6,7,8,9]


workbook=[]
cursor=[]
time_cursor=[]

for i in range(len(cfg_idx)):
    workbook.append(easypyxl.Workbook(result_dir+"NEW_EXP_"+str(cfg_idx[i][0])+".xlsx", backup=False))
    time_cursor.append(workbook[i].new_cursor("Experiment Info", "B5",1, reader=True))
    cursor.append(workbook[i].new_cursor("Succ_300", "C2", len(target_models), reader=True))
    title=cursor[i].read_line()
    if i==len(cfg_idx)-1:
        for j in display_idx:
            print(title[j],end=', ')
        print('')


for i in range(len(source_model_names)):
    print(source_model_names[i])
    for i in range(len(cfg_idx)):
        items=cursor[i].read_line()
        compuation_time=time_cursor[i].read_cell()
        print(cfg_idx[i][1],end=' & ')
        for j in display_idx:
            print(f"{items[j]:2.1f} & ",end='')
        print(f"{float(compuation_time):2.2f} ",end='')
        print('')

    