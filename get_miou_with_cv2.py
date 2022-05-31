"""
made by tree.jhk@gmail.com
https://github.com/tree-jhk
https://velog.io/@tree_jhk/%EB%AA%A8%EB%93%A0-%EB%AA%A8%EC%96%91%EC%9D%98-IoU-%EA%B3%84%EC%82%B0%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95
"""

import cv2
# import numpy
import os
import torch

global standard_threshold
global flag
flag = True
standard_threshold = 135

model=(torch.load("../CrackForest/weights_79680장_20_32_deep.pt", map_location='cpu'))
flag = torch.cuda.is_available()
if (flag==True):
    model=(torch.load("../CrackForest/weights_79680장_20_32_deep.pt"))
    model.cuda()
else:
    model=(torch.load("../CrackForest/weights_79680장_20_32_deep.pt", map_location='cpu'))

def case_threshold(path,threshold_value):
    img=cv2.imread(path)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    if(threshold_value>standard_threshold):
        res,img_gray_thr = cv2.threshold(img_gray,threshold_value,255,cv2.THRESH_BINARY)
    else:
        res,img_gray_thr = cv2.threshold(img_gray,threshold_value,255,cv2.THRESH_BINARY_INV)
    return img_gray_thr

def iou(intersection,union):
    if(union!=0):
        return intersection/union
    else:
        return 0
    
def get_mean_miou(raw_images_path,groundtruth_images_path):
    
    sum_miou=0
    
    raw_images_list=(os.listdir(raw_images_path))
    raw_images_list.sort()
    raw_images_count=len(raw_images_list)
    
    for image_name in raw_images_list:
        if os.path.isfile(raw_images_path+"/"+image_name):
            raw_image = cv2.imread(raw_images_path+"/"+image_name, cv2.IMREAD_COLOR)
            
            raw_image = cv2.resize((cv2.imread(raw_images_path+"/"+image_name, cv2.IMREAD_COLOR))
                                   ,(224,224)).transpose(2,0,1).reshape(1,3,224,224)
            if(torch.cuda.is_available()):
                semantic_image = model(torch.from_numpy(raw_image).type(torch.cuda.FloatTensor)/255)
                semantic_image = semantic_image['out'].cpu().detach().numpy()
                # np.transpose(image,(2,0,1)) for HWC->CHW transformation
                transpose_semantic_image = semantic_image[0].transpose(1,2,0)
                transpose_semantic_image = cv2.resize(transpose_semantic_image,(224,224))
            else:
                semantic_image = model(torch.from_numpy(raw_image).type(torch.FloatTensor)/255)
                semantic_image = semantic_image['out'].detach().numpy()
                transpose_semantic_image = semantic_image[0].transpose(1,2,0)
                transpose_semantic_image = cv2.resize(transpose_semantic_image,(224,224))
                
            transpose_semantic_image_name='../CrackForest/save'+'/'+image_name+'.png'
            
            ground_truth_image_name='../CrackForest/save'+'/'+'g_t'+image_name+'.png'
            
            cv2.imwrite(transpose_semantic_image_name,transpose_semantic_image*255)
            # transpose_semantic_image
            ground_truth_image=cv2.resize(cv2.imread(groundtruth_images_path+'/'+image_name,cv2.IMREAD_COLOR),(224,224))
            cv2.imwrite(ground_truth_image_name,ground_truth_image)
            
            # 차선변경영역
            # 160 이상인 영역은 흰색 나머지 검정색
            class1_semantic=case_threshold(transpose_semantic_image_name,160)
            cv2.imshow(image_name,class1_semantic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            class1_groundtruth=case_threshold(ground_truth_image_name,160)
            cv2.imshow(image_name,class1_groundtruth)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # 배경영역
            # 10 이하인 영역은 흰색 나머지 검정색
            class2_semantic=case_threshold(transpose_semantic_image_name,10)
            class2_groundtruth=case_threshold(ground_truth_image_name,10)

            # 주행가능영역
            # 10 이상 160 이하인 영역은 흰색 나머지 검정색
            class3_semantic=cv2.bitwise_not(cv2.bitwise_or(class1_semantic,class2_semantic))
            class3_groundtruth=cv2.bitwise_not(cv2.bitwise_or(class1_groundtruth,class2_groundtruth))

            # 차선변경영역 iou
            class_iou =[]

            class1_intersection = cv2.countNonZero(cv2.bitwise_and(class1_semantic, class1_groundtruth))
            class1_union=cv2.countNonZero(cv2.bitwise_or(class1_semantic, class1_groundtruth))
            class_iou.append(iou(class1_intersection,class1_union))

            # 배경영역 iou
            class2_intersection = cv2.countNonZero(cv2.bitwise_and(class2_semantic, class2_groundtruth))
            class2_union=cv2.countNonZero(cv2.bitwise_or(class2_semantic, class2_groundtruth))
            class_iou.append(iou(class2_intersection,class2_union))

            # 주행가능영역 iou
            class3_intersection = cv2.countNonZero(cv2.bitwise_and(class3_semantic, class3_groundtruth))
            class3_union=cv2.countNonZero(cv2.bitwise_or(class3_semantic, class3_groundtruth))
            class_iou.append(iou(class3_intersection,class3_union))

            ch = 0
            sum = 0
            for i in range(len(class_iou)):
                if(class_iou != 0):
                    ch+=1
                    sum+=class_iou[i]
            if(ch!=0):
                miou = sum/ch
                sum_miou+=miou
                print(miou,image_name,k)
            else:
                print("No miou")
                
        else:
            print("file does not exists")
            return 0
    return sum_miou/raw_images_count

raw_images_path = "../CrackForest/changed_Images_320장"
groundtruth_images_path = "../CrackForest/changed_Masks_320장"

if os.path.isdir(raw_images_path):
    print(get_mean_miou(raw_images_path,groundtruth_images_path))
else:
    print("directory does not exists")