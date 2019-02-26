# pytorch_face_animals


## Human Face Alignment
### Models trained on 300W - public, comprising 3148 training images.
Random data augmentation was used with following parameters: scale (0.75-1.25), rotation (-30 to 30 degrees), and color jittering on each plane (0.6-1.4). 

#### Dataset: 300W-public, Testing images: 689  
norm_type: interocular distance  

| Arch.  | train_68pts | train_9pts | 
| ------------- | ------------- | ------------- | 
| HG-3 | 0.0342 | 0.0346 |    
| HG-2 | 0.0345 | 0.0352 |  

#### Dataset: 300W-private, Testing images: 600    
norm_type: interocular distance  

| Arch.  | train_68pts | train_9pts | 
| ------------- | ------------- | ------------- | 
| HG-3 | 0.0342 | 0.0346 |    
| HG-2 | 0.0345 | 0.0352 |

#### Dataset: COFW, Testing images: 507    
norm_type: interocular distance

| Arch.  | train_68pts | train_9pts | 
| ------------- | ------------- | ------------- | 
| HG-3 | 0.0342 | 0.0346 |    
| HG-2 | 0.0345 | 0.0352 |


## Animal Face Alignment  
### Models trained on Animal-Faces dataset, comprising training 13991 images. There is no overlap of species between train and test set.  
Random data augmentation: Same as above.  
#### Dataset: Animal-Faces, Testing images: 3577  
norm_type: face size  

| Arch.  | train_9pts |
| ------------- | ------------- | 
| HG-3 | 0.0546 |    
| HG-2 |  |  | 

### Models trained on Animal-Faces dataset, comprising 14168 training images. Split of train/test contains 80/20 per each class.  
Random data augmentation: Same as above.  
#### Dataset: Animal-Faces, Testing images: 3400   
norm_type: face size  

| Arch.  | train_9pts |
| ------------- | ------------- | 
| HG-3 | 0.0546 |    
| HG-2 |  |  | 




 
