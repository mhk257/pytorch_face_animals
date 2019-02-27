# Animal Face Alignment


## Human Face Alignment
### Models trained on 300W - public, comprising 3148 training images.
Random data augmentation was used with following parameters: scale (0.75-1.25), rotation (-30 to 30 degrees), and color jittering on each plane (0.6-1.4). 

#### Dataset: 300W-public, Testing images: 689    

<table>
<tr><th>norm_type: Interocular distance </th><th>norm_type: Face Size</th></tr>
<tr><td>

| Arch.  | train_68pts | train_9pts |   
| -- | -- | -- | 
| HG-3 | 0.0342 | 0.0346 |    
| HG-2 | 0.0345 | 0.0352 |  

</td><td>
 
| Arch.  | train_68pts | train_9pts |   
| -- | -- | -- | 
| HG-3 | 0.0141 | 0.0140 |    
| HG-2 | 0.0142 | 0.0143 |

</td></tr> </table>
 
#### Dataset: 300W-private, Testing images: 600  

<table>
<tr><th>norm_type: Interocular distance </th><th>norm_type: Face Size</th></tr>
<tr><td>

| Arch.  | train_68pts | train_9pts |   
| -- | -- | -- | 
| HG-3 | 0.0426 | 0.0636 |    
| HG-2 | 0.0437 | 0.0613 |  

</td><td>
 
| Arch.  | train_68pts | train_9pts |   
| -- | -- | -- | 
| HG-3 | 0.0178 | 0.0231 |    
| HG-2 | 0.0182 | 0.0226 |

</td></tr> </table>


#### Dataset: COFW, Testing images: 507  

<table>
<tr><th>norm_type: Interocular distance </th><th>norm_type: Face Size</th></tr>
<tr><td>

| Arch.  | train_68pts | train_9pts |   
| -- | -- | -- | 
| HG-3 | 0.0574 | 0.0819 |    
| HG-2 | 0.0587 | 0.0861 |  

</td><td>
 
| Arch.  | train_68pts | train_9pts |   
| -- | -- | -- | 
| HG-3 | 0.0260 | 0.0327 |    
| HG-2 | 0.0267 | 0.0344 |

</td></tr> </table>


## Animal Face Alignment  
### Models trained on Animal-Faces dataset, comprising training 13991 images. There is no overlap of species between train and test set.  
Random data augmentation: Same as above.  
#### Dataset: Animal-Faces, Testing images: 3577  
norm_type: face size  

| Arch.  | train_9pts |
| ------------- | ------------- | 
| HG-3 | 0.0651 |    
| HG-2 | 0.0662 | 

### Models trained on Animal-Faces dataset, comprising 14168 training images. Split of train/test contains 80/20 per each class.  
Random data augmentation: Same as above.  
#### Dataset: Animal-Faces, Testing images: 3400   
norm_type: face size  

| Arch.  | train_9pts |
| ------------- | ------------- | 
| HG-3 | 0.0548 |    
| HG-2 | 0.0556 | 




 
