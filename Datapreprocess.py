import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F 
from pathmanager import segpatient_split_path, segment_path_trimmed
import matplotlib.pyplot as plt
from IPython import display



label_dict = {
    'Right_Ventricle': 5,
    'Left_Ventricle': 3,
    'Myocardium': 1
}


def nii2tensor(nii_data_path:str) -> torch.Tensor:
    nii2_tensor = torch.as_tensor(nib.load(nii_data_path).get_fdata())
   #if input nii2image (W, H, D, T) -> tensor (T, D, H, W), where T= time frame
    if nii2_tensor.dim() == 4:
        return nii2_tensor.permute(3, 2, 0, 1)
   #if input nii2image (W, H, T) -> tensor (T, H, W) or nii2image (W, H, D) -> tensor (D, H, W)
    elif nii2_tensor.dim() == 3:
        return nii2_tensor.permute(2, 0, 1)


def rotation_matrix(theta = 60) -> torch.Tensor(size=(2,2)):
    theta_rad = np.radians(theta)
    s = np.sin(theta_rad)
    c = np.cos(theta_rad)
    R = np.array(((c,-s), 
                  (s,c)))
    R_tensor = torch.as_tensor(R).float()
    
    return R_tensor

def affine_transform(point: torch.Tensor):
    return torch.matmul(rotation_matrix(), point.t()).t().float()
    

def get_sparse_indices(seg_tensor:torch.Tensor, label) -> torch.Tensor:
    segment = (seg_tensor == label)
    indices = segment.to_sparse().indices()
    return indices


def affine_matrix(rotation:torch.Tensor, translation:torch.Tensor):
    return torch.cat((rotation.view(2,2), translation.view(2,1)),dim =1)


def spatialtransformer(input: torch.Tensor, affine_matrix: torch.Tensor):
    grid = F.affine_grid(affine_matrix, input.shape)
    return F.grid_sample(input, grid.to(input), align_corners=False, mode='nearest')

def get_affine_from_rt(rotation_affine:torch.Tensor, translation_vect:torch.Tensor):
    affine = torch.zeros(3,3)
    affine[:2,:2] = rotation_affine.view(2,2)
    affine[:2,-1] = translation_vect
    affine[2,2] = 1.
    return affine

def get_origin_from_label(label, class_idx) -> torch.Tensor:
    return get_sparse_indices(label.flatten(0,2), class_idx).float().mean(1).flip(0)


def get_grid_sample_coord_from_vox(vox_coord, image_shape):
    assert len(image_shape) == 4
    B,C,H,W = image_shape
    spatial_shape = torch.tensor(image_shape[-2:])
    gs_coord = 2*(vox_coord/spatial_shape)-1.0
    return gs_coord

def get_vox_from_grid_sample_coord(gs_coord, image_shape):
    assert len(image_shape) == 4
    B,C,H,W = image_shape
    spatial_shape = torch.tensor(image_shape[-2:])
    vox_coord = (gs_coord+1.0)/2.0*spatial_shape
        
    return vox_coord


        
seg_image_n = nii2tensor(segment_path_trimmed[3])[3]#segmented slice at frame 3
seg_image_n = seg_image_n[:216,:216].unsqueeze(0).unsqueeze(0)


key_points_list = []

##extraction of keypoints at the intersection of the cardio segments 
#for frame in segment_path_trimmed:
#   segmented_frame = nii2tensor(frame)
#   D, H, W = segmented_frame.shape
    

fig = plt.figure(figsize=[3.,4.])
plt.imshow(seg_image_n.squeeze().T)
plt.show()
plt.close(fig)


    
vox_lv_origin = get_origin_from_label(label=seg_image_n, class_idx=label_dict['Left_Ventricle'])
vox_rv_origin = get_origin_from_label(label=seg_image_n, class_idx=label_dict['Right_Ventricle'])
gs_translation_vect = get_grid_sample_coord_from_vox(vox_lv_origin, seg_image_n.shape)

lv_to_rv_vect = vox_rv_origin - vox_lv_origin
lv_rv_base_angle = torch.atan2(lv_to_rv_vect[1], lv_to_rv_vect[0])/np.pi*180 + 90 

fig, axes = plt.subplots(1,6,figsize=[24.,4.])
for angle_idx, angle in enumerate(torch.linspace(0,300,6)+lv_rv_base_angle):

    rot_mat_2d = rotation_matrix(angle)
    move_lv_affine = get_affine_from_rt(rot_mat_2d, gs_translation_vect).unsqueeze(0).float()
    
    centered_lv_label = spatialtransformer(seg_image_n, move_lv_affine[:,:2,:])

    gs_rv_origin = get_grid_sample_coord_from_vox(vox_rv_origin, seg_image_n.shape)
    gs_rv_origin_moved = move_lv_affine.inverse() @ torch.cat([gs_rv_origin, torch.tensor([1])])
    vox_rv_origin_moved = get_vox_from_grid_sample_coord(gs_rv_origin_moved.view(3)[:2], seg_image_n.shape)
    vox_rv_origin_moved = vox_rv_origin_moved.round().int()
    
    show_label = centered_lv_label.transpose(-2,-1)
    axes[angle_idx].plot(216//2, 216//2, marker='o', color="red")
    axes[angle_idx].plot(vox_rv_origin_moved[1], vox_rv_origin_moved[0], marker='o', color="green")
    axes[angle_idx].grid()
    axes[angle_idx].set_xticks(np.arange(0,217,108))
    axes[angle_idx].set_yticks(np.arange(0,217,108))
    axes[angle_idx].imshow(show_label.squeeze(), interpolation='none')

plt.show()
plt.close(fig)


    


    
    


    
            
    
    
        
    
    
    
    
    

        
    

    
    
    

        