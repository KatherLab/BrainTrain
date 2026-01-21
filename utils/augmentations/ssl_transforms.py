
#%%
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, RandZoomd, RandRotated, RandFlipd, RandGaussianNoised,
    RandAdjustContrastd, RandBiasFieldd, CopyItemsd, Compose
)
import numpy as np

#%%
def double_view_transform(img_size=96):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),

        # === View 1: zoomed out + rotated ===
        RandZoomd(
            keys=["image"],
            min_zoom=0.8, max_zoom=0.95,  # zoom OUT
            mode="trilinear",
            align_corners=True,
            keep_size=True,
            prob=1.0
        ),
        RandRotated(
            keys=["image"],
            range_x=np.pi/9, range_y=np.pi/18, range_z=np.pi/9,
            mode='bilinear',
            prob=1.0
        ),
        RandBiasFieldd(keys=["image"], prob=0.3),
        RandAdjustContrastd(keys=["image"], gamma=(0.9, 1.1), prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.02),

        # === View 2: zoomed in + flipped ===
        RandZoomd(
            keys=["image_2"],
            min_zoom=1.3, max_zoom=1.5,  # zoom IN
            mode="trilinear",
            align_corners=True,
            keep_size=True,
            prob=1.0
        ),
        RandFlipd(
            keys=["image_2"],
            spatial_axis=[1],  # coronal flip
            prob=1.0
        ),
        RandBiasFieldd(keys=["image_2"], prob=0.3),
        RandAdjustContrastd(keys=["image_2"], gamma=(0.8, 1.3), prob=0.6),
        RandGaussianNoised(keys=["image_2"], prob=0.4, std=0.04),
    ])

#%%
def multi_view_transform(img_size=96):
    """Return 4 independently augmented views for multi-view contrastive learning."""
    postfixes = [f"v{i+1}" for i in range(4)]
    transform_list = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        SpatialPadd(keys=["image"], spatial_size=(img_size, img_size, img_size)),
        RandSpatialCropSamplesd(keys=["image"], roi_size=(img_size, img_size, img_size), num_samples=4),
        SplitDimd(keys=["image"], dim=0, output_postfixes=postfixes)
    ]
    
    for p in postfixes:
        transform_list += [
            RandFlipd(keys=[f"image_{p}"], spatial_axis=[0, 1, 2], prob=0.5),
            RandRotated(keys=[f"image_{p}"], range_x=np.pi/18, range_y=np.pi/18, range_z=np.pi/18, prob=0.5),
            RandCoarseDropoutd(keys=[f"image_{p}"], holes=6, spatial_size=5, dropout_holes=True, prob=0.5),
            RandCoarseShuffled(keys=[f"image_{p}"], holes=10, spatial_size=8, prob=0.5),
            RandGaussianNoised(keys=[f"image_{p}"], prob=0.3),
            RandBiasFieldd(keys=[f"image_{p}"], prob=0.3),
            RandAdjustContrastd(keys=[f"image_{p}"], prob=0.3),
            Rand3DElasticd(keys=[f"image_{p}"], sigma_range=(5, 8), magnitude_range=(1, 2), prob=0.3)
        ]
    
    return Compose(transform_list)
#%%
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged, CropForegroundd,
    SpatialPadd, RandSpatialCropSamplesd, CopyItemsd, RandCoarseDropoutd,
    RandCoarseShuffled, RandFlipd, RandRotated, RandBiasFieldd, RandAdjustContrastd,
    RandGaussianNoised, Rand3DElasticd, OneOf, Compose, SplitDimd, RandScaleIntensityd,
    RandShiftIntensityd, RandGaussianSmoothd, RandRicianNoised, RandKSpaceSpikeNoised,
    RandZoomd, RandAffined, RandSpatialCropd
)
import numpy as np

def strong_ssl_transform(img_size=96):
    """Strong augmentation strategy for SSL pretraining."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        SpatialPadd(keys=["image"], spatial_size=(img_size + 16, img_size + 16, img_size + 16)),
        
        # Create copies before augmentation
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
        
        # === STRONG AUGMENTATIONS FOR VIEW 1 ===
        # Spatial augmentations
        OneOf(transforms=[
            RandSpatialCropd(keys=["image"], roi_size=(img_size, img_size, img_size), random_size=False),
            RandZoomd(keys=["image"], min_zoom=0.8, max_zoom=1.2, prob=1.0),
        ], weights=[0.6, 0.4]),
        
        RandAffined(
            keys=["image"],
            rotate_range=(np.pi/12, np.pi/12, np.pi/12),  # ±15 degrees
            translate_range=(5, 5, 5),
            scale_range=(0.1, 0.1, 0.1),
            prob=0.7
        ),
        
        RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
        
        # Elastic deformation
        Rand3DElasticd(
            keys=["image"],
            sigma_range=(5, 8),
            magnitude_range=(50, 150),
            prob=0.3
        ),
        
        # Intensity augmentations
        #OneOf(transforms=[
            #RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=0.05),
            #RandRicianNoised(keys=["image"], prob=1.0, mean=0.0, std=0.05),
        #], weights=[0.7, 0.3]),
        
        RandBiasFieldd(keys=["image"], prob=0.5, coeff_range=(0.0, 0.5)),
        RandAdjustContrastd(keys=["image"], prob=0.6, gamma=(0.7, 1.3)),
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.4),
        RandShiftIntensityd(keys=["image"], offsets=0.3, prob=0.4),
        
        # Smoothing
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 2.0),
            sigma_y=(0.5, 2.0),
            sigma_z=(0.5, 2.0),
            prob=0.3
        ),
        
        # Structural augmentations
        RandCoarseDropoutd(
            keys=["image"],
            holes=8,
            spatial_size=(8, 8, 8),
            dropout_holes=True,
            fill_value=0,
            prob=0.4
        ),
        
        RandCoarseShuffled(
            keys=["image"],
            holes=5,
            spatial_size=(16, 16, 16),
            prob=0.3
        ),
        
        # === STRONG AUGMENTATIONS FOR VIEW 2 ===
        # Spatial augmentations (different from view 1)
        OneOf(transforms=[
            RandSpatialCropd(keys=["image_2"], roi_size=(img_size, img_size, img_size), random_size=False),
            RandZoomd(keys=["image_2"], min_zoom=0.85, max_zoom=1.15, prob=1.0),
        ], weights=[0.4, 0.6]),
        
        RandAffined(
            keys=["image_2"],
            rotate_range=(np.pi/10, np.pi/10, np.pi/10),  # ±18 degrees
            translate_range=(7, 7, 7),
            scale_range=(0.15, 0.15, 0.15),
            prob=0.8
        ),
        
        RandFlipd(keys=["image_2"], spatial_axis=[0, 1, 2], prob=0.5),
        
        # Different elastic deformation parameters
        Rand3DElasticd(
            keys=["image_2"],
            sigma_range=(4, 7),
            magnitude_range=(40, 120),
            prob=0.4
        ),
        
        # Intensity augmentations (different parameters)
        OneOf(transforms=[
            RandGaussianNoised(keys=["image_2"], prob=1.0, mean=0.0, std=0.04),
            RandRicianNoised(keys=["image_2"], prob=1.0, mean=0.0, std=0.04),
        ], weights=[0.6, 0.4]),
        
        RandBiasFieldd(keys=["image_2"], prob=0.6, coeff_range=(0.0, 0.4)),
        RandAdjustContrastd(keys=["image_2"], prob=0.7, gamma=(0.75, 1.25)),
        RandScaleIntensityd(keys=["image_2"], factors=0.25, prob=0.5),
        RandShiftIntensityd(keys=["image_2"], offsets=0.25, prob=0.5),
        
        # Different smoothing parameters
        RandGaussianSmoothd(
            keys=["image_2"],
            sigma_x=(0.3, 1.5),
            sigma_y=(0.3, 1.5),
            sigma_z=(0.3, 1.5),
            prob=0.4
        ),
        
        # Different structural augmentations
        RandCoarseDropoutd(
            keys=["image_2"],
            holes=6,
            spatial_size=(12, 12, 12),
            dropout_holes=True,
            fill_value=0,
            prob=0.5
        ),
        
        RandCoarseShuffled(
            keys=["image_2"],
            holes=4,
            spatial_size=(20, 20, 20),
            prob=0.4
        ),
    ])

def asymmetric_ssl_transform(img_size=96):
    """Asymmetric augmentation strategy - one strong, one weak view."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        SpatialPadd(keys=["image"], spatial_size=(img_size + 16, img_size + 16, img_size + 16)),
        
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
        
        # === WEAK AUGMENTATION FOR VIEW 1 (Query) ===
        RandSpatialCropd(keys=["image"], roi_size=(img_size, img_size, img_size), random_size=False),
        RandFlipd(keys=["image"], spatial_axis=[0, 1], prob=0.5),
        RandRotated(keys=["image"], range_x=np.pi/36, range_y=np.pi/36, range_z=np.pi/36, prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
        
        # === STRONG AUGMENTATION FOR VIEW 2 (Key) ===
        OneOf(transforms=[
            RandSpatialCropd(keys=["image_2"], roi_size=(img_size, img_size, img_size), random_size=False),
            RandZoomd(keys=["image_2"], min_zoom=0.8, max_zoom=1.2, prob=1.0),
        ], weights=[0.5, 0.5]),
        
        RandAffined(
            keys=["image_2"],
            rotate_range=(np.pi/8, np.pi/8, np.pi/8),
            translate_range=(8, 8, 8),
            scale_range=(0.2, 0.2, 0.2),
            prob=0.8
        ),
        
        RandFlipd(keys=["image_2"], spatial_axis=[0, 1, 2], prob=0.6),
        
        Rand3DElasticd(
            keys=["image_2"],
            sigma_range=(5, 10),
            magnitude_range=(60, 180),
            prob=0.4
        ),
        
        # Strong intensity augmentations for view 2
        RandGaussianNoised(keys=["image_2"], prob=0.6, mean=0.0, std=0.08),
        RandBiasFieldd(keys=["image_2"], prob=0.7, coeff_range=(0.0, 0.6)),
        RandAdjustContrastd(keys=["image_2"], prob=0.8, gamma=(0.6, 1.4)),
        RandScaleIntensityd(keys=["image_2"], factors=0.4, prob=0.6),
        RandShiftIntensityd(keys=["image_2"], offsets=0.4, prob=0.6),
        
        RandCoarseDropoutd(
            keys=["image_2"],
            holes=10,
            spatial_size=(10, 10, 10),
            dropout_holes=True,
            fill_value=0,
            prob=0.6
        ),
    ])

def progressive_ssl_transform(img_size=96, epoch=0, max_epochs=1000):
    """Progressive augmentation that increases in strength over epochs."""
    # Calculate augmentation strength based on epoch
    progress = min(epoch / (max_epochs * 0.3), 1.0)  # Reach max strength at 30% of training
    
    base_prob = 0.3 + 0.4 * progress  # 0.3 -> 0.7
    rotation_range = (np.pi/24) + (np.pi/12) * progress  # 7.5° -> 22.5°
    noise_std = 0.02 + 0.06 * progress  # 0.02 -> 0.08
    
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        #ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        SpatialPadd(keys=["image"], spatial_size=(img_size + 16, img_size + 16, img_size + 16)),
        
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
        
        # Progressive augmentations for both views
        RandSpatialCropd(keys=["image", "image_2"], roi_size=(img_size, img_size, img_size), random_size=False),
        
        RandAffined(
            keys=["image", "image_2"],
            rotate_range=(rotation_range, rotation_range, rotation_range),
            translate_range=(5 * progress, 5 * progress, 5 * progress),
            scale_range=(0.1 * progress, 0.1 * progress, 0.1 * progress),
            prob=base_prob
        ),
        
        RandFlipd(keys=["image", "image_2"], spatial_axis=[0, 1, 2], prob=0.5),
        
        Rand3DElasticd(
            keys=["image", "image_2"],
            sigma_range=(3, 5 + 5 * progress),
            magnitude_range=(30, 50 + 100 * progress),
            prob=base_prob * 0.8
        ),
        
        RandGaussianNoised(keys=["image", "image_2"], prob=base_prob, mean=0.0, std=noise_std),
        RandBiasFieldd(keys=["image", "image_2"], prob=base_prob, coeff_range=(0.0, 0.3 + 0.3 * progress)),
        #RandAdjustContrastd(keys=["image", "image_2"], prob=base_prob, gamma=(0.8 - 0.2 * progress, 1.2 + 0.2 * progress)),
        
        RandCoarseDropoutd(
            keys=["image", "image_2"],
            holes=int(3 + 7 * progress),
            spatial_size=(8, 8, 8),
            dropout_holes=True,
            fill_value=0,
            prob=base_prob * 0.7
        ),
    ])