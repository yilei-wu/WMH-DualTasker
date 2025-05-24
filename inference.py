import argparse
import torch
import numpy as np
import os
import nibabel as nib
from sfcn_rep import SFCN_rep
import torch.nn.functional as F
import scipy.ndimage


def load_model(model_path, model_type='sfcn_rep1', device='cuda'):
    """
    Load trained model weights
    
    Args:
        model_path: Path to the trained model weights (.pth file)
        model_type: Type of model architecture ('sfcn_rep1' or 'sfcn_rep2')
        device: Computing device ('cuda' or 'cpu')
    
    Returns:
        model: Loaded model ready for inference
    """
    if 'sfcn_rep1' in model_type:
        model = SFCN_rep(mode=1)
    elif 'sfcn_rep2' in model_type:
        model = SFCN_rep(mode=2)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'sfcn_rep1' or 'sfcn_rep2'")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def adaptive_preprocessing(flair_image, thres_FLAIR=70):
    """
    Apply the exact same preprocessing used during training
    
    Args:
        flair_image: Input FLAIR volume array
        thres_FLAIR: Threshold for brain mask creation
        
    Returns:
        flair_image: Preprocessed volume
        paddings: Padding information for potential postprocessing
    """
    # Center crop/pad to shape (256, 256, 64)
    x0, x1, x2 = np.shape(flair_image)
    x0_pad = 256 - x0; x1_pad = 256 - x1; x2_pad = 64 - x2
    x0_a = x0_pad//2; x0_b = x0_pad//2 + x0_pad%2
    x1_a = x1_pad//2; x1_b = x1_pad//2 + x1_pad%2
    x2_a = x2_pad//2; x2_b = x2_pad//2 + x2_pad%2
    
    # Padding/cropping for dimension 0
    if x0_pad > 0:
        flair_image = np.pad(flair_image, ((x0_a, x0_b), (0, 0), (0, 0)), mode='constant')
    elif x0_pad == 0:
        pass
    else:
        flair_image = flair_image[-x0_a:x0_b, ...]
   
    # Padding/cropping for dimension 1
    if x1_pad > 0:
        flair_image = np.pad(flair_image, ((0, 0), (x1_a, x1_b), (0, 0)))
    elif x1_pad == 0:
        pass
    else:
        flair_image = flair_image[:, -x1_a:x1_b, :]

    # Padding/cropping for dimension 2
    if x2_pad > 0:
        flair_image = np.pad(flair_image, ((0, 0), (0, 0), (x2_a, x2_b)))
    else:
        flair_image = flair_image[..., -x2_a:x2_b]
        
    # Create brain mask using threshold and morphological operations
    brain_mask_FLAIR = np.empty((256, 256, 64)) 
    brain_mask_FLAIR[flair_image >= thres_FLAIR] = 1 
    brain_mask_FLAIR[flair_image < thres_FLAIR] = 0 
    
    # Fill holes in brain mask for each slice
    for iii in range(np.shape(flair_image)[2]):
        brain_mask_FLAIR[..., iii] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[..., iii])
    
    # Gaussian normalization using brain-masked statistics
    flair_image -= np.mean(flair_image[brain_mask_FLAIR==1])
    flair_image /= np.std(flair_image[brain_mask_FLAIR==1])

    paddings = (x0_a, x0_b, x1_a, x1_b, x2_a, x2_b)
    return flair_image, paddings


def adaptive_postprocessing(pred, paddings):
    """
    Reverse the preprocessing to restore original dimensions
    
    Args:
        pred: Prediction array to be restored
        paddings: Padding information from preprocessing
        
    Returns:
        pred: Restored prediction array
    """
    x0_a, x0_b, x1_a, x1_b, x2_a, x2_b = paddings

    if x0_a > 0:
        pred = pred[x0_a:-x0_b, ...] if x0_b > 0 else pred[x0_a:, ...]
    else:
        pred = np.pad(pred, ((-x0_a, -x0_b), (0, 0), (0, 0)))

    if x1_a > 0:
        pred = pred[:, x1_a:-x1_b, :] if x1_b > 0 else pred[:, x1_a:, :]
    else:
        pred = np.pad(pred, ((0, 0), (-x1_a, -x1_b), (0, 0)))

    if x2_a > 0:
        pred = pred[..., x2_a:-x2_b] if x2_b > 0 else pred[..., x2_a:]
    else:
        pred = np.pad(pred, ((0, 0), (0, 0), (-x2_a, -x2_b)))
    
    return pred


def perform_inference(model, volume, intensity_percentile=99.0, cam_percentile=96.5, device='cuda'):
    """
    Perform inference on a single volume
    
    Args:
        model: Trained model
        volume: Input volume (H, W, D) - should be preprocessed
        intensity_percentile: Intensity threshold percentile for segmentation
        cam_percentile: CAM threshold percentile for segmentation
        device: Computing device
    
    Returns:
        rating: Visual rating score (float)
        segmentation: Binary segmentation mask (numpy array)
        cam: Class activation map (numpy array)
    """
    # Prepare input tensor
    input_tensor = torch.from_numpy(np.expand_dims(volume, [0, 1])).float().to(device)
    
    with torch.no_grad():
        # Get model outputs: rating and CAM
        rating_output, cam_output = model(input_tensor)
        
        # Extract rating score
        rating = float(rating_output.detach().cpu().numpy()[0][0])

        # round rating to integer
        rating = round(rating)
        
        # Process CAM for segmentation
        # Interpolate CAM to original volume size
        cam = F.interpolate(cam_output.cpu(), size=volume.shape, mode='trilinear').numpy()[0, 0, ...]
        
        # Create segmentation mask using thresholds
        cam_binary = np.where(cam > np.percentile(cam.flatten(), cam_percentile), 1, 0)
        
        # Create brain mask from volume (using same threshold as preprocessing)
        brain_mask = np.zeros_like(volume)
        brain_mask[volume > -2] = 1  # Use a liberal threshold since volume is already normalized
        
        # Combine conditions for final segmentation
        intensity_threshold = np.percentile(volume.flatten(), intensity_percentile)
        condition = (cam_binary > 0.8) & (brain_mask > 0) & (volume > intensity_threshold)
        
        segmentation = np.where(condition, 1, 0).astype(np.uint8)
    
    return rating, segmentation, cam


def save_results(rating, segmentation, cam, output_dir, affine=None, paddings=None, restore_original_size=True):
    """
    Save inference results to files
    
    Args:
        rating: Visual rating score
        segmentation: Binary segmentation mask
        cam: Class activation map
        output_dir: Directory to save results
        affine: Affine transformation matrix for NIfTI files
        paddings: Padding information for restoring original size
        restore_original_size: Whether to restore original dimensions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Restore original size if requested and paddings available
    if restore_original_size and paddings is not None:
        print("Restoring segmentation and CAM to original dimensions...")
        segmentation_original = adaptive_postprocessing(segmentation, paddings)
        cam_original = adaptive_postprocessing(cam, paddings)
    else:
        segmentation_original = segmentation
        cam_original = cam
    
    # Save rating to text file
    with open(os.path.join(output_dir, 'visual_rating.txt'), 'w') as f:
        f.write(f"Visual Rating Score: {rating:.4f}\n")
        f.write(f"WMH Volume (voxels): {np.sum(segmentation_original)}\n")
    
    # Use identity matrix if no affine provided
    if affine is None:
        affine = np.eye(4)
    
    # Save segmentation as NIfTI (original size)
    seg_img = nib.Nifti1Image(segmentation_original.astype(np.uint8), affine)
    nib.save(seg_img, os.path.join(output_dir, 'wmh_segmentation.nii.gz'))
    
    # Save CAM as NIfTI (original size)
    cam_img = nib.Nifti1Image(cam_original.astype(np.float32), affine)
    nib.save(cam_img, os.path.join(output_dir, 'class_activation_map.nii.gz'))
    
    
    print(f"All results saved to: {output_dir}")
    print(f"Original segmentation shape: {segmentation_original.shape}")
    print(f"Processed segmentation shape: {segmentation.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="WMH Visual Rating and Segmentation Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input FLAIR volume (.nii/.nii.gz/.npy)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights (.pth file)')
    
    # Optional arguments
    parser.add_argument('--model_type', type=str, default='sfcn_rep1',
                       choices=['sfcn_rep1', 'sfcn_rep2'],
                       help='Model architecture type')
    parser.add_argument('--output_dir', type=str, default='./wmh_output',
                       help='Directory to save outputs')
    parser.add_argument('--intensity_percentile', type=float, default=99.4,
                       help='Intensity percentile threshold for segmentation')
    parser.add_argument('--cam_percentile', type=float, default=96.5,
                       help='CAM percentile threshold for segmentation')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Computing device')
    parser.add_argument('--thres_flair', type=float, default=70,
                       help='FLAIR intensity threshold for brain mask creation')
    parser.add_argument('--keep_original_size', action='store_true',
                       help='Save results in original input dimensions')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load input volume
    print(f"Loading input volume: {args.input}")
    if args.input.endswith(('.nii.gz', '.nii')):
        nii_img = nib.load(args.input)
        volume = nii_img.get_fdata()
        affine = nii_img.affine
    elif args.input.endswith('.npy'):
        volume = np.load(args.input)
        affine = np.eye(4)  # Default affine
    else:
        raise ValueError("Unsupported file format. Use .nii, .nii.gz, or .npy")
    
    print(f"Original volume shape: {volume.shape}")
    print(f"Volume intensity range: [{volume.min():.3f}, {volume.max():.3f}]")
    
    # Apply the same preprocessing used during training
    print("Applying training preprocessing...")
    volume_processed, paddings = adaptive_preprocessing(volume, thres_FLAIR=args.thres_flair)
    
    print(f"Preprocessed volume shape: {volume_processed.shape}")
    print(f"Preprocessed intensity range: [{volume_processed.min():.3f}, {volume_processed.max():.3f}]")
    
    # Load model
    print(f"Loading model: {args.model_type}")
    print(f"Model weights: {args.model_path}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    
    model = load_model(args.model_path, args.model_type, device)
    
    # Perform inference
    print("Performing inference...")
    rating, segmentation, cam = perform_inference(
        model, volume_processed,
        args.intensity_percentile, args.cam_percentile, device
    )
    
    # Print results
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Visual Rating Score: {rating:.4f}")
    print(f"WMH Volume (processed space): {np.sum(segmentation)} voxels")
    
    # Calculate original space volume if restoring size
    if args.keep_original_size:
        segmentation_original = adaptive_postprocessing(segmentation, paddings)
        print(f"WMH Volume (original space): {np.sum(segmentation_original)} voxels")
        wmh_percentage = (np.sum(segmentation_original) / np.prod(segmentation_original.shape)) * 100
        print(f"WMH Volume (% of original): {wmh_percentage:.3f}%")
    else:
        wmh_percentage = (np.sum(segmentation) / np.prod(segmentation.shape)) * 100
        print(f"WMH Volume (% of processed): {wmh_percentage:.3f}%")
    
    print(f"CAM intensity range: [{cam.min():.3f}, {cam.max():.3f}]")
    print("="*50)
    
    # Save results
    save_results(rating, segmentation, cam, args.output_dir, affine, 
                paddings if args.keep_original_size else None, args.keep_original_size)
    
    print(f"\nInference completed successfully!")
    print(f"Preprocessing paddings: {paddings}")


if __name__ == "__main__":
    main()