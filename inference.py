import argparse
import torch
import numpy as np
import os
import nibabel as nib
from model.sfcn_rep import SFCN_rep
import torch.nn.functional as F


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


def preprocess_volume(volume, target_shape=(256, 256, 64)):
    """
    Preprocess input volume to match expected input shape
    
    Args:
        volume: Input volume array
        target_shape: Expected input shape for the model
        
    Returns:
        volume: Preprocessed volume
    """
    # Ensure the volume is the right shape
    if volume.shape != target_shape:
        print(f"Resizing volume from {volume.shape} to {target_shape}")
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
        volume = F.interpolate(
            volume_tensor,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        ).squeeze().numpy()
    
    return volume


def normalize_volume(volume):
    """
    Basic normalization of the volume
    You may need to adjust this based on your training preprocessing
    """
    # Clip extreme values (optional)
    volume = np.clip(volume, np.percentile(volume, 1), np.percentile(volume, 99))
    
    # Z-score normalization
    mean_val = np.mean(volume)
    std_val = np.std(volume)
    if std_val > 0:
        volume = (volume - mean_val) / std_val
    
    return volume


def perform_inference(model, volume, brain_mask=None, intensity_percentile=99.4, cam_percentile=96.5, device='cuda'):
    """
    Perform inference on a single volume
    
    Args:
        model: Trained model
        volume: Input volume (H, W, D)
        brain_mask: Optional brain mask for constraining segmentation
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
        
        # Process CAM for segmentation
        # Interpolate CAM to original volume size
        cam = F.interpolate(cam_output.cpu(), size=volume.shape, mode='trilinear').numpy()[0, 0, ...]
        
        # Create segmentation mask using thresholds
        cam_binary = np.where(cam > np.percentile(cam.flatten(), cam_percentile), 1, 0)
        intensity_threshold = np.percentile(volume.flatten(), intensity_percentile)
        
        # Combine conditions for final segmentation
        if brain_mask is not None:
            # Use brain mask to constrain segmentation
            condition = (cam_binary > 0.5) & (brain_mask > 0) & (volume > intensity_threshold)
        else:
            condition = (cam_binary > 0.5) & (volume > intensity_threshold)
        
        segmentation = np.where(condition, 1, 0).astype(np.uint8)
    
    return rating, segmentation, cam


def save_results(rating, segmentation, cam, output_dir, affine=None):
    """
    Save inference results to files
    
    Args:
        rating: Visual rating score
        segmentation: Binary segmentation mask
        cam: Class activation map
        output_dir: Directory to save results
        affine: Affine transformation matrix for NIfTI files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save rating to text file
    with open(os.path.join(output_dir, 'visual_rating.txt'), 'w') as f:
        f.write(f"Visual Rating Score: {rating:.4f}\n")
        f.write(f"WMH Volume (voxels): {np.sum(segmentation)}\n")
    
    # Use identity matrix if no affine provided
    if affine is None:
        affine = np.eye(4)
    
    # Save segmentation as NIfTI
    seg_img = nib.Nifti1Image(segmentation.astype(np.uint8), affine)
    nib.save(seg_img, os.path.join(output_dir, 'wmh_segmentation.nii.gz'))
    
    # Save CAM as NIfTI
    cam_img = nib.Nifti1Image(cam.astype(np.float32), affine)
    nib.save(cam_img, os.path.join(output_dir, 'class_activation_map.nii.gz'))
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, 'visual_rating.npy'), rating)
    np.save(os.path.join(output_dir, 'wmh_segmentation.npy'), segmentation)
    np.save(os.path.join(output_dir, 'class_activation_map.npy'), cam)
    
    print(f"All results saved to: {output_dir}")


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
    parser.add_argument('--brain_mask', type=str, default=None,
                       help='Path to brain mask (.nii/.nii.gz/.npy) - optional but recommended')
    parser.add_argument('--output_dir', type=str, default='./wmh_output',
                       help='Directory to save outputs')
    parser.add_argument('--intensity_percentile', type=float, default=99.4,
                       help='Intensity percentile threshold for segmentation')
    parser.add_argument('--cam_percentile', type=float, default=96.5,
                       help='CAM percentile threshold for segmentation')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Computing device')
    parser.add_argument('--normalize', action='store_true',
                       help='Apply basic normalization to input volume')
    
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
    
    # Load brain mask if provided
    brain_mask = None
    if args.brain_mask:
        print(f"Loading brain mask: {args.brain_mask}")
        if args.brain_mask.endswith(('.nii.gz', '.nii')):
            brain_mask = nib.load(args.brain_mask).get_fdata()
        elif args.brain_mask.endswith('.npy'):
            brain_mask = np.load(args.brain_mask)
        brain_mask = preprocess_volume(brain_mask)
    
    # Preprocess volume
    volume = preprocess_volume(volume)
    if args.normalize:
        print("Applying normalization...")
        volume = normalize_volume(volume)
    
    print(f"Preprocessed volume shape: {volume.shape}")
    
    # Load model
    print(f"Loading model: {args.model_type}")
    print(f"Model weights: {args.model_path}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    
    model = load_model(args.model_path, args.model_type, device)
    
    # Perform inference
    print("Performing inference...")
    rating, segmentation, cam = perform_inference(
        model, volume, brain_mask,
        args.intensity_percentile, args.cam_percentile, device
    )
    
    # Print results
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Visual Rating Score: {rating:.4f}")
    print(f"WMH Volume (voxels): {np.sum(segmentation)}")
    if np.sum(segmentation) > 0:
        wmh_percentage = (np.sum(segmentation) / np.prod(segmentation.shape)) * 100
        print(f"WMH Volume (% of total): {wmh_percentage:.3f}%")
    print(f"CAM intensity range: [{cam.min():.3f}, {cam.max():.3f}]")
    print("="*50)
    
    # Save results
    save_results(rating, segmentation, cam, args.output_dir, affine)
    
    print(f"\nInference completed successfully!")


if __name__ == "__main__":
    main()