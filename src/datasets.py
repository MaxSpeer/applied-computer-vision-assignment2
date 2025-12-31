import torch
import open3d as o3d
import numpy as np
import os
import fiftyone as fo


def get_torch_xyza(lidar_depth, azimuth, zenith):
        """
        Exact same conversion as in the notebook:
        x = d * sin(-az) * cos(-ze)
        y = d * cos(-az) * cos(-ze)
        z = d * sin(-ze)
        a = 1 if d < 50 else 0
        """
        x = lidar_depth * torch.sin(-azimuth[:, None]) * torch.cos(-zenith[None, :])
        y = lidar_depth * torch.cos(-azimuth[:, None]) * torch.cos(-zenith[None, :])
        z = lidar_depth * torch.sin(-zenith[None, :])
        a = torch.where(
            lidar_depth < 50.0,
            torch.ones_like(lidar_depth),
            torch.zeros_like(lidar_depth),
        )
        xyza = torch.stack((x, y, z, a))   # shape (4, H, W)
        return xyza


def saveToPcdFile(name, xyza, output_dir):
  # xyza: (4, H, W)  -> x,y,z,a
  assert xyza.ndim == 3 and xyza.shape[0] == 4

  # nach NumPy
  xyza_np = xyza.detach().cpu().numpy()  # (4, H, W)
  xyz = xyza_np[:3]          # (3, H, W)
  a   = xyza_np[3]           # (H, W)

  # flatten
  xyz_flat = xyz.reshape(3, -1).T   # (N, 3)
  a_flat   = a.reshape(-1)          # (N,)

  # nur valide Punkte (z.B. a==1, oder depth>0)
  mask = a_flat > 0.5
  points = xyz_flat[mask]           # (N_valid, 3)
  intensity = a_flat[mask]          # (N_valid,)

  # Open3D PointCloud bauen
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)

  # optional: intensity als Grauwertfarbe
  colors = np.stack([intensity, intensity, intensity], axis=1)  # (N_valid,3) in [0,1]
  pcd.colors = o3d.utility.Vector3dVector(colors)

  # speichern
  pcd_path = os.path.join(output_dir, name + ".pcd")

  # Ensure the output directory exists before writing the file
  os.makedirs(output_dir, exist_ok=True) # Create parent directories if they don't exist

  o3d.io.write_point_cloud(pcd_path, pcd)
  print("wrote", pcd_path)


def ids_in_dir(root, ext):
    if not os.path.isdir(root):
        return set()
    return {
        os.path.splitext(f)[0]
        for f in os.listdir(root)
        if f.lower().endswith(ext.lower())
    }

def add_category_to_grouped_dataset_complete_only(
    dataset,
    category_name,      # "cubes" / "spheres"
    class_label,        # "cube" / "sphere"
    rgb_root,
    lidar_root,         # .npy
    pcd_root,           # .pcd
    SAMPLE_LIMIT=1000,
):
    rgb_ids   = ids_in_dir(rgb_root,   ".png")
    lidar_ids = ids_in_dir(lidar_root, ".npy")
    pcd_ids   = ids_in_dir(pcd_root,   ".pcd")

    # ONLY complete IDs (intersection)
    complete_ids = sorted(rgb_ids & lidar_ids & pcd_ids)

    samples = []
    skipped_non_numeric = 0

    count = 0
    for sid in complete_ids:
        if not sid.isnumeric():
            skipped_non_numeric += 1
            continue

        if count >= SAMPLE_LIMIT:
            break
        count += 1

        # Paths (existence should already be implied by ids_in_dir, but keep it robust)
        rgb_path   = os.path.join(rgb_root,   f"{sid}.png")
        lidar_path = os.path.join(lidar_root, f"{sid}.npy")
        pcd_path   = os.path.join(pcd_root,   f"{sid}.pcd")

        if not (os.path.exists(rgb_path) and os.path.exists(lidar_path) and os.path.exists(pcd_path)):
            # extremely rare (race conditions / weird files), but safe to guard
            continue

        group = fo.Group()
        gt = fo.Classification(label=class_label)

        # globally unique id (recommended)
        sample_id = f"{category_name}-{sid}"

        samples.extend([
            fo.Sample(
                filepath=rgb_path,
                group=group.element("rgb"),
                ground_truth=gt,
                modality="rgb",
                # class_label=class_label,
                #category=category_name,
                sample_id=sample_id,
            ),
            fo.Sample(
                filepath=lidar_path,
                group=group.element("lidar"),
                ground_truth=gt,
                modality="lidar",
                #class_label=class_label,
                #category=category_name,
                sample_id=sample_id,
            ),
            fo.Sample(
                filepath=pcd_path,
                group=group.element("pcd"),
                ground_truth=gt,
                modality="lidar_pcd",
                #class_label=class_label,
                #category=category_name,
                sample_id=sample_id,
            ),
        ])

    dataset.add_samples(samples)

    print(
        f"[{category_name}] complete groups added: {len(complete_ids) - skipped_non_numeric} "
        f"(skipped non-numeric ids: {skipped_non_numeric})"
    )

def get_tensor_xyza_from_lidar_dir(lidar_dir, azimuth, zenith):
    lidar_files = [
        os.path.join(lidar_dir, f)
        for f in os.listdir(lidar_dir)
        if f.endswith(".npy")
    ]

    print(f"Found {len(lidar_files)} lidar files.")
    print("First 5 lidar files:")
    for f in lidar_files[:5]:
        print(f)

    xyza_tensors = {}

    print(f"Processing {len(lidar_files)} lidar files...")

    for lidar_file_path in lidar_files:
        # Extract frame ID from filename (e.g., '3241.npy' -> '3241')
        frame_id = os.path.splitext(os.path.basename(lidar_file_path))[0]

        # Load lidar depth data
        lidar_depth_np = np.load(lidar_file_path).astype(np.float32)
        lidar_depth = torch.from_numpy(lidar_depth_np)                # (H, W)

        # Compute xyza tensor
        xyza = get_torch_xyza(lidar_depth, azimuth, zenith)  # (4, H, W)

        # Store in dictionary
        xyza_tensors[frame_id] = xyza
    return xyza_tensors