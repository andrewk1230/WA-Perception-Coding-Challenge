import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, gaussian_filter
from PIL import Image

def detect_traffic_light_color(rgb_patch):
    """
    Detect the color of the traffic light from RGB patch.
    Returns: 'red', 'yellow', 'green', or 'unknown'
    """
    if rgb_patch.size == 0:
        return 'unknown'
    
    # Calculate average color in the patch
    avg_r = np.mean(rgb_patch[..., 0])
    avg_g = np.mean(rgb_patch[..., 1])
    avg_b = np.mean(rgb_patch[..., 2])
    
    # Normalize to 0-1 range if in 0-255 range
    if avg_r > 1.5:  # Likely in 0-255 range
        avg_r, avg_g, avg_b = avg_r / 255.0, avg_g / 255.0, avg_b / 255.0
    
    # Determine color based on dominant channel
    # GREEN: High G, Low R, Low B (CHECK FIRST - VERY SENSITIVE)
    if avg_g > 0.25 and avg_g > avg_r and avg_g > avg_b:
        return 'green'
    # Red: High R, Low G, Low B
    elif avg_r > 0.4 and avg_r > avg_g + 0.1 and avg_r > avg_b + 0.1:
        return 'red'
    # Yellow: High R, High G, Low B (STRICT)
    elif avg_r > 0.35 and avg_g > 0.35 and (avg_r + avg_g) > avg_b + 0.3 and avg_r > avg_b + 0.2 and avg_g > avg_b + 0.2:
        return 'yellow'
    else:
        return 'unknown'

def process_scene():
    # Load the traffic light bounding boxes
    df_light = pd.read_csv('bbox_light.csv').set_index('frame')
    
    # Identify available depth files (in xyz folder)
    xyz_dir = 'xyz'
    if not os.path.isdir(xyz_dir):
        raise ValueError(f"XYZ folder not found. Please ensure 'xyz' directory exists with depth*.npz files.")
    
    depth_files = sorted([f for f in os.listdir(xyz_dir) if f.startswith('depth') and f.endswith('.npz')])
    frames = [int(f.replace('depth', '').replace('.npz', '')) for f in depth_files]
    
    # BUG FIX: Check if depth files exist
    if not depth_files:
        raise ValueError(f"No depth files found in '{xyz_dir}' folder. Ensure depth*.npz files are present.")
    
    rel_light = []
    rel_cart = []
    rel_barrels = []
    rel_barriers = []  # NEW: Track barriers (connected structures)
    rel_pedestrians = []  # NEW: Track pedestrians across frames
    rel_light_colors = []  # NEW: Track traffic light colors

    # NEW: Identify RGB image directory
    rgb_dir = 'rgb'
    if not os.path.isdir(rgb_dir):
        rgb_dir = '.'  # Fallback to current directory
    
    rgb_files = {}
    if os.path.isdir(rgb_dir):
        for f in os.listdir(rgb_dir):
            # Support both .jpg and .jpeg extensions
            if f.startswith('left') and (f.endswith('.jpg') or f.endswith('.jpeg')):
                frame_num = int(f.replace('left', '').replace('.jpg', '').replace('.jpeg', ''))
                rgb_files[frame_num] = os.path.join(rgb_dir, f)

    for f_name, f_idx in zip(depth_files, frames):
        full_data = np.load(os.path.join(xyz_dir, f_name))['xyz']
        # Data is 4-channel: [X, Y, Z, validity]. Use only XYZ
        data = full_data[..., :3]
        
        # 1. Precise Traffic Light Localization using BBox
        light_pos = [np.nan, np.nan, np.nan]
        if f_idx in df_light.index:
            row = df_light.loc[f_idx]
            if row['x1'] > 0 and row['x2'] > 0:
                # Extract the patch from the XYZ data
                patch = data[int(row['y1']):int(row['y2']), int(row['x1']):int(row['x2']), :3]
                # Filter out invalid points (NaN and inf)
                mask = ~(np.isnan(patch[..., 0]) | np.isinf(patch[..., 0]))
                if np.any(mask):
                    # Use median depth of the bounding box pixels
                    light_pos = [np.nanmedian(patch[mask, i]) for i in range(3)]
        rel_light.append(light_pos)

        # NEW: Detect traffic light color from RGB image
        light_color = 'unknown'
        if f_idx in rgb_files:
            try:
                rgb_image = np.array(Image.open(rgb_files[f_idx]))
                if f_idx in df_light.index:
                    row = df_light.loc[f_idx]
                    if row['x1'] > 0 and row['x2'] > 0:
                        # Extract color patch from RGB image
                        color_patch = rgb_image[int(row['y1']):int(row['y2']), int(row['x1']):int(row['x2']), :3]
                        light_color = detect_traffic_light_color(color_patch)
            except Exception as e:
                pass  # Silently fail if image can't be loaded
        rel_light_colors.append(light_color)

        # 2. General Clustering for other objects
        pts = data[::5, ::5, :3].reshape(-1, 3) # Decimate for speed
        # Filter out NaN and inf values
        mask = (~np.isnan(pts[:, 0])) & (~np.isinf(pts[:, 0])) & (pts[:, 0] > 0.5) & (pts[:, 0] < 60)
        pts = pts[mask]
        
        f_cart, f_bars = None, []
        barriers = []  # NEW: Track barriers
        pedestrians = []  # NEW: Track pedestrians
        
        if len(pts) > 10:
            clusters = DBSCAN(eps=0.8, min_samples=5).fit(pts)
            candidate_barrels = []
            candidate_barriers = []  # NEW: Candidate barriers
            candidate_pedestrians = []
            
            for lbl in set(clusters.labels_):
                if lbl == -1: continue
                c = pts[clusters.labels_ == lbl]
                center = np.mean(c, axis=0)
                size = np.max(c, axis=0) - np.min(c, axis=0)
                cluster_size = len(c)
                
                # Golf Cart Logic: Medium size, not the light (height < 2.5m)
                if (size[0] > 1.0 or size[1] > 1.0) and center[2] < 2.2:
                    if f_cart is None or center[0] < f_cart[0]:
                        f_cart = center
                
                # FIRST: Barrel/Traffic Cone Detection (smallest objects first)
                # Traffic cones are small (0.3-0.6m diameter), barrels (0.6-1.5m)
                # Crucially: they have FEW points in sparse depth data
                if (size[0] < 1.5 and size[1] < 1.5 and 
                      center[2] < 1.8 and center[2] > 0.3 and   # Height range for cones/barrels (0.3-1.8m)
                      center[0] < 50 and cluster_size >= 3 and   # VERY LOW - cones have only 3-10 points!
                      # Cylinders: similar width/depth
                      abs(size[0] - size[1]) < 1.0):  # Allow variety (cones aren't perfect cylinders in depth)
                    
                    # Score by confidence - dense clusters preferred
                    confidence = cluster_size / (size[0] * size[1] + 0.1)
                    candidate_barrels.append({
                        'center': center,
                        'size': size,
                        'cluster_size': cluster_size,
                        'confidence': confidence,
                        'distance': center[0]
                    })
                
                # SECOND: Pedestrian Detection (STRICT - NOT BARRELS!)
                # Pedestrians MUST be:
                # - Tall (1.5-2.2m minimum) - taller than barrels
                # - Very narrow (< 0.6m width/depth) - narrower than barrels  
                # - Many points (>= 10) - dense detection, not sparse like barrels
                # - Very upright (H > 2.5*W) - much more vertical than barrels
                elif (center[2] >= 1.5 and center[2] < 2.2 and         # Tall: 1.5m minimum (exclude barrels)
                      size[0] < 0.6 and size[1] < 0.6 and              # Very narrow: human shoulder width
                      center[0] < 50 and cluster_size >= 10 and        # Many points: dense, not sparse barrels
                      # Height MUCH larger than width/depth for upright human
                      center[2] > max(size[0], size[1]) * 2.5):        # Very strict upright check
                    
                    confidence = cluster_size / (size[0] * size[1] + 1e-6)
                    candidate_pedestrians.append({
                        'center': center,
                        'size': size,
                        'cluster_size': cluster_size,
                        'confidence': confidence,
                        'distance': center[0]
                    })
                
                # NEW: Barrier Detection (Connected structures like guardrails/fencing)
                # Barriers are:
                # - Larger than barrels, connected linear structures
                # - Much longer in one dimension (aspect ratio indicates connected)
                # - Height can vary but typically same as barrels
                # - Continuous structures (long stretches)
                elif (center[2] < 1.5 and center[2] > 0.15 and
                      center[0] < 50 and cluster_size >= 8 and        # Need more points (continuous)
                      # Key: Barrier must be MUCH longer in one dimension
                      max(size[0], size[1]) / (min(size[0], size[1]) + 0.1) > 2.2 and  # Stricter from 2.0 to avoid barrels
                      # At least one dimension should be larger than typical barrel
                      max(size[0], size[1]) > 1.2):  # Increased from 1.1
                    
                    # Score barriers by size and continuity
                    confidence = cluster_size / (max(size[0], size[1]) + 0.1)
                    candidate_barriers.append({
                        'center': center,
                        'size': size,
                        'cluster_size': cluster_size,
                        'confidence': confidence,
                        'distance': center[0],
                        'aspect_ratio': max(size[0], size[1]) / (min(size[0], size[1]) + 0.1)
                    })
                
            # Keep only the best 4 barrels (closest and most confident)
            if candidate_barrels:
                # Sort by confidence (descending) then by distance (ascending)
                candidate_barrels.sort(key=lambda x: (-x['confidence'], x['distance']))
                f_bars = [b['center'] for b in candidate_barrels[:4]]
            
            # Keep all detected pedestrians (no limit)
            if candidate_pedestrians:
                candidate_pedestrians.sort(key=lambda x: (-x['confidence'], x['distance']))
                pedestrians = [p['center'] for p in candidate_pedestrians]
            
            # NEW: Keep all detected barriers (no limit, they're important structures)
            if candidate_barriers:
                candidate_barriers.sort(key=lambda x: (-x['confidence'], x['distance']))
                barriers = [b['center'] for b in candidate_barriers]
        
        rel_cart.append(f_cart)
        rel_barrels.append(f_bars)
        rel_barriers.append(barriers)  # NEW: Append barrier detections
        rel_pedestrians.append(pedestrians)  # NEW: Append pedestrian detections

    # --- WORLD TRANSFORM (Origin = Final Known Light Position) ---
    lights = np.array(rel_light)
    # Interpolate light track
    for i in range(3):
        mask = np.isnan(lights[:, i])
        if np.any(mask) and np.any(~mask):
            lights[mask, i] = np.interp(np.where(mask)[0], np.where(~mask)[0], lights[~mask, i])
    
    world_ego = []
    world_cart = []
    all_barrels = []
    all_barriers = []  # NEW: Track barriers in world coordinates
    all_pedestrians = []  # NEW: Track pedestrians in world coordinates

    for i in range(len(frames)):
        lx, ly, _ = lights[i]
        # Ego at (-lx, -ly) relative to light
        ego_pos = np.array([-lx, -ly])
        world_ego.append(ego_pos)
        
        if rel_cart[i] is not None:
            cx, cy, _ = rel_cart[i]
            wx, wy = cx - lx, cy - ly
            # STRICT CONSTRAINT: Cart must be between Ego and Light (Origin)
            # If lx is 30m, and Ego is at -30, cart must be between -30 and 0.
            wx = np.clip(wx, -lx, -1.0) # Always in front of light, behind or at ego
            world_cart.append([wx, wy])
        else:
            world_cart.append([np.nan, np.nan])
            
        for b in rel_barrels[i]:
            bx, by = b[0] - lx, b[1] - ly
            # Barrels should be static and near the light (in front)
            if bx > 0: bx = -0.5
            all_barrels.append([bx, by])
        
        # NEW: Convert barrier detections to world coordinates
        for barrier in rel_barriers[i]:
            px, py = barrier[0] - lx, barrier[1] - ly
            # Barriers should be in front of the light (negative X)
            if px > 0: px = -0.5
            all_barriers.append([px, py])
        
        # NEW: Convert pedestrian detections to world coordinates
        for ped in rel_pedestrians[i]:
            px, py = ped[0] - lx, ped[1] - ly
            # Pedestrians should be in front of the light (negative X)
            if px > 0: px = -1.0
            all_pedestrians.append([px, py, ped[2]])  # Include Z (height) for pedestrians

    # Clean and Smooth Cart Trajectory (S-Curve)
    world_cart = np.array(world_cart)
    world_ego = np.array(world_ego)
    
    # BUG FIX: Handle NaN interpolation safely
    for i in range(2):
        mask = np.isnan(world_cart[:, i])
        # Only interpolate if there's at least one valid point
        if np.any(mask) and np.any(~mask):
            world_cart[mask, i] = np.interp(np.where(mask)[0], np.where(~mask)[0], world_cart[~mask, i])
    
    # BUG FIX: Handle edge case where all ego values might be NaN
    for i in range(2):
        mask = np.isnan(world_ego[:, i])
        if np.any(mask) and np.any(~mask):
            world_ego[mask, i] = np.interp(np.where(mask)[0], np.where(~mask)[0], world_ego[~mask, i])
    
    # SMOOTHING: Apply temporal median filtering for noise reduction
    window_size = 9  # Smooth over 9 frames (increased from 5 for more smoothing)
    world_cart[:, 0] = median_filter(world_cart[:, 0], size=window_size, mode='nearest')
    world_cart[:, 1] = median_filter(world_cart[:, 1], size=window_size, mode='nearest')
    world_ego[:, 0] = median_filter(world_ego[:, 0], size=window_size, mode='nearest')
    world_ego[:, 1] = median_filter(world_ego[:, 1], size=window_size, mode='nearest')
    
    # Additional smoothing: Apply Gaussian filter for extra smoothness
    world_cart[:, 0] = gaussian_filter(world_cart[:, 0], sigma=1.5)
    world_cart[:, 1] = gaussian_filter(world_cart[:, 1], sigma=1.5)
    world_ego[:, 0] = gaussian_filter(world_ego[:, 0], sigma=1.5)
    world_ego[:, 1] = gaussian_filter(world_ego[:, 1], sigma=1.5)
    
    t = np.linspace(0, 1, len(frames))
    t_new = np.linspace(0, 1, 100)
    
    # BUG FIX: Use linear interpolation if fewer than 4 points (cubic needs 4+ points)
    cart_kind = 'cubic' if len(frames) >= 4 else 'linear'
    ego_kind = 'linear'  # Linear is always safe and appropriate for ego
    
    fx = interp1d(t, world_cart[:, 0], kind=cart_kind, fill_value='extrapolate')
    fy = interp1d(t, world_cart[:, 1], kind=cart_kind, fill_value='extrapolate')
    smooth_cart = np.column_stack([fx(t_new), fy(t_new)])
    
    ex = interp1d(t, world_ego[:, 0], kind=ego_kind, fill_value='extrapolate')
    ey = interp1d(t, world_ego[:, 1], kind=ego_kind, fill_value='extrapolate')
    smooth_ego = np.column_stack([ex(t_new), ey(t_new)])

    # Consolidate Barrels across frames with improved filtering
    # Only show barrels that are consistently detected (≥6 detections)
    static_barrels = []
    if all_barrels:
        # Aggregate all detected barrel positions across frames
        all_barrels_array = np.array(all_barrels)
        
        # Cluster barrels across time (group detections of same barrel)
        # Use tighter clustering since barrels are static
        cl = DBSCAN(eps=1.0, min_samples=3).fit(all_barrels_array)
        
        barrel_clusters = []
        for l in set(cl.labels_):
            if l == -1: continue
            barrel_points = all_barrels_array[cl.labels_ == l]
            
            # Calculate median position (robust to outliers)
            median_pos = np.median(barrel_points, axis=0)
            # Calculate detection count (confidence metric)
            detection_count = len(barrel_points)
            
            barrel_clusters.append({
                'pos': median_pos,
                'detections': detection_count,
                'std': np.std(barrel_points, axis=0)
            })
        
        # Keep only the best 4 barrels (most frequently detected)
        # Require at least 2 detections to show a barrel (for sparse traffic cones)
        barrel_clusters.sort(key=lambda x: -x['detections'])
        static_barrels = [b['pos'] for b in barrel_clusters 
                         if b['detections'] >= 2][:4]  # Min 2 detections + max 4 barrels
        
        # Print barrel detection stats
        if len(static_barrels) > 0:
            print(f"Detected {len(static_barrels)} static barrels (with ≥2 detections):")
            for i, b in enumerate(static_barrels):
                print(f"  Barrel {i+1}: position=({b[0]:.2f}, {b[1]:.2f})")
        else:
            print(f"No barrels with sufficient consistent detections (need ≥3, found {max([b['detections'] for b in barrel_clusters], default=0)})")

    # NEW: Consolidate Barriers across frames
    static_barriers = []
    if all_barriers:
        # Aggregate all detected barrier positions across frames
        all_barriers_array = np.array(all_barriers)
        
        # Cluster barriers across time (group detections of same barrier)
        # Use looser clustering since barriers can be large and slightly mobile
        cl = DBSCAN(eps=1.5, min_samples=2).fit(all_barriers_array)
        
        barrier_clusters = []
        for l in set(cl.labels_):
            if l == -1: continue
            barrier_points = all_barriers_array[cl.labels_ == l]
            
            # Calculate median position (robust to outliers)
            median_pos = np.median(barrier_points, axis=0)
            # Calculate detection count (confidence metric)
            detection_count = len(barrier_points)
            
            barrier_clusters.append({
                'pos': median_pos,
                'detections': detection_count,
                'std': np.std(barrier_points, axis=0)
            })
        
        # Keep all detected barriers (they're important infrastructure)
        # Require at least 3 detections to show a barrier
        barrier_clusters.sort(key=lambda x: -x['detections'])
        static_barriers = [b['pos'] for b in barrier_clusters 
                          if b['detections'] >= 3]
        
        # Print barrier detection stats
        if len(static_barriers) > 0:
            print(f"\nDetected {len(static_barriers)} barriers (with ≥3 detections):")
            for i, b in enumerate(static_barriers):
                print(f"  Barrier {i+1}: position=({b[0]:.2f}, {b[1]:.2f})")

    # NEW: Consolidate Pedestrians across frames (LENIENT FOR MOVEMENT)
    static_pedestrians = []
    # NEW: Consolidate Pedestrians as DYNAMIC TRAJECTORIES (not static points)
    pedestrian_trajectories = []  # List of trajectories, one per pedestrian
    if all_pedestrians:
        all_pedestrians_array = np.array(all_pedestrians)
        
        # Cluster pedestrians with LARGE radius to account for continuous movement
        # Pedestrians move every pixel, so need large eps to group same person across frames
        # With max 2 pedestrians, use large clustering radius
        cl = DBSCAN(eps=5.0, min_samples=2).fit(all_pedestrians_array[:, :2])  # Large eps for movement
        
        pedestrian_clusters = []
        for l in set(cl.labels_):
            if l == -1: continue
            ped_points = all_pedestrians_array[cl.labels_ == l]
            
            # Any cluster with 2+ detections is a valid pedestrian
            if len(ped_points) < 2:
                continue
            
            # Sort by X coordinate (temporal order - they move in X direction)
            ped_sorted = ped_points[np.argsort(ped_points[:, 0])]
            
            # Calculate detection count (confidence metric)
            detection_count = len(ped_points)
            
            pedestrian_clusters.append({
                'trajectory': ped_sorted,  # Full trajectory points
                'detections': detection_count,
                'start_pos': ped_sorted[0],
                'end_pos': ped_sorted[-1]
            })
        
        # Keep top 2 pedestrians (max possible) sorted by frequency detected
        pedestrian_clusters.sort(key=lambda x: -x['detections'])
        pedestrian_trajectories = [p for p in pedestrian_clusters[:2]]  # Max 2 pedestrians
        
        # Print pedestrian detection stats
        if len(pedestrian_trajectories) > 0:
            print(f"\nDetected {len(pedestrian_trajectories)} pedestrian(s) with dynamic trajectories:")
            for i, p in enumerate(pedestrian_trajectories):
                start = p['start_pos']
                end = p['end_pos']
                print(f"  Pedestrian {i+1}: {p['detections']} detections")
                print(f"    Start: ({start[0]:.2f}, {start[1]:.2f})")
                print(f"    End: ({end[0]:.2f}, {end[1]:.2f})")

    # NEW: Determine most common traffic light color
    traffic_light_color = 'unknown'
    if rel_light_colors:
        color_counts = {}
        for color in rel_light_colors:
            if color != 'unknown':
                color_counts[color] = color_counts.get(color, 0) + 1
        
        if color_counts:
            traffic_light_color = max(color_counts, key=color_counts.get)
            print(f"\nTraffic Light Color: {traffic_light_color.upper()}")
            for color, count in sorted(color_counts.items(), key=lambda x: -x[1]):
                print(f"  {color}: {count} frames")

    return smooth_ego, smooth_cart, static_barrels, static_barriers, pedestrian_trajectories, traffic_light_color, rel_light_colors

ego, cart, barrels, barriers, pedestrians, light_color, light_colors_per_frame = process_scene()

# Final Visualization
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
ax.set_facecolor('white')

# NEW: Create dynamic traffic light marker (will change color in animation)
light_marker, = ax.plot([], [], '*', color='red', markersize=30, label='light', 
                        zorder=5, markeredgecolor='darkred', markeredgewidth=0.5)

# Add traffic light color annotation (will update in animation)
color_text_box = ax.text(2, 3, "Light: UNKNOWN", fontsize=10, color='red', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red', linewidth=1.5))

# Plot barrels with clean style
if barrels:
    print(f"\n✓ Displaying {len(barrels)} barrels in visualization:")
    for i, b in enumerate(barrels):
        # Draw barrel with simple circle marker
        ax.scatter(b[0], b[1], color='orange', marker='o', s=150, zorder=4, 
                  edgecolors='darkorange', linewidth=1, alpha=0.9)
        print(f"  Barrel {i+1}: position=({b[0]:.2f}, {b[1]:.2f})")
    # Add single barrel label to legend
    ax.scatter([], [], color='orange', marker='o', s=150, label='barrel', 
              edgecolors='darkorange', linewidth=1)
else:
    print("\n⚠ No barrels detected in scene")

# NEW: Plot barriers with clean style
if barriers:
    print(f"\n✓ Displaying {len(barriers)} barrier(s) in visualization:")
    for i, b in enumerate(barriers):
        # Draw barrier with simple square marker
        ax.scatter(b[0], b[1], color='brown', marker='s', s=150, zorder=4, 
                  edgecolors='maroon', linewidth=1, alpha=0.9)
        print(f"  Barrier {i+1}: position=({b[0]:.2f}, {b[1]:.2f})")
    # Add single barrier label to legend
    ax.scatter([], [], color='brown', marker='s', s=150, label='barrier', 
              edgecolors='maroon', linewidth=1)
else:
    print("\n⚠ No barriers detected in scene")

# NEW: Plot pedestrian trajectories with clean style
if pedestrians:
    print(f"\n✓ Displaying {len(pedestrians)} pedestrian trajectory/trajectories in visualization:")
    for i, ped_traj in enumerate(pedestrians):
        # Get trajectory points
        traj = ped_traj['trajectory'][:, :2]  # X, Y only
        color = ['red', 'orange'][min(i, 1)]
        
        # Plot trajectory path
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, alpha=0.6, zorder=3)
        
        # Plot start and end points
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker='o', s=100, zorder=4, 
                  label=f'pedestrian {i+1} start', edgecolors='black', linewidth=1)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker='X', s=150, zorder=4, 
                  label=f'pedestrian {i+1} end', edgecolors='black', linewidth=1)
        
        print(f"  Pedestrian {i+1}: {len(traj)} trajectory points, {ped_traj['detections']} detections")
else:
    print("\n⚠ No pedestrians detected in scene")

# Plot ego and cart with clean style
e_p, = ax.plot([], [], 'o', color='blue', markersize=8, zorder=6)
c_p, = ax.plot([], [], 's', color='purple', markersize=8, zorder=6)
e_t, = ax.plot([], [], color='blue', linewidth=2.5, alpha=0.6, zorder=2)
c_t, = ax.plot([], [], '--', color='purple', linewidth=2.5, alpha=0.6, zorder=2)

# Add to legend
ax.plot([], [], 'o', color='blue', markersize=8, label='Ego')
ax.plot([], [], 's', color='purple', markersize=8, label='cart')
ax.plot([], [], color='blue', linewidth=2.5, label='Ego trajectory')
ax.plot([], [], '--', color='purple', linewidth=2.5, label='cart trajectory')

# Clean axis labels
ax.set_xlabel('X (forward distance, m)', fontsize=11, fontweight='normal')
ax.set_ylabel('Y (lateral, m)', fontsize=11, fontweight='normal')
ax.set_xlim(np.min(ego[:,0])-5, 5)
ax.set_ylim(np.min(ego[:,1])-20, np.max(ego[:,1])+20)

# Clean legend - positioned in upper right
ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='black', fancybox=False)

# NEW: Interpolate pedestrian trajectories to 100-frame animation (MUST BE BEFORE MARKERS)
pedestrian_smooth = []
if pedestrians and len(pedestrians) > 0:
    for ped in pedestrians:
        # Each pedestrian has a trajectory of detected positions
        ped_array = ped['trajectory'][:, :2]  # Use only X, Y
        
        if len(ped_array) > 1:
            # Interpolate trajectory to 100 frames
            t_ped = np.linspace(0, 1, len(ped_array))
            t_smooth = np.linspace(0, 1, 100)
            
            px = interp1d(t_ped, ped_array[:, 0], kind='linear', fill_value='extrapolate')
            py = interp1d(t_ped, ped_array[:, 1], kind='linear', fill_value='extrapolate')
            
            smooth_trajectory = np.column_stack([px(t_smooth), py(t_smooth)])
            pedestrian_smooth.append(smooth_trajectory)
        else:
            # Single point - repeat for all frames
            pedestrian_smooth.append(np.tile(ped_array[0], (100, 1)))
else:
    pedestrian_smooth = []

# NOW: Create dynamic pedestrian markers and trails (after pedestrian_smooth is defined)
pedestrian_markers = []
pedestrian_trails = []
pedestrian_colors = ['red', 'orange']  # Colors for up to 2 pedestrians

for j in range(min(2, len(pedestrian_smooth))):
    p_marker, = ax.plot([], [], 'o', color=pedestrian_colors[j], markersize=8, zorder=6)
    p_trail, = ax.plot([], [], '-', color=pedestrian_colors[j], linewidth=2.5, alpha=0.5, zorder=2)
    pedestrian_markers.append(p_marker)
    pedestrian_trails.append(p_trail)

# Update legend with pedestrian entries
for j in range(min(2, len(pedestrian_smooth))):
    ax.plot([], [], 'o', color=pedestrian_colors[j], markersize=8, label=f'pedestrian {j+1}')
    ax.plot([], [], color=pedestrian_colors[j], linewidth=2.5, label=f'pedestrian {j+1} trail')

# Refresh legend
ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='black', fancybox=False)

# NEW: Interpolate light colors to match 100-frame animation
# Map original frame colors to smooth 100-frame timeline
light_colors_smooth = []
if light_colors_per_frame:
    # Create interpolation for frame indices
    original_indices = np.linspace(0, len(light_colors_per_frame)-1, len(light_colors_per_frame))
    smooth_indices = np.linspace(0, len(light_colors_per_frame)-1, 100)
    
    for smooth_idx in smooth_indices:
        # Find nearest original frame
        nearest_idx = int(np.round(smooth_idx))
        nearest_idx = min(nearest_idx, len(light_colors_per_frame)-1)
        light_colors_smooth.append(light_colors_per_frame[nearest_idx])
else:
    light_colors_smooth = ['unknown'] * 100

def get_light_color_display(color):
    """Convert color name to display color and text"""
    color_map = {
        'red': ('red', 'RED'),
        'yellow': ('gold', 'YELLOW'),
        'green': ('lime', 'GREEN'),
        'unknown': ('gray', 'UNKNOWN')
    }
    return color_map.get(color, ('gray', 'UNKNOWN'))

# BUG FIX: Include current position in trail (i+1 instead of i)
def anim(i):
    e_p.set_data([ego[i, 0]], [ego[i, 1]])
    c_p.set_data([cart[i, 0]], [cart[i, 1]])
    e_t.set_data(ego[:i+1, 0], ego[:i+1, 1])
    c_t.set_data(cart[:i+1, 0], cart[:i+1, 1])
    
    # NEW: Update pedestrian positions and trails dynamically
    updated_objects = [e_p, c_p, e_t, c_t]
    
    for j, (p_marker, p_trail) in enumerate(zip(pedestrian_markers, pedestrian_trails)):
        if j < len(pedestrian_smooth):
            # Update pedestrian position
            p_marker.set_data([pedestrian_smooth[j][i, 0]], [pedestrian_smooth[j][i, 1]])
            # Update pedestrian trail (showing path so far)
            p_trail.set_data(pedestrian_smooth[j][:i+1, 0], pedestrian_smooth[j][:i+1, 1])
            updated_objects.extend([p_marker, p_trail])
    
    # NEW: Update traffic light color dynamically
    light_color_current = light_colors_smooth[i] if i < len(light_colors_smooth) else 'unknown'
    marker_color, color_text = get_light_color_display(light_color_current)
    
    # Update star position and color
    light_marker.set_data([0], [0])
    light_marker.set_color(marker_color)
    light_marker.set_markeredgecolor('darkred')
    
    # Update color annotation text
    color_text_box.set_text(f"Light: {color_text}")
    color_text_box.set_color(marker_color)
    color_text_box.get_bbox_patch().set_edgecolor(marker_color)
    
    updated_objects.extend([light_marker, color_text_box])
    return updated_objects

ani = animation.FuncAnimation(fig, anim, frames=100, interval=50, blit=True)

# Save as both GIF (for quick preview) and MP4 (for better quality)
print("\nSaving animations...")
ani.save('trajectory.gif', writer='pillow')
print("trajectory.gif saved")

# Save as MP4 (requires ffmpeg)
try:
    ani.save('trajectory.mp4', writer='ffmpeg', fps=20, dpi=100)
    print("trajectory.mp4 saved (MP4 - higher quality)")
except Exception as e:
    print(f"MP4 save failed (ffmpeg may not be installed): {e}")
    print("Install ffmpeg: sudo apt-get install ffmpeg")

plt.savefig('trajectory.png')
print("trajectory.png saved\n")