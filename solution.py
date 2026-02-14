import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import re
from scipy.signal import savgol_filter

def smooth_trajectory(data_list, window=15, polyorder=2):
    """Cleans noise from X,Y coordinates using Savitzky-Golay."""
    df = pd.DataFrame(data_list, columns=['x', 'y'])
    df = df.interpolate(method='linear', limit_direction='both').ffill().bfill()
    
    if len(df) < window:
        window = len(df) if len(df) % 2 != 0 else len(df) - 1
        
    if window > polyorder and window > 3:
        smoothed_x = savgol_filter(df['x'].values, window, polyorder)
        smoothed_y = savgol_filter(df['y'].values, window, polyorder)
    else:
        smoothed_x, smoothed_y = df['x'].values, df['y'].values
        
    return np.column_stack((smoothed_x, smoothed_y))

def process_data():
    """Extracts coordinates for both World View and Ego-Centric View."""
    CSV_FILE = 'bbox_light.csv'
    XYZ_FOLDER = 'xyz' 
    
    if not os.path.exists(XYZ_FOLDER):
        print(f"Error: Folder '{XYZ_FOLDER}' not found.")
        return None
    
    npz_files = [f for f in os.listdir(XYZ_FOLDER) if f.endswith('.npz')]
    npz_files.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    df_csv = pd.read_csv(CSV_FILE)
    
    raw_light_cam = [] 
    raw_cart_cam = []

    print(f"Processing {len(npz_files)} frames...")
    
    for filename in npz_files:
        frame_num = int(re.sub(r'\D', '', filename))
        row = df_csv[df_csv['frame'] == frame_num]
        
        if row.empty:
            raw_light_cam.append([np.nan, np.nan, np.nan])
            raw_cart_cam.append([np.nan, np.nan, np.nan])
            continue
        
        row = row.iloc[0]
        data_load = np.load(os.path.join(XYZ_FOLDER, filename))
        key = 'xyz' if 'xyz' in data_load.files else data_load.files[0]
        data = data_load[key].copy()
        data[np.isinf(data)] = np.nan 

        # --- PART A: Traffic Light (Relative) ---
        p_light = [np.nan, np.nan, np.nan]
        if row['x1'] != 0:
            patch = data[int(row['y1']):int(row['y2']), int(row['x1']):int(row['x2']), :3]
            if np.any(~np.isnan(patch)):
                p_light = np.nanmedian(patch, axis=(0,1))
        raw_light_cam.append(p_light)

        # --- PART B: Golf Cart (Detection with Leading Edge) ---
        x, y, z = data[:,:,0], data[:,:,1], data[:,:,2]
        cart_mask = (x > 3.0) & (x < 45.0) & (y > -7.0) & (y < 7.0) & (z > 0.1) & (z < 1.4)
        cart_pts = data[cart_mask][:, :3]
        
        if len(cart_pts) > 100:
            raw_cart_cam.append([np.nanpercentile(cart_pts[:,0], 20), np.nanmedian(cart_pts[:,1])])
        else:
            raw_cart_cam.append([np.nan, np.nan])

    # --- COORDINATE TRANSFORMATIONS ---
    light_interp = pd.DataFrame(raw_light_cam).interpolate().ffill().bfill().values
    v0 = light_interp[0]
    theta0 = np.arctan2(v0[1], v0[0])
    c, s = np.cos(-theta0), np.sin(-theta0)
    R = np.array([[c, -s], [s, c]])

    world_ego_raw, world_cart_raw = [], []
    for i in range(len(light_interp)):
        L, C = light_interp[i][:2], np.array(raw_cart_cam[i][:2])
        world_ego_raw.append(-(R @ L))
        world_cart_raw.append(R @ (C - L) if not np.isnan(C).any() else [np.nan, np.nan])

    # Smoothing
    world_ego = smooth_trajectory(world_ego_raw)
    world_cart = smooth_trajectory(world_cart_raw)
    rel_light = smooth_trajectory([lc[:2] for lc in raw_light_cam])
    rel_cart = smooth_trajectory(raw_cart_cam)

    return world_ego, world_cart, rel_light, rel_cart

def save_world_animation(world_ego, world_cart):
    """World-View Animation (Fixed Origin at Light)"""
    fig, ax = plt.subplots(figsize=(8, 8))
    # SET AXIS LIMITS AS REQUESTED
    ax.set_ylim(-20, 20)
    ax.set_xlim(-50, 30)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.scatter(0, 0, c='gold', marker='*', s=200, label='Traffic Light (Origin)', zorder=5)
    ego_l, = ax.plot([], [], 'b-', alpha=0.3)
    ego_p, = ax.plot([], [], 'bo', label='Ego Car')
    cart_l, = ax.plot([], [], 'r--', alpha=0.3)
    cart_p, = ax.plot([], [], 'rs', label='Golf Cart')
    
    ax.legend(loc='upper right')
    ax.set_title("World Frame Trajectory")

    def update(f):
        ego_l.set_data(world_ego[:f, 0], world_ego[:f, 1])
        ego_p.set_data([world_ego[f, 0]], [world_ego[f, 1]])
        valid_c = world_cart[:f][~np.isnan(world_cart[:f, 0])]
        if len(valid_c) > 0: cart_l.set_data(valid_c[:, 0], valid_c[:, 1])
        if not np.isnan(world_cart[f, 0]): cart_p.set_data([world_cart[f, 0]], [world_cart[f, 1]])
        return ego_l, ego_p, cart_l, cart_p

    ani = animation.FuncAnimation(fig, update, frames=len(world_ego), interval=50, blit=True)
    
    # Save both GIF and MP4
    ani.save('trajectory.gif', writer='pillow', fps=20)
    try:
        ani.save('trajectory.mp4', writer='ffmpeg', fps=20)
        print("Saved trajectory.mp4 and .gif")
    except:
        print("Saved trajectory.gif")
    plt.close()

def save_ego_view_gif(light_rel, cart_rel):
    """Ego-Centric POV Animation with Tracing Lines"""
    fig, ax = plt.subplots(figsize=(8, 8))
    # SET AXIS LIMITS AS REQUESTED
    ax.set_xlim(-5, 50) 
    ax.set_ylim(-15, 30)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Static Ego
    ax.plot([0], [0], 'bo', markersize=12, label='Ego (Fixed Center)', zorder=5)
    
    # Tracing lines (historical paths)
    light_trace, = ax.plot([], [], 'g-', alpha=0.3, linewidth=1)
    cart_trace, = ax.plot([], [], 'r--', alpha=0.3, linewidth=1)
    
    # Current positions
    l_p, = ax.plot([], [], 'g*', markersize=15, label='Light (Closing In)')
    c_p, = ax.plot([], [], 'rs', label='Golf Cart')
    
    ax.set_title("Ego-Centric POV (Tracing Enabled)")
    ax.set_xlabel("Meters Ahead")
    ax.set_ylabel("Meters Lateral")
    ax.legend(loc='upper right')

    def update(i):
        # Update Traces
        light_trace.set_data(light_rel[:i, 0], light_rel[:i, 1])
        cart_trace.set_data(cart_rel[:i, 0], cart_rel[:i, 1])
        
        # Update Points
        l_p.set_data([light_rel[i,0]], [light_rel[i,1]])
        c_p.set_data([cart_rel[i,0]], [cart_rel[i,1]])
        return light_trace, cart_trace, l_p, c_p

    ani = animation.FuncAnimation(fig, update, frames=len(light_rel), interval=50, blit=True)
    ani.save('ego_view.gif', writer='pillow', fps=20)
    plt.close()

if __name__ == "__main__":
    results = process_data()
    if results:
        w_ego, w_cart, r_light, r_cart = results
        save_world_animation(w_ego, w_cart)
        save_ego_view_gif(r_light, r_cart)
        print("All animations complete.")