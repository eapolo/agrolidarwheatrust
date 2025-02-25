import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def fit_plane(points):
    """
    Fit a plane to the given set of 3D points using PCA.
    Returns:
      - normal (ndarray): the plane's normal vector
      - d (float): offset in the plane equation ax + by + cz + d = 0
      - ground_level (float): mean Z value of the fitted plane
    """
    if len(points) < 3:
        raise ValueError("Not enough points to fit a plane")
    pca = PCA(n_components=3)
    pca.fit(points)
    normal = pca.components_[-1]
    point_on_plane = np.mean(points, axis=0)
    d = -np.dot(normal, point_on_plane)
    return normal, d, point_on_plane[2]  # We'll use the plane's mean Z as ground_level

def load_txt_point_cloud(file_path):
    """
    Load a point cloud from a TXT file.
    Assumes columns are X, Y, Z, and Intensity (at least 4 columns).
    Skips the first row if it's a header.
    """
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Skip header row
    # Extract X, Y, Z, Intensity (first 4 columns)
    points = data[:, :3]       # X, Y, Z
    intensities = data[:, 3]   # Intensity
    return points, intensities

def calculate_plant_height_and_plot(txt_file):
    """
    Steps:
      1. Load the point cloud (X, Y, Z, intensity).
      2. Detect 'ground' points using a 23.1% slope threshold; fit plane; normalize Z.
      3. Remove points below 0 (under the plane).
      4. Exclude points below 0.15 m (dark/low points).
      5. Among points >= 0.15 m, compute key percentiles (50%, 60%, 70%, 80%, 90%).
      6. For each percentile, compute:
         - # of points (cumulative)
         - Average intensity
      7. For points >= 50th percentile:
         - Compute bounding-box area in X–Y
         - Compute bounding-box volume: area * (maxZ - p50)
      8. Plot XZ view with color-coded bins.
      9. Return the 99.5th percentile height.
    """
    # 1. Load data
    points, intensities = load_txt_point_cloud(txt_file)
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    
    # 2. Identify ground points & fit plane
    lowest_z = np.min(Z)
    x_range = np.max(X) - np.min(X)
    ground_threshold = lowest_z + (0.231 * x_range)  # slope-based threshold
    ground_mask = (Z <= ground_threshold)
    
    ground_points = points[ground_mask]
    if len(ground_points) < 3:
        raise ValueError("Insufficient ground points to fit plane.")
    
    # Keep the 90th percentile of ground points for plane fitting
    ground_z_90th = np.percentile(ground_points[:, 2], 90)
    high_ground_mask = (ground_points[:, 2] >= ground_z_90th)
    selected_ground_points = ground_points[high_ground_mask]
    
    # Fit plane
    normal, d, ground_level = fit_plane(selected_ground_points)
    
    # Subtract the plane's Z to normalize
    Z -= ground_level
    
    # Remove points below the plane
    above_plane_mask = (Z >= 0)
    X = X[above_plane_mask]
    Y = Y[above_plane_mask]
    Z = Z[above_plane_mask]
    intensities = intensities[above_plane_mask]
    
    # 3. Exclude points below 0.15 m (dark/low points)
    dark_threshold = 0.15
    dark_mask = (Z > 0) & (Z < dark_threshold)
    
    above_015_mask = (Z >= dark_threshold)
    X_above_015 = X[above_015_mask]
    Y_above_015 = Y[above_015_mask]
    Z_above_015 = Z[above_015_mask]
    intensities_above_015 = intensities[above_015_mask]
    
    if len(Z_above_015) == 0:
        print("No points above 0.15 m. Exiting.")
        return 0
    
    # 4. Compute percentiles among points >= 0.15 m
    p50 = np.percentile(Z_above_015, 50)
    p60 = np.percentile(Z_above_015, 60)
    p70 = np.percentile(Z_above_015, 70)
    p80 = np.percentile(Z_above_015, 80)
    p90 = np.percentile(Z_above_015, 90)
    
    # Cumulative counts & average intensities for each threshold
    count_top_50 = np.sum(Z_above_015 >= p50)
    count_top_60 = np.sum(Z_above_015 >= p60)
    count_top_70 = np.sum(Z_above_015 >= p70)
    count_top_80 = np.sum(Z_above_015 >= p80)
    count_top_90 = np.sum(Z_above_015 >= p90)
    
    avg_int_50 = np.mean(intensities_above_015[Z_above_015 >= p50]) if count_top_50 > 0 else 0
    avg_int_60 = np.mean(intensities_above_015[Z_above_015 >= p60]) if count_top_60 > 0 else 0
    avg_int_70 = np.mean(intensities_above_015[Z_above_015 >= p70]) if count_top_70 > 0 else 0
    avg_int_80 = np.mean(intensities_above_015[Z_above_015 >= p80]) if count_top_80 > 0 else 0
    avg_int_90 = np.mean(intensities_above_015[Z_above_015 >= p90]) if count_top_90 > 0 else 0
    
    # 5. For points >= the 50th percentile, compute bounding-box area & volume
    above_p50_mask = (Z_above_015 >= p50)
    X_50 = X_above_015[above_p50_mask]
    Y_50 = Y_above_015[above_p50_mask]
    Z_50 = Z_above_015[above_p50_mask]
    
    if len(X_50) < 2:
        bounding_box_area = 0.0
        bounding_box_volume = 0.0
    else:
        x_min, x_max = np.min(X_50), np.max(X_50)
        y_min, y_max = np.min(Y_50), np.max(Y_50)
        width = x_max - x_min
        length = y_max - y_min
        bounding_box_area = width * length
        
        # Height of the prism is from Z = p50 to Z = max(Z_50)
        z_max_50 = np.max(Z_50)
        prism_height = z_max_50 - p50
        if prism_height < 0:
            prism_height = 0.0
        bounding_box_volume = bounding_box_area * prism_height
    
    # 6. Compute the 99.5th percentile (plant height)
    p99_top_z = np.percentile(Z, 99.5)
    
    # Print results
    print(f"Number of points >= 0.15 m: {len(X_above_015)}")
    print(f"50th percentile (p50): {p50:.3f} m")
    print(f"60th percentile (p60): {p60:.3f} m")
    print(f"70th percentile (p70): {p70:.3f} m")
    print(f"80th percentile (p80): {p80:.3f} m")
    print(f"90th percentile (p90): {p90:.3f} m\n")
    
    print(f"Points >= 50th pct: {count_top_50}  | Avg Intensity = {avg_int_50:.2f}")
    print(f"Points >= 60th pct: {count_top_60}  | Avg Intensity = {avg_int_60:.2f}")
    print(f"Points >= 70th pct: {count_top_70}  | Avg Intensity = {avg_int_70:.2f}")
    print(f"Points >= 80th pct: {count_top_80}  | Avg Intensity = {avg_int_80:.2f}")
    print(f"Points >= 90th pct: {count_top_90}  | Avg Intensity = {avg_int_90:.2f}\n")
    
    print(f"Bounding-box area (>= p50): {bounding_box_area:.3f} sq.units")
    print(f"Bounding-box volume (>= p50): {bounding_box_volume:.3f} cubic units")
    print(f"99.5th percentile height: {p99_top_z:.3f} m\n")
    
    # 7. Plot the data in XZ, color-coded
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # All points above plane in light gray
    ax.scatter(X, Z, color='lightgray', s=1, label='All Points (Above Plane)')
    
    # Ground points (shift them for plotting)
    ground_pts_shifted = selected_ground_points.copy()
    ground_pts_shifted[:, 2] -= ground_level
    ax.scatter(ground_pts_shifted[:, 0], ground_pts_shifted[:, 2],
               color='red', s=5, label='Ground Points (Fitted)')
    
    # Dark/low points in black (0 < Z < 0.15)
    ax.scatter(X[dark_mask], Z[dark_mask], color='black', s=5, label='Points < 0.15 m')
    
    # Color-code the bins above 0.15 m
    # We'll define bins: [0.15, p50), [p50, p60), [p60, p70), [p70, p80), [p80, p90), [p90, ∞)
    bin1_mask = (Z_above_015 >= 0.15) & (Z_above_015 < p50)
    bin2_mask = (Z_above_015 >= p50)  & (Z_above_015 < p60)
    bin3_mask = (Z_above_015 >= p60)  & (Z_above_015 < p70)
    bin4_mask = (Z_above_015 >= p70)  & (Z_above_015 < p80)
    bin5_mask = (Z_above_015 >= p80)  & (Z_above_015 < p90)
    bin6_mask = (Z_above_015 >= p90)
    
    ax.scatter(X_above_015[bin1_mask], Z_above_015[bin1_mask],
               color='blue', s=5, label='0.15 to <50%')
    ax.scatter(X_above_015[bin2_mask], Z_above_015[bin2_mask],
               color='green', s=5, label='50% to <60%')
    ax.scatter(X_above_015[bin3_mask], Z_above_015[bin3_mask],
               color='orange', s=5, label='60% to <70%')
    ax.scatter(X_above_015[bin4_mask], Z_above_015[bin4_mask],
               color='purple', s=5, label='70% to <80%')
    ax.scatter(X_above_015[bin5_mask], Z_above_015[bin5_mask],
               color='fuchsia', s=5, label='80% to <90%')
    ax.scatter(X_above_015[bin6_mask], Z_above_015[bin6_mask],
               color='yellow', s=5, label='>=90% (Top 10%)')
    
    # Draw a horizontal line for the 99.5th percentile
    ax.axhline(y=p99_top_z, color='blue', linestyle='--', label='99.5th Percentile')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Height, Normalized)')
    ax.set_title('XZ View with Percentile Color-Coding, Bounding Box/Volume (>=50%)')
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.show()
    
    # 8. Return the final 99.5th percentile height
    return p99_top_z

# Example usage
if __name__ == "__main__":
    txt_file_path = 'test1.txt'
    plant_height = calculate_plant_height_and_plot(txt_file_path)
    print(f"Returned Plant Height (99.5th percentile): {plant_height:.3f} m")
