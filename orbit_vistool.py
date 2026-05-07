import streamlit as st
import ephem
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. Physics & TLE Handling ---
def get_altitude_from_tle(line1, line2):
    try:
        mean_motion_revs_day = float(line2[52:63])
    except ValueError:
        return 828.0 # Fallback
    
    mu = 398600.4418
    Re = 6378.137
    n_rad_s = mean_motion_revs_day * (2 * np.pi / 86400.0)
    semi_major_axis = (mu / n_rad_s**2)**(1/3)
    altitude = semi_major_axis - Re
    return altitude

def propagate_orbit_daytime_segments(line1, line2, duration_hours=24, steps_per_orbit=120):
    """
    Propagates the orbit and segments it into daytime passes.
    Returns a list of segments, where each segment is a dict {'lats': [], 'lons': []}
    """
    sat = ephem.readtle("Sat", line1, line2)
    start_date = sat._epoch.datetime()
    
    # Calculate step size
    try:
        mm = float(line2[52:63])
        period_min = (24 * 60) / mm
    except:
        period_min = 100.0
        
    step_seconds = (period_min * 60) / steps_per_orbit
    total_steps = int((duration_hours * 3600) / step_seconds)
    
    segments = []
    current_segment = {'lats': [], 'lons': []}
    is_in_daylight = False
    
    sun = ephem.Sun()
    observer = ephem.Observer()
    observer.elevation = 0 # Sea level check
    
    for i in range(total_steps):
        t = start_date + timedelta(seconds=i*step_seconds)
        
        sat.compute(t)
        
        # Check Sun Elevation at Sub-Satellite Point
        observer.lat = sat.sublat
        observer.lon = sat.sublong
        observer.date = t
        sun.compute(observer)
        
        # Sun altitude > -6 degrees (Civil Twilight) or 0 (Geometric Day)
        # Using -0.1 to avoid flickering at the terminator
        sun_is_up = np.degrees(sun.alt) > -0.5 
        
        if sun_is_up:
            if not is_in_daylight:
                # DAWN: Start a new segment
                if len(current_segment['lats']) > 1:
                    segments.append(current_segment)
                current_segment = {'lats': [], 'lons': []}
                is_in_daylight = True
            
            # Add point to current daytime segment
            current_segment['lats'].append(np.degrees(sat.sublat))
            current_segment['lons'].append(np.degrees(sat.sublong))
            
        else:
            if is_in_daylight:
                # DUSK: End the current segment
                if len(current_segment['lats']) > 1:
                    segments.append(current_segment)
                # Reset
                current_segment = {'lats': [], 'lons': []}
                is_in_daylight = False
    
    # Capture the last segment if it ended in daylight
    if is_in_daylight and len(current_segment['lats']) > 1:
        segments.append(current_segment)
        
    return segments, start_date

def calculate_polygon_edges(lats, lons, swath_km):
    """
    Given a single segment center track, calculate the Left and Right edge coordinates.
    """
    R = 6378.137
    half_swath = swath_km / 2.0
    angular_dist = half_swath / R
    
    left_lats, left_lons = [], []
    right_lats, right_lons = [], []
    
    for i in range(len(lats)-1):
        lat1 = np.radians(lats[i])
        lon1 = np.radians(lons[i])
        lat2 = np.radians(lats[i+1])
        lon2 = np.radians(lons[i+1])
        
        d_lon = lon2 - lon1
        y = np.sin(d_lon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
        bearing = np.arctan2(y, x)
        
        # Left (-90) and Right (+90)
        for angle_offset, lat_list, lon_list in [(-np.pi/2, left_lats, left_lons), (np.pi/2, right_lats, right_lons)]:
            theta = bearing + angle_offset
            
            lat_new = np.arcsin(np.sin(lat1)*np.cos(angular_dist) + 
                                np.cos(lat1)*np.sin(angular_dist)*np.cos(theta))
            lon_new = lon1 + np.arctan2(np.sin(theta)*np.sin(angular_dist)*np.cos(lat1),
                                        np.cos(angular_dist)-np.sin(lat1)*np.sin(lat_new))
            
            lat_list.append(np.degrees(lat_new))
            lon_list.append(np.degrees(lon_new))
            
    return left_lats, left_lons, right_lats, right_lons

# --- 2. Session State ---
if 'swath_val' not in st.session_state:
    st.session_state.swath_val = 2826.0
if 'pixel_val' not in st.session_state:
    st.session_state.pixel_val = 1024

def reset_defaults():
    st.session_state.swath_val = 2826.0
    st.session_state.pixel_val = 1024

# --- 3. Calculation Core ---
def calculate_metrics(target_swath, num_pixels, altitude):
    h = altitude
    Re = 6378.137
    K = (Re + h) / Re
    
    if h < 100: return None, "Altitude too low."

    try:
        gamma_max = np.arccos(1/K)
    except:
        return None, "Geometric Error (Altitude too low?)"
        
    max_swath = 2 * Re * gamma_max
    
    if target_swath >= max_swath:
        return None, f"Error: Horizon limit is {max_swath:.0f} km."

    gamma = (target_swath / 2.0) / Re
    num = np.sin(gamma)
    denom = K - np.cos(gamma)
    theta_rad = np.arctan(num / denom)
    fov_deg = np.degrees(2 * theta_rad)
    beta = (2 * theta_rad) / num_pixels
    w_nadir = h * beta
    
    num_edge = K * np.cos(theta_rad)
    sin_sq = (K * np.sin(theta_rad))**2
    if sin_sq >= 1.0: sin_sq = 0.999999
    denom_edge = np.sqrt(1 - sin_sq)
    ds_dtheta_edge = Re * ((num_edge / denom_edge) - 1)
    w_edge = ds_dtheta_edge * beta
    
    return {
        "fov": fov_deg,
        "w_nadir": w_nadir,
        "w_edge": w_edge,
        "beta": beta,
        "h": h,
        "theta_rad": theta_rad
    }, None

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Orbit Designer", layout="wide")
st.title("🛰️ Daytime Orbit Visualizer")

# --- Sidebar ---
st.sidebar.header("Mission Parameters")

# TLE INPUT
st.sidebar.subheader("Satellite TLE")
default_tle_1 = "1 43013U 17073A   22146.79629330  .00000059  00000-0  48737-4 0  9990"
default_tle_2 = "2 43013  98.7159  85.6898 0001514  97.0846 263.0503 14.19554052234151"
tle_line1 = st.sidebar.text_input("Line 1", value=default_tle_1)
tle_line2 = st.sidebar.text_input("Line 2", value=default_tle_2)

# Calculate Altitude
try:
    derived_altitude = get_altitude_from_tle(tle_line1, tle_line2)
    st.sidebar.success(f"Orbit Altitude: **{derived_altitude:.1f} km**")
except Exception as e:
    st.sidebar.error("Invalid TLE")
    derived_altitude = 828.0 

# Other Sliders
target_swath = st.sidebar.slider("Swath Width (km)", 100.0, 3000.0, step=10.0, key='swath_val')
num_pixels = st.sidebar.slider("Sensor Pixels", 100, 5000, step=16, key='pixel_val')

st.sidebar.markdown("---")
st.sidebar.button("↺ Reset Defaults", on_click=reset_defaults)


# --- Run Calculations ---
data, error = calculate_metrics(target_swath, num_pixels, derived_altitude)

if error:
    st.error(error)
else:
    # --- Metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Required FOV", f"{data['fov']:.2f}°")
    c2.metric("Nadir Pixel", f"{data['w_nadir']:.3f} km")
    c3.metric("Edge Pixel", f"{data['w_edge']:.2f} km")
    c4.metric("Distortion", f"{data['w_edge']/data['w_nadir']:.1f}x")
    
    st.markdown("---")
    
    # --- TABS ---
    tab1, tab2 = st.tabs(["📊 Pixel Analysis", "🌍 3D Swath (Daytime Only)"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("Analysis of pixel growth from Nadir to Swath Edge.")
            st.write(f"**Calculated Altitude:** {derived_altitude:.1f} km")
            st.write(f"**Target Swath:** {target_swath} km")
            
        with col2:
            angles = np.linspace(-data['theta_rad'], data['theta_rad'], 100)
            Re = 6378.137
            K = (Re + data['h']) / Re
            beta = data['beta']
            curved, flat = [], []
            for theta in angles:
                theta_abs = np.abs(theta)
                num = K * np.cos(theta_abs)
                denom = np.sqrt(1 - min((K * np.sin(theta_abs))**2, 0.9999))
                curved.append(Re * ((num/denom)-1) * beta)
                flat.append((data['h'] / np.cos(theta_abs)**2) * beta)
                
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(np.degrees(angles), curved, color='#FF4B4B', lw=2.5, label='Real Earth')
            ax.fill_between(np.degrees(angles), curved, color='#FF4B4B', alpha=0.1)
            ax.plot(np.degrees(angles), flat, color='#28a745', lw=2.5, ls='--', label='Flat Earth')
            ax.set_xlim(np.degrees(min(angles)), np.degrees(max(angles)))
            ax.set_ylim(0, max(curved)*1.15)
            ax.grid(True, ls='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)

    with tab2:
        st.subheader("Daytime Coverage Visualization")
        st.write("Displaying only orbit segments where the satellite is in sunlight.")
        
        with st.spinner("Calculating Day/Night cycles and Swath Geometry..."):
            # 1. Get List of Daytime Segments
            segments, sim_date = propagate_orbit_daytime_segments(tle_line1, tle_line2, duration_hours=24)
            st.success(f"Visualizing Date: {sim_date.strftime('%Y-%m-%d')} | Found {len(segments)} daytime passes.")

            # 2. Build Plotly Figure
            fig3d = go.Figure()
            
            # 3. Iterate over segments and plot filled polygons
            for seg in segments:
                lats = seg['lats']
                lons = seg['lons']
                
                if len(lats) < 5: continue # Skip tiny segments
                
                # Calculate edges for this specific segment
                lats_L, lons_L, lats_R, lons_R = calculate_polygon_edges(lats, lons, target_swath)
                
                # Construct the Closed Polygon (Counter-Clockwise)
                # Left Edge -> Right Edge (Reversed) -> Close Loop
                poly_lats = lats_L + lats_R[::-1] + [lats_L[0]]
                poly_lons = lons_L + lons_R[::-1] + [lons_L[0]]
                
                fig3d.add_trace(go.Scattergeo(
                    lon = poly_lons,
                    lat = poly_lats,
                    mode = 'lines',
                    fill = 'toself',
                    fillcolor = 'rgba(0, 255, 255, 0.3)', # Cyan, Transparent
                    line = dict(width=0, color='cyan'), # No border line
                    hoverinfo = 'skip'
                ))
                
                # Optional: Add center line for reference
                fig3d.add_trace(go.Scattergeo(
                    lon = lons, lat = lats,
                    mode = 'lines',
                    line = dict(width=1, color='white', dash='dot'),
                    opacity=0.5,
                    hoverinfo='skip'
                ))

            fig3d.update_geos(
                projection_type="orthographic",
                showcoastlines=True, coastlinecolor="RebeccaPurple",
                showland=True, landcolor="rgb(20, 20, 40)",
                showocean=True, oceancolor="rgb(10, 10, 20)",
                projection_rotation=dict(lon=0, lat=20, roll=0)
            )
            
            fig3d.update_layout(
                height=600,
                margin={"r":0,"t":0,"l":0,"b":0},
                paper_bgcolor="black",
                font_color="white",
                showlegend=False
            )
            
            st.plotly_chart(fig3d, use_container_width=True)


