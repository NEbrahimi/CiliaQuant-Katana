import streamlit as st
import cv2
from nd2reader import ND2Reader
import os
import tempfile
import subprocess
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import UnivariateSpline
import seaborn as sns
import matplotlib as mpl
from scipy.signal import hilbert
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from scipy.ndimage import center_of_mass




# Set page configuration and styles
# st.set_page_config(page_title="Cilia Analysis Tool", layout="wide")
# st.markdown(
#     """b
#     <style>
#     .css-18e3th9 {background-color: #F8F9FA;}
#     .css-1d391kg {background-color: white;}
#     </style>
#     """, unsafe_allow_html=True)

# Clear session state at the beginning of the script
if 'clear_session' not in st.session_state:
    st.session_state.clear()
    st.session_state['clear_session'] = True

# Set page configuration
st.set_page_config(page_title="Cilia Analysis Tool", layout="wide")

# Add custom CSS for the table borders and text color
st.markdown(
    """
    <style>
    table {
        border-collapse: collapse;
        color: black !important;
    }
    table, th, td {
        border: 1px solid black !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def convert_nd2_to_video(input_file, output_dir, fps):
    with ND2Reader(input_file) as images:
        height, width = images.metadata['height'], images.metadata['width']
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_filename = os.path.splitext(os.path.basename(input_file.name))[0] + '.mp4'
        video_path = os.path.join(output_dir, video_filename)
        out = cv2.VideoWriter(video_path, fourcc, int(fps), (width, height))
        for frame in images:
            frame_8bit = cv2.normalize(frame, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_color = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2BGR)
            out.write(frame_color)
        out.release()
        return video_path


def convert_video_for_streamlit(input_path):
    output_path = input_path.replace('.mp4', '_compatible.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast', output_path], check=True)
    return output_path


# Function to load the masked video and convert it into a data cube
def load_masked_video_to_data_cube(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    cap.release()
    data_cube = np.stack(frames, axis=2)
    return data_cube


# def calculate_phase_synchronization(data_cube):
#     n_frames = data_cube.shape[2]
#     analytic_signal = hilbert(data_cube, axis=2)
#     instantaneous_phase = np.angle(analytic_signal)
#
#     # Calculate PLV
#     n_pixels = data_cube.shape[0] * data_cube.shape[1]
#     reshaped_phases = instantaneous_phase.reshape(n_pixels, n_frames)
#     plv = np.abs(np.sum(np.exp(1j * reshaped_phases), axis=1)) / n_frames
#     plv = plv.reshape(data_cube.shape[:2])
#
#     # Calculate MPC
#     mpc = np.mean(np.cos(reshaped_phases - reshaped_phases.mean(axis=1, keepdims=True)), axis=1)
#     mpc = mpc.reshape(data_cube.shape[:2])
#
#     return plv, mpc

def calculate_phase_synchronization(data_cube):
    n_frames = data_cube.shape[2]
    analytic_signal = hilbert(data_cube, axis=2)
    instantaneous_phase = np.angle(analytic_signal)

    # Calculate PLV
    n_pixels = data_cube.shape[0] * data_cube.shape[1]
    reshaped_phases = instantaneous_phase.reshape(n_pixels, n_frames)
    plv = np.abs(np.sum(np.exp(1j * reshaped_phases), axis=1)) / n_frames
    plv = plv.reshape(data_cube.shape[:2])

    # Calculate MPC
    mpc = np.mean(np.cos(reshaped_phases - reshaped_phases.mean(axis=1, keepdims=True)), axis=1)
    mpc = mpc.reshape(data_cube.shape[:2])

    return plv, mpc


def calculate_summary_statistics(plv, mpc, threshold=0.2):
    valid_plv = plv[plv > 0]
    valid_mpc = mpc[mpc > 0]

    summary = {
        "Mean PLV": np.mean(valid_plv) if len(valid_plv) > 0 else 0,
        "Std PLV": np.std(valid_plv) if len(valid_plv) > 0 else 0,
        "High PLV %": np.sum(valid_plv >= threshold) / len(valid_plv) * 100 if len(valid_plv) > 0 else 0,
        "Mean MPC": np.mean(valid_mpc) if len(valid_mpc) > 0 else 0,
        "Std MPC": np.std(valid_mpc) if len(valid_mpc) > 0 else 0,
        "High MPC %": np.sum(valid_mpc >= threshold) / len(valid_mpc) * 100 if len(valid_mpc) > 0 else 0
    }

    return summary


def segment_into_rois(mask_img, min_size=10):
    labeled_img = label(mask_img)
    regions = regionprops(labeled_img)
    rois = [region.bbox for region in regions if region.area >= min_size]
    return rois


def calculate_roi_phase_synchronization(data_cube, rois):
    plv_map = np.zeros_like(data_cube[..., 0], dtype=np.float32)
    mpc_map = np.zeros_like(data_cube[..., 0], dtype=np.float32)

    for roi in rois:
        minr, minc, maxr, maxc = roi
        roi_data = data_cube[minr:maxr, minc:maxc, :]
        plv, mpc = calculate_phase_synchronization(roi_data)
        plv_map[minr:maxr, minc:maxc] = plv
        mpc_map[minr:maxr, minc:maxc] = mpc

    return plv_map, mpc_map


def segment_into_clusters(data_cube, mask_img, n_clusters):
    height, width, _ = data_cube.shape
    X, Y = np.meshgrid(range(width), range(height))
    coords = np.stack([Y.flatten(), X.flatten()], axis=-1)

    if mask_img is not None:
        mask_flat = mask_img.flatten()
        coords = coords[mask_flat > 0]
        data_flat = data_cube.reshape(-1, data_cube.shape[2])
        data_flat = data_flat[mask_flat > 0]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    labels = np.zeros((height, width), dtype=int)
    labels_flat = kmeans.predict(coords)
    labels[mask_img > 0] = labels_flat + 1  # Ensure that background is not labeled as a cluster

    return labels, kmeans.cluster_centers_


def calculate_cluster_phase_synchronization(data_cube, labels, n_clusters):
    height, width, _ = data_cube.shape
    plv_map = np.zeros((height, width), dtype=np.float32)
    mpc_map = np.zeros((height, width), dtype=np.float32)

    for cluster_id in range(1, n_clusters + 1):
        mask = (labels == cluster_id)
        if np.sum(mask) == 0:
            continue  # Skip empty clusters
        roi_data = data_cube * mask[..., np.newaxis]
        plv, mpc = calculate_phase_synchronization(roi_data)
        plv_map[mask] = plv[mask]
        mpc_map[mask] = mpc[mask]

    return plv_map, mpc_map


def calculate_wave_properties(data_cube, fps):
    """
    Calculate wave properties such as speed and direction using the phase gradient method.

    Args:
        data_cube (numpy.ndarray): The input data cube.
        fps (float): Frames per second of the video.

    Returns:
        wave_speed (numpy.ndarray): Speed of the waves.
        wave_direction (numpy.ndarray): Direction of the waves.
        wave_speed_mean (float): Mean speed of the waves.
        wave_speed_std (float): Standard deviation of wave speed.
        wave_direction_mean (float): Mean direction of the waves.
        wave_direction_std (float): Standard deviation of wave direction.
    """
    analytic_signal = hilbert(data_cube, axis=2)
    instantaneous_phase = np.angle(analytic_signal)
    phase_diff_x = np.diff(instantaneous_phase, axis=0, prepend=0)
    phase_diff_y = np.diff(instantaneous_phase, axis=1, prepend=0)
    phase_diff_t = np.diff(instantaneous_phase, axis=2, prepend=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        speed_x = np.where(phase_diff_x != 0, phase_diff_t / phase_diff_x, 0)
        speed_y = np.where(phase_diff_y != 0, phase_diff_t / phase_diff_y, 0)

    wave_speed = np.sqrt(speed_x ** 2 + speed_y ** 2).mean(axis=2)
    wave_direction = np.arctan2(speed_y, speed_x).mean(axis=2)

    # Calculate mean and standard deviation for speed and direction
    wave_speed_mean = np.mean(wave_speed)
    wave_speed_std = np.std(wave_speed)
    wave_direction_mean = np.mean(wave_direction)
    wave_direction_std = np.std(wave_direction)

    return wave_speed, wave_direction, wave_speed_mean, wave_speed_std, wave_direction_mean, wave_direction_std


def compute_wave_properties(roi_phases, fps):
    # Calculate the gradient of the phase
    grad_y, grad_x = np.gradient(roi_phases, axis=(0, 1))

    # Calculate the speed
    speed = np.sqrt(grad_x**2 + grad_y**2) * fps / (2 * np.pi)  # Speed in pixels/frame

    # Calculate the direction
    direction = np.arctan2(grad_y, grad_x)

    return speed, direction

def normalize_wave_speed(wave_speed):
    # Apply a threshold to remove extreme values
    threshold = np.percentile(wave_speed, 99)  # Threshold at 99th percentile
    wave_speed[wave_speed > threshold] = threshold

    # Normalize the wave speed to the range [0, 1]
    wave_speed_normalized = (wave_speed - wave_speed.min()) / (wave_speed.max() - wave_speed.min())

    return wave_speed_normalized


def pixel_wise_fft_filtered_and_masked(video_path, fps, freq_min, freq_max, mag_threshold):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return None, None, None # Return paths for mask and magnitude images
    frames = []
    while ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frames.append(gray_frame)
        frames.append(gray_frame.astype(np.float32)) # Convert frame to float32 immediately for mean subtraction
        ret, frame = cap.read()
    cap.release()

    data_cube = np.stack(frames, axis=2)

    # Subtract the mean from each pixel's time series
    mean_per_pixel = np.mean(data_cube, axis=2, keepdims=True)
    data_cube -= mean_per_pixel  # Remove the DC component

    fft_cube = fftshift(fft(data_cube, axis=2), axes=(2,))
    magnitude = np.abs(fft_cube)
    phase = np.angle(fft_cube)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(frames), d=1 / fps))
    valid_indices = (freqs >= freq_min) & (freqs <= freq_max)
    magnitude_filtered = magnitude[:, :, valid_indices]
    phase_filtered = phase[:, :, valid_indices]
    freqs_filtered = freqs[valid_indices]

    dominant_freq_indices = np.argmax(magnitude_filtered, axis=2)
    dominant_magnitude = np.max(magnitude_filtered, axis=2)
    dominant_phase = np.take_along_axis(phase_filtered, dominant_freq_indices[:, :, np.newaxis], axis=2).squeeze()
    dominant_frequencies = freqs_filtered[dominant_freq_indices]

    mask = (dominant_magnitude >= mag_threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = 10  # Minimum area threshold to keep a contour

    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) > area_threshold:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    mask_path = os.path.join(tempfile.gettempdir(), 'mask.png')
    cv2.imwrite(mask_path, filtered_mask)
    print(f"Mask saved to {mask_path}")

    plt.figure(figsize=(15, 8))
    im = plt.imshow(dominant_frequencies, cmap='jet', interpolation='nearest', vmin=0, vmax=50)
    plt.colorbar(im, label='Dominant Frequency (Hz)')
    # plt.title('Ciliary Beat Frequency Map')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    frequency_map_path = os.path.join(tempfile.gettempdir(), 'frequency_map.png')
    plt.savefig(frequency_map_path)
    plt.close()
    print(f"Frequency map saved to {frequency_map_path}")

    # Create the magnitude map figure
    fig, ax = plt.subplots()
    im = ax.imshow(dominant_magnitude, cmap='hot')
    ax.axis('off')

    # Adjust the position of the color bar to match the image height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.ax.tick_params(labelsize=7)  # Adjust the color bar tick size

    # Save the figure without extra space
    magnitude_path = os.path.join(r"C:\Users\z3541106\codes\datasets\images\storage", 'magnitude_map.png')
    fig.savefig(magnitude_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Save with high DPI for clarity
    plt.close(fig)
    print(f"Magnitude map saved to {magnitude_path}")

    return mask_path, frequency_map_path, magnitude_path


def apply_mask_to_video(video_path, mask_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Mask image not found or unable to read.")
        return

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ret, frame = cap.read()
    if not ret:
        print("Unable to read video frame.")
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, 333, (width, height))

    while ret:
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        out_video.write(masked_frame)
        ret, frame = cap.read()

    cap.release()
    out_video.release()

## FFT with percentile first and then frequency filtering
# def pixel_wise_fft(video_path, fps, freq_min, freq_max):
#     capture = cv2.VideoCapture(video_path)
#     if not capture.isOpened():
#         raise ValueError("Error opening video file")
#
#     num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     ret, frame = capture.read()
#     if not ret:
#         capture.release()
#         raise ValueError("Unable to read video frame")
#
#     frame_height, frame_width = frame.shape[:2]
#     pixel_time_series = np.zeros((frame_height, frame_width, num_frames), dtype=np.float32)
#
#     capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     for i in range(num_frames):
#         ret, frame = capture.read()
#         if not ret:
#             break
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         pixel_time_series[:, :, i] = gray_frame
#
#     capture.release()
#
#     all_power_spectrum = np.abs(np.fft.fft(pixel_time_series.reshape(-1, num_frames), axis=-1))**2
#     power_threshold = np.percentile(all_power_spectrum, 95)
#
#     cbf_map = np.zeros((frame_height, frame_width))
#     max_power_map = np.zeros((frame_height, frame_width))
#     freq_amp_data = []
#
#     freq_bins = np.fft.fftfreq(n=num_frames, d=1 / fps)
#     amplitude_distribution = np.zeros_like(freq_bins)
#     power_distribution = np.zeros_like(freq_bins)
#
#     for i in range(frame_height):
#         for j in range(frame_width):
#             intensity_series = pixel_time_series[i, j, :]
#             fft_result = np.fft.fft(intensity_series)
#             fft_frequencies = np.fft.fftfreq(n=num_frames, d=1 / fps)
#             positive_frequencies = fft_frequencies > 0
#             power_spectrum = np.abs(fft_result[positive_frequencies]) ** 2
#             amplitude_spectrum = np.abs(fft_result[positive_frequencies])
#
#             significant_power_mask = power_spectrum > power_threshold
#             significant_frequencies = fft_frequencies[positive_frequencies][significant_power_mask]
#             significant_power = power_spectrum[significant_power_mask]
#             significant_amplitude = amplitude_spectrum[significant_power_mask]
#
#             freq_range_mask = (significant_frequencies >= freq_min) & (significant_frequencies <= freq_max)
#             filtered_frequencies = significant_frequencies[freq_range_mask]
#             filtered_power = significant_power[freq_range_mask]
#
#             if filtered_frequencies.size > 0:
#                 max_power_idx = np.argmax(filtered_power)
#                 cbf_map[i, j] = filtered_frequencies[max_power_idx]
#                 max_power_map[i, j] = filtered_power[max_power_idx]
#
#             for freq, power, amplitude in zip(significant_frequencies, significant_power, significant_amplitude):
#                 if freq_min <= freq <= freq_max:
#                     freq_idx = np.argmin(np.abs(freq_bins - freq))
#                     amplitude_distribution[freq_idx] += amplitude
#                     power_distribution[freq_idx] += power
#
#         max_amplitude_freq = freq_bins[np.argmax(amplitude_distribution)]
#         max_power_freq = freq_bins[np.argmax(power_distribution)]
#
#         return cbf_map, max_power_map, pd.DataFrame(freq_amp_data), max_amplitude_freq, max_power_freq



# ## FFT with frequency first and then percentile filtering (most recommended)
# def pixel_wise_fft(video_path, fps, freq_min, freq_max):
#     capture = cv2.VideoCapture(video_path)
#     if not capture.isOpened():
#         raise ValueError("Error opening video file")
#
#     num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     ret, frame = capture.read()
#     if not ret:
#         capture.release()
#         raise ValueError("Unable to read video frame")
#
#     frame_height, frame_width = frame.shape[:2]
#     pixel_time_series = np.zeros((frame_height, frame_width, num_frames), dtype=np.float32)
#
#     capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     for i in range(num_frames):
#         ret, frame = capture.read()
#         if not ret:
#             break
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         pixel_time_series[:, :, i] = gray_frame
#
#     capture.release()
#
#     freq_bins = np.fft.fftfreq(n=num_frames, d=1 / fps)
#     amplitude_distribution = np.zeros_like(freq_bins)
#     power_distribution = np.zeros_like(freq_bins)
#
#     freq_amp_data = []
#
#     for i in range(frame_height):
#         for j in range(frame_width):
#             intensity_series = pixel_time_series[i, j, :]
#             fft_result = np.fft.fft(intensity_series)
#             fft_frequencies = np.fft.fftfreq(n=num_frames, d=1 / fps)
#             positive_frequencies = fft_frequencies > 0
#             power_spectrum = np.abs(fft_result[positive_frequencies]) ** 2
#             amplitude_spectrum = np.abs(fft_result[positive_frequencies])
#
#             # Step 1: Frequency Range Filtering
#             freq_range_mask = (fft_frequencies[positive_frequencies] >= freq_min) & (
#                         fft_frequencies[positive_frequencies] <= freq_max)
#             filtered_frequencies = fft_frequencies[positive_frequencies][freq_range_mask]
#             filtered_power = power_spectrum[freq_range_mask]
#             filtered_amplitude = amplitude_spectrum[freq_range_mask]
#
#             # Step 2: Power Threshold Filtering
#             if filtered_power.size > 0:
#                 power_threshold = np.percentile(filtered_power, 95)
#                 significant_power_mask = filtered_power > power_threshold
#                 significant_frequencies = filtered_frequencies[significant_power_mask]
#                 significant_power = filtered_power[significant_power_mask]
#                 significant_amplitude = filtered_amplitude[significant_power_mask]
#
#                 for freq, power, amplitude in zip(significant_frequencies, significant_power, significant_amplitude):
#                     freq_idx = np.argmin(np.abs(freq_bins - freq))
#                     amplitude_distribution[freq_idx] += amplitude
#                     power_distribution[freq_idx] += power
#                     freq_amp_data.append({"Frequency": freq, "Amplitude": amplitude, "Power": power})
#
#     max_amplitude_freq = freq_bins[np.argmax(amplitude_distribution)]
#     max_power_freq = freq_bins[np.argmax(power_distribution)]
#
#     return pd.DataFrame(freq_amp_data), max_amplitude_freq, max_power_freq

# ### FFT without percentile filtering
# def pixel_wise_fft(video_path, fps, freq_min, freq_max):
#     capture = cv2.VideoCapture(video_path)
#     if not capture.isOpened():
#         raise ValueError("Error opening video file")
#
#     num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     ret, frame = capture.read()
#     if not ret:
#         capture.release()
#         raise ValueError("Unable to read video frame")
#
#     frame_height, frame_width = frame.shape[:2]
#     pixel_time_series = np.zeros((frame_height, frame_width, num_frames), dtype=np.float32)
#
#     capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     for i in range(num_frames):
#         ret, frame = capture.read()
#         if not ret:
#             break
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         pixel_time_series[:, :, i] = gray_frame
#
#     capture.release()
#
#     freq_bins = np.fft.fftfreq(n=num_frames, d=1 / fps)
#     amplitude_distribution = np.zeros_like(freq_bins)
#     power_distribution = np.zeros_like(freq_bins)
#
#     freq_amp_data = []
#
#     for i in range(frame_height):
#         for j in range(frame_width):
#             intensity_series = pixel_time_series[i, j, :]
#             fft_result = np.fft.fft(intensity_series)
#             fft_frequencies = np.fft.fftfreq(n=num_frames, d=1 / fps)
#             positive_frequencies = fft_frequencies > 0
#             power_spectrum = np.abs(fft_result[positive_frequencies]) ** 2
#             amplitude_spectrum = np.abs(fft_result[positive_frequencies])
#
#             # Step 1: Frequency Range Filtering
#             freq_range_mask = (fft_frequencies[positive_frequencies] >= freq_min) & (
#                         fft_frequencies[positive_frequencies] <= freq_max)
#             filtered_frequencies = fft_frequencies[positive_frequencies][freq_range_mask]
#             filtered_power = power_spectrum[freq_range_mask]
#             filtered_amplitude = amplitude_spectrum[freq_range_mask]
#
#             # Debug: Print filtered frequencies and amplitudes
#             print(f"Filtered Frequencies: {filtered_frequencies}")
#             print(f"Filtered Amplitudes: {filtered_amplitude}")
#
#             # Accumulate amplitude and power distributions
#             for freq, power, amplitude in zip(filtered_frequencies, filtered_power, filtered_amplitude):
#                 freq_idx = np.argmin(np.abs(freq_bins - freq))
#                 amplitude_distribution[freq_idx] += amplitude
#                 power_distribution[freq_idx] += power
#                 freq_amp_data.append({"Frequency": freq, "Amplitude": amplitude, "Power": power})
#
#     max_amplitude_freq = freq_bins[np.argmax(amplitude_distribution)]
#     max_power_freq = freq_bins[np.argmax(power_distribution)]
#
#     # Debug: Print the frequency distributions
#     print(f"Amplitude Distribution: {amplitude_distribution}")
#     print(f"Power Distribution: {power_distribution}")
#
#     return pd.DataFrame(freq_amp_data), max_amplitude_freq, max_power_freq


# #FFT with interpolation and percentile
# def pixel_wise_fft(video_path, fps, freq_min, freq_max):
#     capture = cv2.VideoCapture(video_path)
#     if not capture.isOpened():
#         raise ValueError("Error opening video file")
#
#     num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     ret, frame = capture.read()
#     if not ret:
#         capture.release()
#         raise ValueError("Unable to read video frame")
#
#     frame_height, frame_width = frame.shape[:2]
#     pixel_time_series = np.zeros((frame_height, frame_width, num_frames), dtype=np.float32)
#
#     capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     for i in range(num_frames):
#         ret, frame = capture.read()
#         if not ret:
#             break
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         pixel_time_series[:, :, i] = gray_frame
#
#     capture.release()
#
#     freq_bins = np.fft.fftfreq(n=num_frames, d=1 / fps)
#     positive_freq_bins = freq_bins[freq_bins > 0]
#     amplitude_distribution = np.zeros(len(positive_freq_bins))
#     power_distribution = np.zeros(len(positive_freq_bins))
#
#     freq_amp_data = []
#
#     for i in range(frame_height):
#         for j in range(frame_width):
#             intensity_series = pixel_time_series[i, j, :]
#             fft_result = np.fft.fft(intensity_series)
#             fft_frequencies = np.fft.fftfreq(n=num_frames, d=1 / fps)
#             positive_frequencies = fft_frequencies > 0
#             power_spectrum = np.abs(fft_result[positive_frequencies]) ** 2
#             amplitude_spectrum = np.abs(fft_result[positive_frequencies])
#
#             # Step 1: Frequency Range Filtering
#             freq_range_mask = (fft_frequencies[positive_frequencies] >= freq_min) & (
#                         fft_frequencies[positive_frequencies] <= freq_max)
#             filtered_frequencies = fft_frequencies[positive_frequencies][freq_range_mask]
#             filtered_power = power_spectrum[freq_range_mask]
#             filtered_amplitude = amplitude_spectrum[freq_range_mask]
#
#             # Step 2: Power Threshold Filtering
#             if filtered_power.size > 0:
#                 power_threshold = np.percentile(filtered_power, 95)
#                 significant_power_mask = filtered_power > power_threshold
#                 significant_frequencies = filtered_frequencies[significant_power_mask]
#                 significant_power = filtered_power[significant_power_mask]
#                 significant_amplitude = filtered_amplitude[significant_power_mask]
#
#                 for freq, power, amplitude in zip(significant_frequencies, significant_power, significant_amplitude):
#                     freq_idx = np.argmin(np.abs(positive_freq_bins - freq))
#                     amplitude_distribution[freq_idx] += amplitude
#                     power_distribution[freq_idx] += power
#                     freq_amp_data.append({"Frequency": freq, "Amplitude": amplitude, "Power": power})
#
#     # Interpolate distributions
#     interpolated_freqs = np.linspace(positive_freq_bins.min(), positive_freq_bins.max(), 1000)
#     amplitude_spline = UnivariateSpline(positive_freq_bins, amplitude_distribution, s=0)
#     power_spline = UnivariateSpline(positive_freq_bins, power_distribution, s=0)
#
#     interpolated_amplitude_distribution = amplitude_spline(interpolated_freqs)
#     interpolated_power_distribution = power_spline(interpolated_freqs)
#
#     max_amplitude_freq = interpolated_freqs[np.argmax(interpolated_amplitude_distribution)]
#     max_power_freq = interpolated_freqs[np.argmax(interpolated_power_distribution)]
#
#     return pd.DataFrame(freq_amp_data), max_amplitude_freq, max_power_freq

#FFT with interpolation and percentile and mean subtraction
# def pixel_wise_fft(video_path, fps, freq_min, freq_max):
#     capture = cv2.VideoCapture(video_path)
#     if not capture.isOpened():
#         raise ValueError("Error opening video file")
#
#     num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     ret, frame = capture.read()
#     if not ret:
#         capture.release()
#         raise ValueError("Unable to read video frame")
#
#     frame_height, frame_width = frame.shape[:2]
#     pixel_time_series = np.zeros((frame_height, frame_width, num_frames), dtype=np.float32)
#
#     capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     for i in range(num_frames):
#         ret, frame = capture.read()
#         if not ret:
#             break
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         pixel_time_series[:, :, i] = gray_frame
#
#     capture.release()
#
#     # Subtract the mean from each pixel's time series
#     mean_per_pixel = np.mean(pixel_time_series, axis=2, keepdims=True)
#     pixel_time_series -= mean_per_pixel
#
#     freq_bins = np.fft.fftfreq(n=num_frames, d=1 / fps)
#     positive_freq_bins = freq_bins[freq_bins > 0]
#     amplitude_distribution = np.zeros(len(positive_freq_bins))
#     power_distribution = np.zeros(len(positive_freq_bins))
#
#     freq_amp_data = []
#
#     fft_results = np.zeros((frame_height, frame_width, len(positive_freq_bins)), dtype=np.float32)
#
#     for i in range(frame_height):
#         for j in range(frame_width):
#             intensity_series = pixel_time_series[i, j, :]
#             fft_result = np.fft.fft(intensity_series)
#             fft_frequencies = np.fft.fftfreq(n=num_frames, d=1 / fps)
#             positive_frequencies = fft_frequencies > 0
#             power_spectrum = np.abs(fft_result[positive_frequencies]) ** 2
#             amplitude_spectrum = np.abs(fft_result[positive_frequencies])
#
#             # Step 1: Frequency Range Filtering
#             freq_range_mask = (fft_frequencies[positive_frequencies] >= freq_min) & (
#                         fft_frequencies[positive_frequencies] <= freq_max)
#             filtered_frequencies = fft_frequencies[positive_frequencies][freq_range_mask]
#             filtered_power = power_spectrum[freq_range_mask]
#             filtered_amplitude = amplitude_spectrum[freq_range_mask]
#
#             # Step 2: Power Threshold Filtering
#             if filtered_power.size > 0:
#                 power_threshold = np.percentile(filtered_power, 95)
#                 significant_power_mask = filtered_power > power_threshold
#                 significant_frequencies = filtered_frequencies[significant_power_mask]
#                 significant_power = filtered_power[significant_power_mask]
#                 significant_amplitude = filtered_amplitude[significant_power_mask]
#
#                 for freq, power, amplitude in zip(significant_frequencies, significant_power, significant_amplitude):
#                     freq_idx = np.argmin(np.abs(positive_freq_bins - freq))
#                     amplitude_distribution[freq_idx] += amplitude
#                     power_distribution[freq_idx] += power
#                     freq_amp_data.append({"Frequency": freq, "Amplitude": amplitude, "Power": power})
#
#             fft_results[i, j, :] = power_spectrum
#
#     # Interpolate distributions
#     interpolated_freqs = np.linspace(positive_freq_bins.min(), positive_freq_bins.max(), 1000)
#     amplitude_spline = UnivariateSpline(positive_freq_bins, amplitude_distribution, s=0)
#     power_spline = UnivariateSpline(positive_freq_bins, power_distribution, s=0)
#
#     interpolated_amplitude_distribution = amplitude_spline(interpolated_freqs)
#     interpolated_power_distribution = power_spline(interpolated_freqs)
#
#     max_amplitude_freq = interpolated_freqs[np.argmax(interpolated_amplitude_distribution)]
#     max_power_freq = interpolated_freqs[np.argmax(interpolated_power_distribution)]
#
#     return pd.DataFrame(freq_amp_data), max_amplitude_freq, max_power_freq, interpolated_freqs, interpolated_power_distribution, fft_results
#     # return pd.DataFrame(freq_amp_data), max_amplitude_freq, max_power_freq, interpolated_freqs, interpolated_power_distribution

#FFT with interpolation and percentile and mean subtraction with masked issue resolved

def pixel_wise_fft(video_path, mask_path, fps, freq_min, freq_max):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Error opening video file")

    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()
    if not ret:
        capture.release()
        raise ValueError("Unable to read video frame")

    frame_height, frame_width = frame.shape[:2]
    pixel_time_series = np.zeros((frame_height, frame_width, num_frames), dtype=np.float32)

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(num_frames):
        ret, frame = capture.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pixel_time_series[:, :, i] = gray_frame

    capture.release()

    # Load and apply the mask image if provided
    if mask_path:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise ValueError("Failed to read the mask image.")
        mask_img = mask_img.astype(np.bool_)
        pixel_time_series[~mask_img] = 0

    # Subtract the mean from each pixel's time series
    mean_per_pixel = np.mean(pixel_time_series, axis=2, keepdims=True)
    pixel_time_series -= mean_per_pixel

    freq_bins = np.fft.fftfreq(num_frames, d=1 / fps)
    positive_freq_bins = freq_bins[freq_bins > 0]
    amplitude_distribution = np.zeros(len(positive_freq_bins))
    power_distribution = np.zeros(len(positive_freq_bins))

    freq_amp_data = []

    fft_results = np.zeros((frame_height, frame_width, len(positive_freq_bins)), dtype=np.float32)

    for i in range(frame_height):
        for j in range(frame_width):
            if not mask_path or mask_img[i, j]:  # Process only if no mask or pixel is not masked
                intensity_series = pixel_time_series[i, j, :]
                fft_result = np.fft.fft(intensity_series)
                fft_frequencies = np.fft.fftfreq(num_frames, d=1 / fps)
                positive_frequencies = fft_frequencies > 0
                power_spectrum = np.abs(fft_result[positive_frequencies]) ** 2
                amplitude_spectrum = np.abs(fft_result[positive_frequencies])

                # Step 1: Frequency Range Filtering
                freq_range_mask = (fft_frequencies[positive_frequencies] >= freq_min) & (
                        fft_frequencies[positive_frequencies] <= freq_max)
                filtered_frequencies = fft_frequencies[positive_frequencies][freq_range_mask]
                filtered_power = power_spectrum[freq_range_mask]
                filtered_amplitude = amplitude_spectrum[freq_range_mask]

                # Step 2: Power Threshold Filtering
                if filtered_power.size > 0:
                    power_threshold = np.percentile(filtered_power, 95)
                    significant_power_mask = filtered_power > power_threshold
                    significant_frequencies = filtered_frequencies[significant_power_mask]
                    significant_power = filtered_power[significant_power_mask]
                    significant_amplitude = filtered_amplitude[significant_power_mask]

                    for freq, power, amplitude in zip(significant_frequencies, significant_power, significant_amplitude):
                        freq_idx = np.argmin(np.abs(positive_freq_bins - freq))
                        amplitude_distribution[freq_idx] += amplitude
                        power_distribution[freq_idx] += power
                        freq_amp_data.append({"Frequency": freq, "Amplitude": amplitude, "Power": power})

                fft_results[i, j, :] = power_spectrum

    # Interpolate distributions
    interpolated_freqs = np.linspace(positive_freq_bins.min(), positive_freq_bins.max(), 1000)
    amplitude_spline = UnivariateSpline(positive_freq_bins, amplitude_distribution, s=0)
    power_spline = UnivariateSpline(positive_freq_bins, power_distribution, s=0)

    interpolated_amplitude_distribution = amplitude_spline(interpolated_freqs)
    interpolated_power_distribution = power_spline(interpolated_freqs)

    max_amplitude_freq = interpolated_freqs[np.argmax(interpolated_amplitude_distribution)]
    max_power_freq = interpolated_freqs[np.argmax(interpolated_power_distribution)]

    return pd.DataFrame(freq_amp_data), max_amplitude_freq, max_power_freq, interpolated_freqs, interpolated_power_distribution, fft_results

def calculate_statistics(valid_cbfs):
    if valid_cbfs.size > 0:
        mean_cbf = np.mean(valid_cbfs)
        median_cbf = np.median(valid_cbfs)
        std_cbf = np.std(valid_cbfs)
        p25_cbf = np.percentile(valid_cbfs, 25)
        p75_cbf = np.percentile(valid_cbfs, 75)

        return {
            'Mean': mean_cbf,
            'Median': median_cbf,
            '25%': p25_cbf,
            '75%': p75_cbf,
            'std': std_cbf
        }
    else:
        raise ValueError("No valid CBFs found. Adjust filtering criteria.")


def report_dominant_frequencies(max_amplitude_freq, max_power_freq):
    return max_amplitude_freq, max_power_freq


def save_maps(cbf_map, max_power_map, cbf_map_path, max_power_map_path):
    plt.figure(figsize=(10, 5))
    plt.imshow(cbf_map, cmap='jet')
    plt.colorbar(label='CBF (Hz)')
    plt.title('Ciliary Beat Frequency Map')
    plt.savefig(cbf_map_path)
    plt.close()
    if os.path.exists(cbf_map_path):
        print(f"CBF map saved to {cbf_map_path}")
    else:
        print(f"Failed to save CBF map to {cbf_map_path}")

    plt.figure(figsize=(10, 5))
    plt.imshow(max_power_map, cmap='hot')
    plt.colorbar(label='Power')
    plt.title('Maximum Power Spectrum Map')
    plt.savefig(max_power_map_path)
    plt.close()
    if os.path.exists(max_power_map_path):
        print(f"Max Power map saved to {max_power_map_path}")
    else:
        print(f"Failed to save Max Power map to {max_power_map_path}")

def compute_grid_cbf(fft_results, fps, grid_size, freq_filter_min, freq_filter_max):
    frame_height, frame_width, num_freqs = fft_results.shape
    cell_width = frame_width // grid_size
    cell_height = frame_height // grid_size

    # Ensure freq_bins and positive_freq_bins align with fft_results
    freq_bins = np.fft.fftfreq(num_freqs * 2, d=1 / fps)[:num_freqs]  # Adjusted to align with fft_results
    positive_freq_bins = freq_bins[freq_bins > 0]

    # Handle the case where the number of positive frequencies does not match the FFT results
    if len(positive_freq_bins) > num_freqs:
        positive_freq_bins = positive_freq_bins[:num_freqs]
    elif len(positive_freq_bins) < num_freqs:
        positive_freq_bins = np.pad(positive_freq_bins, (0, num_freqs - len(positive_freq_bins)), 'constant')

    freq_range_mask = (positive_freq_bins >= freq_filter_min) & (positive_freq_bins <= freq_filter_max)
    filtered_freqs = positive_freq_bins[freq_range_mask]

    cbf_grid = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            cell_fft = fft_results[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width, :]
            cell_fft = cell_fft.reshape(-1, num_freqs)
            mean_fft = np.mean(cell_fft, axis=0)

            power_spectrum = np.abs(mean_fft) ** 2
            filtered_power = power_spectrum[freq_range_mask]

            if filtered_power.size > 0:
                power_threshold = np.percentile(filtered_power, 95)
                significant_power_mask = filtered_power > power_threshold
                significant_power = filtered_power[significant_power_mask]
                significant_freqs = filtered_freqs[significant_power_mask]

                if significant_power.size > 0:
                    dominant_freq_idx = np.argmax(significant_power)
                    cbf_grid[i, j] = significant_freqs[dominant_freq_idx]
                else:
                    cbf_grid[i, j] = 0
            else:
                cbf_grid[i, j] = 0

    return cbf_grid



# Input widgets
with st.sidebar:
    st.title("Step 1: File Upload and Review")
    uploaded_file = st.file_uploader("Upload a .nd2 file", type=["nd2"])
    exposure_time = st.number_input("Exposure Time (seconds)", min_value=0.0001, value=0.003, step=0.0001,
                                    format="%.3f", help="Adjust the Exposure time according to your acquisition.")
    run_step_1 = st.button("Run Step 1")

    st.title("Step 2: Masking")
    freq_min = st.number_input("Min Frequency", value=2, help="Set the minimum frequency for filtering.")
    freq_max = st.number_input("Max Frequency", value=30, help="Set the maximum frequency for filtering.")
    mag_threshold = st.number_input("Magnitude Threshold", value=300,
                                    help="Set the threshold for background detection sensitivity.")
    run_step_2 = st.button("Run Step 2")

    st.title("Step 3: Cilia Beat Frequency Analysis")
    video_source = st.radio("Select Video Source", options=['Original', 'Masked'], index=0,
                            help="Choose whether to use the original or masked video for analysis.")
    freq_filter_min = st.number_input("Frequency Filter Min", value=2, help="Minimum frequency for CBF analysis.")
    freq_filter_max = st.number_input("Frequency Filter Max", value=30, help="Maximum frequency for CBF analysis.")
    run_step_3 = st.button("Run Step 3")

    # Sidebar Input for Step 4
    st.title("Step 4: Grid Analysis")
    grid_size = st.number_input("Grid Size", min_value=1, max_value=20, value=7, step=1,
                                help="Set the size of the grid.")
    run_step_4 = st.button("Run Step 4")

    # Sidebar Input for Step 5
    st.title("Step 5: Cilia Beat Coordination Analysis")
    coordination_video_source = st.radio("Select Video Source", options=['Original', 'Masked'], index=0,
                                         help="Choose whether to use the original or masked video for analysis for coordination.")
    grid_size = st.number_input("Grid Size for Coordination Analysis", min_value=1, max_value=20, value=10, step=1)
    n_clusters = st.number_input("Number of Clusters for Coordination Analysis", min_value=1, max_value=10, value=5,
                                 step=1)
    run_step_5 = st.button("Run Step 5")


# Define tooltips for each metric
tooltips = {
    'Standard Deviation of CBF': 'The standard deviation of the Ciliary Beat Frequency, indicating variability.',
    'Coefficient of Variation': 'The ratio of the standard deviation to the mean, indicating relative variability.'
}


# Step 1 processing
if uploaded_file and exposure_time > 0 and run_step_1:
    # Clear session state for subsequent steps
    keys_to_clear = ['mask_path', 'fft_results', 'selected_video_path', 'original_video_permanent_path', 'masked_video_permanent_path', 'compatible_masked_video_path']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    fps = 1 / exposure_time
    if 'temp_dir' not in st.session_state:
        st.session_state['temp_dir'] = tempfile.mkdtemp()

    video_path = convert_nd2_to_video(uploaded_file, st.session_state['temp_dir'], fps)
    compatible_video_path = convert_video_for_streamlit(video_path)
    st.session_state['original_video_path'] = video_path
    st.session_state['compatible_video_path'] = compatible_video_path
    with open(compatible_video_path, "rb") as file:
        file_data = file.read()
    st.download_button("Download Original Video", file_data, compatible_video_path.split(os.path.sep)[-1])
    col1, col2 = st.columns(2)
    with col1:
        st.video(compatible_video_path, format='video/mp4', start_time=0, loop=True)


# Define the permanent storage path
storage_path = r"C:\Users\z3541106\codes\datasets\images\storage"
os.makedirs(storage_path, exist_ok=True)

# Step 2 processing
if 'original_video_path' in st.session_state and run_step_2:
    fps = 1 / exposure_time
    video_height = 360  # Height of the video window (adjust based on your video's dimensions)

    # Generate the maps
    mask_path, frequency_map_path, magnitude_path = pixel_wise_fft_filtered_and_masked(
        st.session_state['original_video_path'],
        fps, freq_min, freq_max, mag_threshold
    )

    # Save mask_path to session state
    st.session_state['mask_path'] = mask_path

    frames_output_dir = tempfile.mkdtemp()
    masked_video_path = os.path.join(frames_output_dir, 'masked_video.mp4')

    apply_mask_to_video(st.session_state['original_video_path'], mask_path, masked_video_path)
    compatible_masked_video_path = convert_video_for_streamlit(masked_video_path)

    original_video_permanent_path = os.path.join(r"C:\Users\z3541106\codes\datasets\images\storage",
                                                 'original_video.mp4')
    masked_video_permanent_path = os.path.join(r"C:\Users\z3541106\codes\datasets\images\storage", 'masked_video.mp4')
    shutil.copy(st.session_state['original_video_path'], original_video_permanent_path)
    shutil.copy(masked_video_path, masked_video_permanent_path)

    st.session_state['original_video_permanent_path'] = original_video_permanent_path
    st.session_state['masked_video_permanent_path'] = masked_video_permanent_path
    st.session_state['compatible_masked_video_path'] = compatible_masked_video_path

    col1, col2, col3 = st.columns([1, 1, 1.135])
    with col1:
        st.markdown("<h5 style='text-align: center;'>Original Video</h5>", unsafe_allow_html=True)
        st.video(st.session_state['compatible_video_path'], format='video/mp4', start_time=0, loop=True)
        with open(st.session_state['original_video_path'], "rb") as file:
            st.download_button("Download Original Video", file.read(), file_name='Original_Video.mp4',
                               key="download_orig_video")
    with col2:
        st.markdown("<h5 style='text-align: center;'>Masked Video</h5>", unsafe_allow_html=True)
        st.video(st.session_state['compatible_masked_video_path'], format='video/mp4', start_time=0, loop=True)
        with open(masked_video_path, "rb") as file:
            st.download_button("Download Masked Video", file.read(), file_name='Masked_Video.mp4',
                               key="download_masked_video")
    with col3:
        st.markdown("<h5 style='text-align: center;'>Magnitude Map</h5>", unsafe_allow_html=True)
        magnitude_image = plt.imread(magnitude_path)

        # Ensure the image values are within the 0-1 range for floating-point values
        if magnitude_image.dtype == np.float32 or magnitude_image.dtype == np.float64:
            magnitude_image = np.clip(magnitude_image, 0, 1)
        else:
            magnitude_image = magnitude_image.astype(np.float32) / 255.0

        # Resize the magnitude image to match the height of the video
        aspect_ratio = magnitude_image.shape[1] / magnitude_image.shape[0]
        new_width = int(video_height * aspect_ratio)

        fig, ax = plt.subplots(figsize=(new_width / 100, video_height / 100), dpi=100)
        im = ax.imshow(magnitude_image)
        ax.axis('off')

        # Save resized image without additional color bar
        resized_magnitude_path = os.path.join(r"C:\Users\z3541106\codes\datasets\images\storage",
                                              'resized_magnitude_map.png')
        fig.savefig(resized_magnitude_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)

        # Load the resized image back to add padding
        resized_image = plt.imread(resized_magnitude_path)

        st.image(resized_magnitude_path, use_column_width=True)
        with open(resized_magnitude_path, "rb") as file:
            st.download_button("Download Magnitude Map", file.read(), file_name='Magnitude_Map.png',
                               key="download_magnitude_map")

# Step 3 processing with updated visualizations and structured table layout
if 'original_video_permanent_path' in st.session_state and run_step_3:
    mpl.rcParams['agg.path.chunksize'] = 10000  # You can adjust this value as needed

    fps = 1 / exposure_time  # Calculate frames per second based on exposure time
    video_path = st.session_state['original_video_permanent_path']  # Default to original video

    if video_source == 'Masked' and 'mask_path' in st.session_state and os.path.exists(st.session_state['mask_path']):
        video_path = st.session_state['masked_video_permanent_path']
        mask_path = st.session_state['mask_path']
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            coverage_percent = np.count_nonzero(mask_img) / mask_img.size * 100
        else:
            st.error("Failed to read the mask image. Check the file at the specified path.")
            coverage_percent = "N/A"  # In case mask image is not readable
    else:
        coverage_percent = "100" if video_source == 'Original' else "N/A"

    # Save the selected video path for use in step 4
    st.session_state['selected_video_path'] = video_path

    # Proceed with analysis only if coverage_percent is not "N/A"
    if coverage_percent != "N/A":
        # Analyze video with the specified frequency parameters
        freq_amp_df, max_amplitude_freq, max_power_freq, interpolated_freqs, interpolated_power_distribution, fft_results = pixel_wise_fft(
            video_path, mask_path, fps, freq_filter_min, freq_filter_max
        )

        # Save FFT results to session state for use in step 4
        st.session_state['fft_results'] = fft_results

        valid_cbfs = freq_amp_df[freq_amp_df['Frequency'] > 0]['Frequency']
        stats = calculate_statistics(valid_cbfs)

        col1, col2, col3 = st.columns([3, 1, 4])
        with col1:
            scaled_power_distribution = [val / 1e9 for val in interpolated_power_distribution]
            st.markdown(
                "<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 55px '>Power Spectral Density</h5>",
                unsafe_allow_html=True
            )
            fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
            ax.plot(interpolated_freqs, scaled_power_distribution)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power (x10^9)')
            ax.set_xlim(0, 100)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
            ax.grid(False)
            st.pyplot(fig)

        with col2:
            st.markdown(
                "<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 50px '> Box Plot</h5>",
                unsafe_allow_html=True
            )
            fig, ax = plt.subplots(figsize=(2, 6), dpi=400)
            ax.set_ylabel('Frequency (Hz)')
            sns.boxplot(y=valid_cbfs)
            st.pyplot(fig)

        with col3:
            data = {
                'Measurement': [
                    'Name', 'CFB (power) (Hz)', 'CBF (amplitude) (Hz)', 'Mean Frequency (Hz)',
                    'Median Frequency (Hz)', 'Standard Deviation', '25th Percentile', '75th Percentile',
                    'Coverage %*'
                ],
                'Value': [
                    uploaded_file.name,
                    f"{max_power_freq:.2f}",
                    f"{max_amplitude_freq:.2f}",
                    f"{stats['Mean']:.2f}",
                    f"{stats['Median']:.2f}",
                    f"{stats['std']:.2f}",
                    f"{stats['25%']:.2f}",
                    f"{stats['75%']:.2f}",
                    f"{coverage_percent:.2f}" if coverage_percent != "100" else coverage_percent
                ]
            }
            df = pd.DataFrame(data)
            st.markdown(
                "<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 0px '>Measurement Results</h5>",
                unsafe_allow_html=True
            )
            st.table(df)
            st.markdown('*Coverage % is calculated based on the masked video, and is 100% for the original video.')
    else:
        st.error(
            "Mask path not found or the original video is selected without a need for masking. Please complete the necessary steps to generate the mask if using a masked video."
        )

# Step 4 processing
if 'fft_results' in st.session_state and run_step_4:
    fps = 1 / exposure_time
    fft_results = st.session_state['fft_results']

    # Use the selected video source from Step 3
    video_path = st.session_state['selected_video_path']

    if video_path is None:
        st.error(
            "The selected video source is not available. Please complete the necessary steps for generating the video."
        )
    else:
        cbf_grid = compute_grid_cbf(fft_results, fps, grid_size, freq_filter_min, freq_filter_max)

        # Calculate the variation metrics
        non_zero_cbfs = cbf_grid[cbf_grid > 0]
        std_dev_cbf = np.std(non_zero_cbfs)
        mean_cbf = np.mean(non_zero_cbfs)
        cv_cbf = std_dev_cbf / mean_cbf if mean_cbf != 0 else 0  # Coefficient of Variation

        # Create the grid map
        grid_map_path = os.path.join(tempfile.gettempdir(), 'grid_map.png')
        plt.figure(figsize=(10, 8))
        plt.imshow(cbf_grid, cmap='jet', interpolation='nearest', vmin=0, vmax=50)
        plt.colorbar(label='Dominant Frequency (Hz)')
        plt.savefig(grid_map_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Overlay grid lines and CBF values on video frames
        capture = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G')
        grid_video_path = os.path.join(tempfile.gettempdir(), 'grid_video.avi')
        out = cv2.VideoWriter(grid_video_path, fourcc, fps,
                              (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        font_scale = max(0.3, min(1.0, 1.5 / grid_size))  # Adjusted font scale logic

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            for i in range(grid_size):
                for j in range(grid_size):
                    y1 = i * frame.shape[0] // grid_size
                    y2 = (i + 1) * frame.shape[0] // grid_size
                    x1 = j * frame.shape[1] // grid_size
                    x2 = (j + 1) * frame.shape[1] // grid_size

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, f'{cbf_grid[i, j]:.2f}', (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (0, 255, 0), 1)

            out.write(frame)

        capture.release()
        out.release()


        # Convert video to MP4 format for compatibility
        def convert_video_to_mp4(input_path):
            output_path = input_path.replace('.avi', '_converted.mp4')
            subprocess.run(
                ['ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast', output_path],
                check=True
            )
            return output_path


        converted_grid_video_path = convert_video_to_mp4(grid_video_path)

        col1, col2, col3, col4, col5 = st.columns(
            [2.5, 0.2, 3.0, 0.01, 2]
        )  # Added two columns 2 AND 4 to create spacing

        with col1:
            st.markdown(
                f"<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 0px;'>Grid Video</h5>",
                unsafe_allow_html=True
            )
            st.video(converted_grid_video_path, format='video/mp4')
            with open(converted_grid_video_path, "rb") as file:
                st.download_button("Download Grid Video", file.read(), file_name='grid_video.mp4',
                                   key="download_grid_video")

        with col3:
            st.markdown(
                f"<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 0px;'>CBF Heatmap</h5>",
                unsafe_allow_html=True
            )
            st.image(grid_map_path, use_column_width=True)
            with open(grid_map_path, "rb") as file:
                st.download_button("Download Grid CBF Map", file.read(), file_name='grid_cbf_map.png',
                                   key="download_grid_cbf_map")

        with col5:
            st.markdown(
                f"<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 0px;'>Variation Metrics</h5>",
                unsafe_allow_html=True
            )
            variation_data = {
                'Metric': [
                    'Mean CBF <span class="tooltip"><span class="tooltip-icon">?</span><span class="tooltiptext">The average Ciliary Beat Frequency across all grid cells.</span></span>',
                    'Standard Deviation of CBF <span class="tooltip"><span class="tooltip-icon">?</span><span class="tooltiptext">The standard deviation of Ciliary Beat Frequency across grid cells, measured in Hertz (Hz), representing variability from the mean.</span></span>',
                    'Coefficient of Variation <span class="tooltip"><span class="tooltip-icon">?</span><span class="tooltiptext">The ratio of the standard deviation to the mean, indicating relative variability (0 means no variability, close to 0 means low variability, >1 means high variability).</span></span>'
                ],
                'Value': [f"{mean_cbf:.2f}", f"{std_dev_cbf:.2f}", f"{cv_cbf:.2f}"]
            }
            variation_df = pd.DataFrame(variation_data)

            # Adding custom CSS for tooltips similar to sidebar
            st.markdown(
                """
                <style>
                .tooltip {
                    position: relative;
                    display: inline-block;
                    cursor: pointer;
                    font-size: 14px;
                    color: #6c757d;
                    margin-left: 5px;
                }
                .tooltip .tooltiptext {
                    visibility: hidden;
                    width: 200px;
                    background-color: #f0f0f0;
                    color: #333;
                    text-align: center;
                    border-radius: 6px;
                    border: 1px solid #ccc;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    padding: 5px;
                    position: absolute;
                    z-index: 1;
                    bottom: 150%; /* Position the tooltip above the text */
                    left: 50%;
                    margin-left: -100px;
                    opacity: 0;
                    transition: opacity 0.3s;
                }
                .tooltip:hover .tooltiptext {
                    visibility: visible;
                    opacity: 1;
                }
                .tooltip-icon {
                    font-size: 12px;
                    font-weight: bold;
                    color: #6e7075; /* Darker gray color */
                    background-color: transparent;
                    border: 1.5px solid #6e7075;
                    border-radius: 50%;
                    width: 15px;
                    height: 15px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-left: 5px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    border: 1px solid black;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                variation_df.to_html(index=False, escape=False, justify='center', table_id="variation_metrics_table"),
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <style>
                #variation_metrics_table {
                    margin-left: auto;
                    margin-right: auto;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

# # Step 2 processing
# if 'original_video_path' in st.session_state and run_step_2:
#     fps = 1 / exposure_time
#     video_height = 360  # Height of the video window (adjust based on your video's dimensions)
#
#     # Generate the maps
#     mask_path, frequency_map_path, magnitude_path = pixel_wise_fft_filtered_and_masked(st.session_state['original_video_path'],
#                                                                                        fps, freq_min, freq_max, mag_threshold)
#
#     # Save mask_path to session state
#     st.session_state['mask_path'] = mask_path
#
#     frames_output_dir = tempfile.mkdtemp()
#     masked_video_path = os.path.join(frames_output_dir, 'masked_video.mp4')
#
#     apply_mask_to_video(st.session_state['original_video_path'], mask_path, masked_video_path)
#     compatible_masked_video_path = convert_video_for_streamlit(masked_video_path)
#
#     original_video_permanent_path = os.path.join(r"C:\Users\z3541106\codes\datasets\images\storage", 'original_video.mp4')
#     masked_video_permanent_path = os.path.join(r"C:\Users\z3541106\codes\datasets\images\storage", 'masked_video.mp4')
#     shutil.copy(st.session_state['original_video_path'], original_video_permanent_path)
#     shutil.copy(masked_video_path, masked_video_permanent_path)
#
#     st.session_state['original_video_permanent_path'] = original_video_permanent_path
#     st.session_state['masked_video_permanent_path'] = masked_video_permanent_path
#     st.session_state['compatible_masked_video_path'] = compatible_masked_video_path
#
#     col1, col2, col3 = st.columns([1, 1, 1.135])
#     with col1:
#         # st.markdown("##### Original Video")
#         st.markdown("<h5 style='text-align: center;'>Original Video</h5>", unsafe_allow_html=True)
#         st.video(st.session_state['compatible_video_path'], format='video/mp4', start_time=0, loop=True)
#         with open(st.session_state['original_video_path'], "rb") as file:
#             st.download_button("Download Original Video", file.read(), file_name='Original_Video.mp4', key="download_orig_video")
#     with col2:
#         # st.markdown("##### Masked Video")
#         st.markdown("<h5 style='text-align: center;'>Masked Video</h5>", unsafe_allow_html=True)
#         st.video(st.session_state['compatible_masked_video_path'], format='video/mp4', start_time=0, loop=True)
#         with open(masked_video_path, "rb") as file:
#             st.download_button("Download Masked Video", file.read(), file_name='Masked_Video.mp4', key="download_masked_video")
#     with col3:
#         # st.markdown("##### Magnitude Map")
#         st.markdown("<h5 style='text-align: center;'>Magnitude Map</h5>", unsafe_allow_html=True)
#         magnitude_image = plt.imread(magnitude_path)
#
#         # Ensure the image values are within the 0-1 range for floating-point values
#         if magnitude_image.dtype == np.float32 or magnitude_image.dtype == np.float64:
#             magnitude_image = np.clip(magnitude_image, 0, 1)
#         else:
#             magnitude_image = magnitude_image.astype(np.float32) / 255.0
#
#         # Resize the magnitude image to match the height of the video
#         aspect_ratio = magnitude_image.shape[1] / magnitude_image.shape[0]
#         new_width = int(video_height * aspect_ratio)
#
#         fig, ax = plt.subplots(figsize=(new_width / 100, video_height / 100), dpi=100)
#         im = ax.imshow(magnitude_image)
#         ax.axis('off')
#
#         # Save resized image without additional color bar
#         resized_magnitude_path = os.path.join(r"C:\Users\z3541106\codes\datasets\images\storage", 'resized_magnitude_map.png')
#         fig.savefig(resized_magnitude_path, bbox_inches='tight', pad_inches=0, dpi=300)
#         plt.close(fig)
#
#         # Load the resized image back to add padding
#         resized_image = plt.imread(resized_magnitude_path)
#
#         st.image(resized_magnitude_path, use_column_width=True)
#         with open(resized_magnitude_path, "rb") as file:
#             st.download_button("Download Magnitude Map", file.read(), file_name='Magnitude_Map.png', key="download_magnitude_map")
#
#
# # Step 3 processing with updated visualizations and structured table layout
# if 'original_video_permanent_path' in st.session_state and run_step_3:
#
#     mpl.rcParams['agg.path.chunksize'] = 10000  # You can adjust this value as needed
#
#     fps = 1 / exposure_time  # Calculate frames per second based on exposure time
#     video_path = st.session_state['original_video_permanent_path']  # Get the path of the original video
#
#     mask_path = None
#     if video_source == 'Masked' and 'mask_path' in st.session_state and os.path.exists(st.session_state['mask_path']):
#         mask_path = st.session_state['mask_path']
#         mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask_img is not None:
#             coverage_percent = np.count_nonzero(mask_img) / mask_img.size * 100
#         else:
#             st.error("Failed to read the mask image. Check the file at the specified path.")
#             coverage_percent = "N/A"  # In case mask image is not readable
#     else:
#         coverage_percent = "100" if video_source == 'Original' else "N/A"
#
#     # Proceed with analysis only if coverage_percent is not "N/A"
#     if coverage_percent != "N/A":
#         # Analyze video with the specified frequency parameters
#         freq_amp_df, max_amplitude_freq, max_power_freq, interpolated_freqs, interpolated_power_distribution, fft_results = pixel_wise_fft(
#             video_path, mask_path, fps, freq_filter_min, freq_filter_max)
#
#         # Save FFT results and selected video path to session state for use in step 4
#         st.session_state['fft_results'] = fft_results
#         st.session_state['selected_video_path'] = video_path
#
#         valid_cbfs = freq_amp_df[freq_amp_df['Frequency'] > 0]['Frequency']
#         stats = calculate_statistics(valid_cbfs)
#
#         col1, col2, col3 = st.columns([3, 1, 4])
#         with col1:
#             scaled_power_distribution = [val / 1e9 for val in interpolated_power_distribution]
#             st.markdown(
#                 "<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 55px '>Power Spectral Density</h5>",
#                 unsafe_allow_html=True)
#             fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
#             ax.plot(interpolated_freqs, scaled_power_distribution)
#             ax.set_xlabel('Frequency (Hz)')
#             ax.set_ylabel('Power (x10^9)')
#             ax.set_xlim(0, 100)
#             ax.set_ylim(bottom=0)
#             ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
#             ax.grid(False)
#             st.pyplot(fig)
#
#         with col2:
#             st.markdown(
#                 "<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 50px '> Box Plot</h5>",
#                 unsafe_allow_html=True)
#             fig, ax = plt.subplots(figsize=(2, 6), dpi=400)
#             ax.set_ylabel('Frequency (Hz)')
#             sns.boxplot(y=valid_cbfs)
#             st.pyplot(fig)
#
#         with col3:
#             data = {
#                 'Measurement': ['Name', 'CFB (power) (Hz)', 'CBF (amplitude) (Hz)', 'Mean Frequency (Hz)',
#                                 'Median Frequency (Hz)', 'Standard Deviation', '25th Percentile', '75th Percentile',
#                                 'Coverage %*'],
#                 'Value': [
#                     uploaded_file.name,
#                     f"{max_power_freq:.2f}",
#                     f"{max_amplitude_freq:.2f}",
#                     f"{stats['Mean']:.2f}",
#                     f"{stats['Median']:.2f}",
#                     f"{stats['std']:.2f}",
#                     f"{stats['25%']:.2f}",
#                     f"{stats['75%']:.2f}",
#                     f"{coverage_percent:.2f}" if coverage_percent != "100" else coverage_percent
#                 ]
#             }
#             df = pd.DataFrame(data)
#             st.markdown(
#                 "<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 0px '>Measurement Results</h5>",
#                 unsafe_allow_html=True)
#             st.table(df)
#             st.markdown('*Coverage % is calculated based on the masked video, and is 100% for the original video.')
#     else:
#         st.error(
#             "Mask path not found or the original video is selected without a need for masking. Please complete the necessary steps to generate the mask if using a masked video.")
#
# # Step 4 processing
# if 'fft_results' in st.session_state and run_step_4:
#     fps = 1 / exposure_time
#     fft_results = st.session_state['fft_results']
#
#     # Use the selected video source from Step 3
#     video_path = st.session_state['selected_video_path']
#
#     if video_path is None:
#         st.error(
#             "The selected video source is not available. Please complete the necessary steps for generating the video.")
#     else:
#         cbf_grid = compute_grid_cbf(fft_results, fps, grid_size, freq_filter_min, freq_filter_max)
#
#         # Calculate the variation metrics
#         non_zero_cbfs = cbf_grid[cbf_grid > 0]
#         std_dev_cbf = np.std(non_zero_cbfs)
#         mean_cbf = np.mean(non_zero_cbfs)
#         cv_cbf = std_dev_cbf / mean_cbf if mean_cbf != 0 else 0  # Coefficient of Variation
#
#         # Create the grid map
#         grid_map_path = os.path.join(tempfile.gettempdir(), 'grid_map.png')
#         plt.figure(figsize=(10, 8))
#         plt.imshow(cbf_grid, cmap='jet', interpolation='nearest', vmin=0, vmax=50)
#         plt.colorbar(label='Dominant Frequency (Hz)')
#         # plt.title('Grid Ciliary Beat Frequency Map')
#         # plt.xlabel('Grid X')
#         # plt.ylabel('Grid Y')
#         plt.savefig(grid_map_path, bbox_inches='tight', pad_inches=0)
#         plt.close()
#
#         # Overlay grid lines and CBF values on video frames
#         capture = cv2.VideoCapture(video_path)
#         fourcc = cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G')
#         grid_video_path = os.path.join(tempfile.gettempdir(), 'grid_video.avi')
#         out = cv2.VideoWriter(grid_video_path, fourcc, fps,
#                               (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
#
#         font_scale = max(0.3, min(1.0, 1.5 / grid_size))  # Adjusted font scale logic
#
#         while capture.isOpened():
#             ret, frame = capture.read()
#             if not ret:
#                 break
#
#             for i in range(grid_size):
#                 for j in range(grid_size):
#                     y1 = i * frame.shape[0] // grid_size
#                     y2 = (i + 1) * frame.shape[0] // grid_size
#                     x1 = j * frame.shape[1] // grid_size
#                     x2 = (j + 1) * frame.shape[1] // grid_size
#
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
#                     cv2.putText(frame, f'{cbf_grid[i, j]:.2f}', (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
#                                 (0, 255, 0), 1)
#
#             out.write(frame)
#
#         capture.release()
#         out.release()
#
#         # Convert video to MP4 format for compatibility
#         def convert_video_to_mp4(input_path):
#             output_path = input_path.replace('.avi', '_converted.mp4')
#             subprocess.run(
#                 ['ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast', output_path],
#                 check=True)
#             return output_path
#
#         converted_grid_video_path = convert_video_to_mp4(grid_video_path)
#
#         col1, col2, col3, col4, col5 = st.columns(
#             [2.5, 0.2, 3.0, 0.01, 2])  # Added two columns 2 AND 4 to create spacing
#
#         with col1:
#             st.markdown(
#                 f"<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 0px;'>Grid Video</h5>",
#                 unsafe_allow_html=True)
#             st.video(converted_grid_video_path, format='video/mp4')
#             with open(converted_grid_video_path, "rb") as file:
#                 st.download_button("Download Grid Video", file.read(), file_name='grid_video.mp4',
#                                    key="download_grid_video")
#
#         with col3:
#             st.markdown(
#                 f"<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 0px;'>CBF Heatmap</h5>",
#                 unsafe_allow_html=True)
#             st.image(grid_map_path, use_column_width=True)
#             with open(grid_map_path, "rb") as file:
#                 st.download_button("Download Grid CBF Map", file.read(), file_name='grid_cbf_map.png',
#                                    key="download_grid_cbf_map")
#
#
#         with col5:
#             st.markdown(
#                 f"<h5 style='text-align: center; font-size: 18px; font-weight: bold; margin-left: 0px;'>Variation Metrics</h5>",
#                 unsafe_allow_html=True)
#             variation_data = {
#                 'Metric': [
#                     'Mean CBF <span class="tooltip"><span class="tooltip-icon">?</span><span class="tooltiptext">The average Ciliary Beat Frequency across all grid cells.</span></span>',
#                     'Standard Deviation of CBF <span class="tooltip"><span class="tooltip-icon">?</span><span class="tooltiptext">The standard deviation of Ciliary Beat Frequency across grid cells, measured in Hertz (Hz), representing variability from the mean.</span></span>',
#                     'Coefficient of Variation <span class="tooltip"><span class="tooltip-icon">?</span><span class="tooltiptext">The ratio of the standard deviation to the mean, indicating relative variability (0 means no variability, close to 0 means low variability, >1 means high variability).</span></span>'
#                 ],
#                 'Value': [f"{mean_cbf:.2f}", f"{std_dev_cbf:.2f}", f"{cv_cbf:.2f}"]
#             }
#             variation_df = pd.DataFrame(variation_data)
#
#             # Adding custom CSS for tooltips similar to sidebar
#             st.markdown(
#                 """
#                 <style>
#                 .tooltip {
#                     position: relative;
#                     display: inline-block;
#                     cursor: pointer;
#                     font-size: 14px;
#                     color: #6c757d;
#                     margin-left: 5px;
#                 }
#                 .tooltip .tooltiptext {
#                     visibility: hidden;
#                     width: 200px;
#                     background-color: #f0f0f0;
#                     color: #333;
#                     text-align: center;
#                     border-radius: 6px;
#                     border: 1px solid #ccc;
#                     box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
#                     padding: 5px;
#                     position: absolute;
#                     z-index: 1;
#                     bottom: 150%; /* Position the tooltip above the text */
#                     left: 50%;
#                     margin-left: -100px;
#                     opacity: 0;
#                     transition: opacity 0.3s;
#                 }
#                 .tooltip:hover .tooltiptext {
#                     visibility: visible;
#                     opacity: 1;
#                 }
#                 .tooltip-icon {
#                     font-size: 12px;
#                     font-weight: bold;
#                     color: #6e7075; /* Darker gray color */
#                     background-color: transparent;
#                     border: 1.5px solid #6e7075;
#                     border-radius: 50%;
#                     width: 15px;
#                     height: 15px;
#                     display: flex;
#                     align-items: center;
#                     justify-content: center;
#                     margin-left: 5px;
#                 }
#                 table {
#                     width: 100%;
#                     border-collapse: collapse;
#                 }
#                 th, td {
#                     border: 1px solid black;
#                     padding: 8px;
#                     text-align: left;
#                 }
#                 th {
#                     background-color: #f2f2f2;
#                 }
#                 </style>
#                 """,
#                 unsafe_allow_html=True
#             )
#             st.markdown(
#                 variation_df.to_html(index=False, escape=False, justify='center', table_id="variation_metrics_table"),
#                 unsafe_allow_html=True
#             )
#             st.markdown(
#                 """
#                 <style>
#                 #variation_metrics_table {
#                     margin-left: auto;
#                     margin-right: auto;
#                 }
#                 </style>
#                 """,
#                 unsafe_allow_html=True
#             )

# if 'fft_results' in st.session_state and run_step_5:
#     fps = 1 / exposure_time
#     fft_results = st.session_state['fft_results']
#
#     # Choose video source
#     if coordination_video_source == 'Masked' and 'mask_path' in st.session_state and os.path.exists(
#             st.session_state['mask_path']):
#         mask_path = st.session_state['mask_path']
#         mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask_img is not None:
#             fft_results = fft_results * (mask_img[..., np.newaxis] > 0)
#
#     # Ensure correct data shape for Hilbert transform
#     data_cube = np.abs(fft_results)
#
#     # Calculate Phase-locking Value (PLV) and Mean Phase Coherence (MPC)
#     plv, mpc = calculate_phase_synchronization(data_cube)
#
#     # Set PLV and MPC to zero in masked regions for masked video
#     if coordination_video_source == 'Masked' and mask_img is not None:
#         plv[mask_img == 0] = 0
#         mpc[mask_img == 0] = 0
#
#     # Calculate summary statistics
#     stats = calculate_summary_statistics(plv, mpc, threshold=0.5)
#
#     col1, col2 = st.columns(2)
#
#     # Visualize PLV
#     with col1:
#         st.markdown("<h5 style='text-align: center;'>Phase-locking Value (PLV)</h5>", unsafe_allow_html=True)
#         fig, ax = plt.subplots(figsize=(10, 8))
#         plv_img = ax.imshow(plv, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
#         fig.colorbar(plv_img, ax=ax, label='PLV')
#         plt.title('Phase-locking Value (PLV) Map')
#         plt.xlabel('Pixel X')
#         plt.ylabel('Pixel Y')
#         plv_map_path = os.path.join(tempfile.gettempdir(), 'plv_map.png')
#         plt.savefig(plv_map_path)
#         plt.close(fig)
#         st.image(plv_map_path, use_column_width=True)
#
#     # Visualize MPC
#     with col2:
#         st.markdown("<h5 style='text-align: center;'>Mean Phase Coherence (MPC)</h5>", unsafe_allow_html=True)
#         fig, ax = plt.subplots(figsize=(10, 8))
#         mpc_img = ax.imshow(mpc, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
#         fig.colorbar(mpc_img, ax=ax, label='MPC')
#         plt.title('Mean Phase Coherence (MPC) Map')
#         plt.xlabel('Pixel X')
#         plt.ylabel('Pixel Y')
#         mpc_map_path = os.path.join(tempfile.gettempdir(), 'mpc_map.png')
#         plt.savefig(mpc_map_path)
#         plt.close(fig)
#         st.image(mpc_map_path, use_column_width=True)
#
#     # Display summary statistics
#     st.markdown("<h5 style='text-align: center;'>Summary Statistics</h5>", unsafe_allow_html=True)
#     st.write(stats)


#### step 5 plv, mpc global and roi
# if 'fft_results' in st.session_state and run_step_5:
#     fps = 1 / exposure_time
#     fft_results = st.session_state['fft_results']
#
#     # Choose video source
#     if coordination_video_source == 'Masked' and 'mask_path' in st.session_state and os.path.exists(st.session_state['mask_path']):
#         mask_path = st.session_state['mask_path']
#         mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask_img is not None:
#             fft_results = fft_results * (mask_img[..., np.newaxis] > 0)
#
#     # Ensure correct data shape for Hilbert transform
#     data_cube = np.abs(fft_results)
#
#     # Calculate Global Phase-locking Value (PLV) and Mean Phase Coherence (MPC)
#     global_plv, global_mpc = calculate_phase_synchronization(data_cube)
#
#     # Set PLV and MPC to zero in masked regions for masked video
#     if coordination_video_source == 'Masked' and mask_img is not None:
#         global_plv[mask_img == 0] = 0
#         global_mpc[mask_img == 0] = 0
#
#     # Segment the image into ROIs
#     if coordination_video_source == 'Masked' and mask_img is not None:
#         rois = segment_into_rois(mask_img)
#     else:
#         rois = [(0, 0, data_cube.shape[0], data_cube.shape[1])]
#
#     # Calculate Region-based PLV and MPC
#     roi_plv_map, roi_mpc_map = calculate_roi_phase_synchronization(data_cube, rois)
#
#     # Calculate summary statistics for global PLV and MPC
#     global_stats = calculate_summary_statistics(global_plv, global_mpc, threshold=0.5)
#
#     # Calculate summary statistics for region-based PLV and MPC
#     roi_stats = calculate_summary_statistics(roi_plv_map, roi_mpc_map, threshold=0.5)
#
#     col1, col2, col3, col4 = st.columns(4)
#
#     # Visualize Global PLV
#     with col1:
#         st.markdown("<h5 style='text-align: center;'>Global Phase-locking Value (PLV)</h5>", unsafe_allow_html=True)
#         fig, ax = plt.subplots(figsize=(5, 4))
#         plv_img = ax.imshow(global_plv, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
#         fig.colorbar(plv_img, ax=ax, label='PLV')
#         plt.title('Global PLV Map')
#         plt.xlabel('Pixel X')
#         plt.ylabel('Pixel Y')
#         global_plv_map_path = os.path.join(tempfile.gettempdir(), 'global_plv_map.png')
#         plt.savefig(global_plv_map_path)
#         plt.close(fig)
#         st.image(global_plv_map_path, use_column_width=True)
#
#     # Visualize Global MPC
#     with col2:
#         st.markdown("<h5 style='text-align: center;'>Global Mean Phase Coherence (MPC)</h5>", unsafe_allow_html=True)
#         fig, ax = plt.subplots(figsize=(5, 4))
#         mpc_img = ax.imshow(global_mpc, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
#         fig.colorbar(mpc_img, ax=ax, label='MPC')
#         plt.title('Global MPC Map')
#         plt.xlabel('Pixel X')
#         plt.ylabel('Pixel Y')
#         global_mpc_map_path = os.path.join(tempfile.gettempdir(), 'global_mpc_map.png')
#         plt.savefig(global_mpc_map_path)
#         plt.close(fig)
#         st.image(global_mpc_map_path, use_column_width=True)
#
#     # Visualize Region-based PLV
#     with col3:
#         st.markdown("<h5 style='text-align: center;'>Region-based Phase-locking Value (PLV)</h5>", unsafe_allow_html=True)
#         fig, ax = plt.subplots(figsize=(5, 4))
#         roi_plv_img = ax.imshow(roi_plv_map, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
#         fig.colorbar(roi_plv_img, ax=ax, label='PLV')
#         plt.title('Region-based PLV Map')
#         plt.xlabel('Pixel X')
#         plt.ylabel('Pixel Y')
#         roi_plv_map_path = os.path.join(tempfile.gettempdir(), 'roi_plv_map.png')
#         plt.savefig(roi_plv_map_path)
#         plt.close(fig)
#         st.image(roi_plv_map_path, use_column_width=True)
#
#     # Visualize Region-based MPC
#     with col4:
#         st.markdown("<h5 style='text-align: center;'>Region-based Mean Phase Coherence (MPC)</h5>", unsafe_allow_html=True)
#         fig, ax = plt.subplots(figsize=(5, 4))
#         roi_mpc_img = ax.imshow(roi_mpc_map, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
#         fig.colorbar(roi_mpc_img, ax=ax, label='MPC')
#         plt.title('Region-based MPC Map')
#         plt.xlabel('Pixel X')
#         plt.ylabel('Pixel Y')
#         roi_mpc_map_path = os.path.join(tempfile.gettempdir(), 'roi_mpc_map.png')
#         plt.savefig(roi_mpc_map_path)
#         plt.close(fig)
#         st.image(roi_mpc_map_path, use_column_width=True)
#
#     # Display summary statistics
#     st.markdown("<h5 style='text-align: center;'>Summary Statistics</h5>", unsafe_allow_html=True)
#     st.write("### Global Statistics")
#     st.write(global_stats)
#     st.write("### Region-based Statistics")
#     st.write(roi_stats)

# Step 5 processing
if 'fft_results' in st.session_state and run_step_5:
    fps = 1 / exposure_time
    fft_results = st.session_state['fft_results']

    # Choose video source
    if coordination_video_source == 'Masked' and 'mask_path' in st.session_state and os.path.exists(st.session_state['mask_path']):
        mask_path = st.session_state['mask_path']
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            fft_results = fft_results * (mask_img[..., np.newaxis] > 0)

    # Ensure correct data shape for Hilbert transform
    data_cube = np.abs(fft_results)

    # Calculate Global Phase-locking Value (PLV) and Mean Phase Coherence (MPC)
    global_plv, global_mpc = calculate_phase_synchronization(data_cube)

    # Set PLV and MPC to zero in masked regions for masked video
    if coordination_video_source == 'Masked' and mask_img is not None:
        global_plv[mask_img == 0] = 0
        global_mpc[mask_img == 0] = 0

    # Segment the image into ROIs
    if coordination_video_source == 'Masked' and mask_img is not None:
        rois = segment_into_rois(mask_img)
    else:
        rois = [(0, 0, data_cube.shape[0], data_cube.shape[1])]

    # Calculate Region-based PLV and MPC
    roi_plv_map, roi_mpc_map = calculate_roi_phase_synchronization(data_cube, rois)

    # Calculate summary statistics for global PLV and MPC
    global_stats = calculate_summary_statistics(global_plv, global_mpc, threshold=0.5)

    # Calculate summary statistics for region-based PLV and MPC
    roi_stats = calculate_summary_statistics(roi_plv_map, roi_mpc_map, threshold=0.5)

    # Calculate wave properties for the data cube
    global_wave_speed, global_wave_direction, wave_speed_mean, wave_speed_std, wave_direction_mean, wave_direction_std = calculate_wave_properties(data_cube, fps)
    global_wave_speed_normalized = normalize_wave_speed(global_wave_speed)

    col1, col2, col3, col4 = st.columns(4)

    # Visualize Global PLV
    with col1:
        st.markdown("<h5 style='text-align: center;'>Global Phase-locking Value (PLV)</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        plv_img = ax.imshow(global_plv, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
        fig.colorbar(plv_img, ax=ax, label='PLV')
        plt.title('Global PLV Map')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        global_plv_map_path = os.path.join(tempfile.gettempdir(), 'global_plv_map.png')
        plt.savefig(global_plv_map_path)
        plt.close(fig)
        st.image(global_plv_map_path, use_column_width=True)

    # Visualize Global MPC
    with col2:
        st.markdown("<h5 style='text-align: center;'>Global Mean Phase Coherence (MPC)</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        mpc_img = ax.imshow(global_mpc, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
        fig.colorbar(mpc_img, ax=ax, label='MPC')
        plt.title('Global MPC Map')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        global_mpc_map_path = os.path.join(tempfile.gettempdir(), 'global_mpc_map.png')
        plt.savefig(global_mpc_map_path)
        plt.close(fig)
        st.image(global_mpc_map_path, use_column_width=True)

    # Visualize Region-based PLV
    with col3:
        st.markdown("<h5 style='text-align: center;'>Region-based Phase-locking Value (PLV)</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        roi_plv_img = ax.imshow(roi_plv_map, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
        fig.colorbar(roi_plv_img, ax=ax, label='PLV')
        plt.title('Region-based PLV Map')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        roi_plv_map_path = os.path.join(tempfile.gettempdir(), 'roi_plv_map.png')
        plt.savefig(roi_plv_map_path)
        plt.close(fig)
        st.image(roi_plv_map_path, use_column_width=True)

    # Visualize Region-based MPC
    with col4:
        st.markdown("<h5 style='text-align: center;'>Region-based Mean Phase Coherence (MPC)</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        roi_mpc_img = ax.imshow(roi_mpc_map, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
        fig.colorbar(roi_mpc_img, ax=ax, label='MPC')
        plt.title('Region-based MPC Map')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        roi_mpc_map_path = os.path.join(tempfile.gettempdir(), 'roi_mpc_map.png')
        plt.savefig(roi_mpc_map_path)
        plt.close(fig)
        st.image(roi_mpc_map_path, use_column_width=True)

    # New row for Wave Speed and Direction
    col5, col6 = st.columns(2)

    # Visualize Wave Speed
    with col5:
        st.markdown("<h5 style='text-align: center;'>Wave Speed Map</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        wave_speed_img = ax.imshow(global_wave_speed_normalized, cmap='jet', interpolation='nearest')
        fig.colorbar(wave_speed_img, ax=ax, label='Speed (normalized)')
        plt.title('Wave Speed Map')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        wave_speed_map_path = os.path.join(tempfile.gettempdir(), 'wave_speed_map.png')
        plt.savefig(wave_speed_map_path)
        plt.close(fig)
        st.image(wave_speed_map_path, use_column_width=True)

    # Visualize Wave Direction
    with col6:
        st.markdown("<h5 style='text-align: center;'>Wave Direction Map</h5>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        wave_direction_img = ax.imshow(global_wave_direction, cmap='hsv', interpolation='nearest')
        fig.colorbar(wave_direction_img, ax=ax, label='Direction (radians)')
        plt.title('Wave Direction Map')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        wave_direction_map_path = os.path.join(tempfile.gettempdir(), 'wave_direction_map.png')
        plt.savefig(wave_direction_map_path)
        plt.close(fig)
        st.image(wave_direction_map_path, use_column_width=True)

    # Display summary statistics
    st.markdown("<h5 style='text-align: center;'>Summary Statistics</h5>", unsafe_allow_html=True)
    st.write("### Global Statistics")
    st.write(global_stats)
    st.write("### Region-based Statistics")
    st.write(roi_stats)

    # Additional statistics for wave speed and direction
    st.write("### Wave Speed and Direction Statistics")
    st.write({
        "Mean Wave Speed (pixels/frame)": wave_speed_mean,
        "Std Wave Speed (pixels/frame)": wave_speed_std,
        "Mean Wave Direction (radians)": wave_direction_mean,
        "Std Wave Direction (radians)": wave_direction_std
    })





























