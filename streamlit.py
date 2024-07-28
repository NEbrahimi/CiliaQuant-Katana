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
    magnitude_path = os.path.join(storage_path, 'magnitude_map.png')
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
    # video_source = st.radio("Select Video Source", options=['Original', 'Masked'], index=0,
    #                         help="Choose whether to use the original or masked video for analysis.")
    freq_filter_min = st.number_input("Frequency Filter Min", value=2, help="Minimum frequency for CBF analysis.")
    freq_filter_max = st.number_input("Frequency Filter Max", value=30, help="Maximum frequency for CBF analysis.")
    run_step_3 = st.button("Run Step 3")

    # Sidebar Input for Step 4
    st.title("Step 4: Grid Analysis")
    grid_size = st.number_input("Grid Size", min_value=1, max_value=20, value=7, step=1,
                                help="Set the size of the grid.")
    run_step_4 = st.button("Run Step 4")

    # # Sidebar Input for Step 5
    # st.title("Step 5: Cilia Beat Coordination Analysis")
    # coordination_video_source = st.radio("Select Video Source", options=['Original', 'Masked'], index=0,
    #                                      help="Choose whether to use the original or masked video for analysis for coordination.")
    # grid_size = st.number_input("Grid Size for Coordination Analysis", min_value=1, max_value=20, value=10, step=1)
    # n_clusters = st.number_input("Number of Clusters for Coordination Analysis", min_value=1, max_value=10, value=5,
    #                              step=1)
    # run_step_5 = st.button("Run Step 5")


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
storage_path = r"/home/z3541106/ondemand/dev/CiliaQuant-Katana/storage"
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

    original_video_permanent_path = os.path.join(storage_path, 'original_video.mp4')
    masked_video_permanent_path = os.path.join(storage_path, 'masked_video.mp4')
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
        resized_magnitude_path = os.path.join(storage_path, 'resized_magnitude_map.png')
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

    if 'mask_path' in st.session_state and os.path.exists(st.session_state['mask_path']):
        video_path = st.session_state['masked_video_permanent_path']
        mask_path = st.session_state['mask_path']
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            coverage_percent = np.count_nonzero(mask_img) / mask_img.size * 100
        else:
            st.error("Failed to read the mask image. Check the file at the specified path.")
            coverage_percent = "N/A"  # In case mask image is not readable
    else:
        st.error("Mask path not found. Please complete the necessary steps to generate the mask.")
        coverage_percent = "N/A"

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
            # data = {
            #     'Measurement': [
            #         'Name', 'CFB (power) (Hz)', 'CBF (amplitude) (Hz)', 'Mean Frequency (Hz)',
            #         'Median Frequency (Hz)', 'Standard Deviation', '25th Percentile', '75th Percentile',
            #         'Coverage %*'
            #     ],

            data = {
                'Measurement': [
                    'Name', 'CFB (power) (Hz)', 'Mean Frequency (Hz)',
                    'Median Frequency (Hz)', 'Standard Deviation', '25th Percentile', '75th Percentile',
                    'Coverage %*'
                ],
                'Value': [
                    uploaded_file.name,
                    f"{max_power_freq:.2f}",
                    # f"{max_amplitude_freq:.2f}",
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


























