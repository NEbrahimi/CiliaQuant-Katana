# CiliaQuant-Katana

This repository contains the CiliaQuant application tailored for deployment on the UNSW Katana HPC system. CiliaQuant is a tool designed for the analysis of ciliary beat frequency (CBF) and coordination in microscopic videos, specifically optimized to work in a Linux environment. This version of CiliaQuant includes modifications for path compatibility, environment setup scripts, and module loading required for the Katana system.

## Features

- **Ciliary Beat Frequency (CBF) Analysis**: Calculate and visualize the ciliary beat frequency using Fourier Transform and other signal processing techniques.
- **Cilia Coordination Analysis**: Measure and visualize the coordination and synchronization of ciliary movement.
- **Wave Properties Calculation**: Determine wave speed and direction using phase gradient methods.
- **Grid-Based Analysis**: Perform detailed analysis by dividing the video into grids to study local variations.
- **Mask Application**: Apply masks to videos for focused analysis on specific regions of interest.
- **High-Quality Visualizations**: Generate detailed plots and heatmaps for in-depth analysis.

## Setup

To set up the environment and run the CiliaQuant application on Katana:

### Clone the Repository

```bash
cd $HOME
git clone https://github.com/NEbrahimi/CiliaQuant-Katana.git
cd CiliaQuant-Katana
