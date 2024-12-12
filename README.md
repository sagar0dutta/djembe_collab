# Dance Motion Capture and Onset Analysis

This repository provides tools and methods to process dance motion capture data, extract musical and movement onsets, tempos, and analyze various metrics related to music and dance.

## Overview

The main functionalities of this repository include:

1. **Processing Motion Capture Data**:
   - Extract positional, velocity, and acceleration data for various body segments.
   - Segment-specific data access for selected sections in a dance performance.

2. **Onset and Tempo**:
   - Extract cycle and beat onsets from annotated data.
   - Calculate cycle periods, windowed onset metrics, and tempo (BPM) for different dance sections.

3. **Music Data**:
   - Process and analyze drum onset data and dance annotations.

4. **Section-wise Analysis**:
  - Enables breakdown of motion capture data by specific sections of a dance performance.
  - Extracts metadata like start time, end time, duration, category etc. for each section.

---

## Getting Started

### Prerequisites

- Python 3.12.2
- Required libraries: `numpy`, `pandas`

### Installation

Clone the repository:
```bash
git clone https://github.com/sagar0dutta/djembe_collab.git
cd djembe_collab
```

Install dependencies:
```bash
pip install -r requirements.txt (or install manually)
```

---

## Usage

### 1. Load and Process Data
Use the `read_data()` function to load and process motion capture data:
```python
fname, motion_data, drum_df, dance_annotation_df, metric_df, section_data, bpm_values = read_data(select_idx=3, mode="in")
```

- `select_idx`: Index of the file to process from the dataset.
- `mode`: Dance mode (`"in"`: individual, `"gr"`: group, `"au"`: audience).

### 2. Extract Section Data
Analyze a specific section of the performance:
```python
section = section_data[section_idx]
section_meta_data = section["section_meta_data"]
section_onset_data = section["section_onset_data"]
```

### 3. Access Motion Data
Extract position, velocity, and other data for a selected segment and section:
```python
position_data = motion_data['position']['SEGMENT_RIGHT_HAND'][start_f:end_f, :]
velocity_data = motion_data['velocity']['SEGMENT_RIGHT_HAND'][start_f:end_f, :]
```

---

## Data Details

### Body Segment Names
The dataset supports the following segments:
- `SEGMENT_PELVIS`, `SEGMENT_HEAD`, `SEGMENT_RIGHT_HAND`, `SEGMENT_LEFT_HAND`, `SEGMENT_RIGHT_FOOT`, `SEGMENT_LEFT_FOOT`, and more in the notebook.

### Motion Data Types
Each segment includes:
- **Position**: `motion_data['position']`
- **Velocity**: `motion_data['velocity']`
- **Acceleration**: `motion_data['acceleration']`
- **Orientation**: `motion_data['orientation']`
- **Angular Velocity**: `motion_data['angular_velocity']`

---

## Example Output

### Section Metadata:
```plaintext
Piece: dance_example
Section: Section 3
Start (sec): 12.34
End (sec): 45.67
Category: group
Duration (sec): 33.33
```

### Onset Data:
```plaintext
Cycle Onsets: [0.5, 1.0, 1.5, 2.0, ...]
Cycle Period List: [0.5, 0.5, 0.5, ...]
BPM: [120, 123, 125, ...]
Total Cycles: 10
```


---

Feel free to modify the structure and content based on your specific needs!
