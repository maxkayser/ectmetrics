# ECTMetrics

ECTMetrics is a Python library for analyzing EEG signals, particularly focusing on seizure metrics. It provides functionalities to generate synthetic EEG signals, visualize them, and calculate various metrics related to seizure activity.

## Features

- Generate synthetic ECT specific EEG signals with customizable parameters.
- Visualize EEG signals for better understanding and analysis.
- Import EEG data from GPT

- Calculate various seizure metrics, including:
  - **Average Seizure Energy Index (ASEI)**
  - **Seizure Energy Index (SEI)**
  - **Postictal Suppressiom Index (PSI)**
  - **Earlyictal Amplitude (EIA)**
  - **Midictal Amplitude (MIA)**
  - **Maximum Sustained Power (MSP)**
  - **Time to Peak Power (TTPP)**
  - **Maximum Sustained Coherence (COH)**
  - **Time to Peak Coherence (TTPC)**
  
  
## Installation

To install the `ectmetrics` library, clone the repository and use pip to install the dependencies:

```bash
git clone https://github.com/maxkayser/ECTMetrics
cd ectmetrics
pip install -r requirements.txt```



## Running Tests

To run the tests, you will need pytest. Install it via pip if you haven’t already:

pip install pytest

Then run the tests with:

pytest tests/