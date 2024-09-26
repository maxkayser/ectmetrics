##Usage

#Importing the library

import ectmetrics
from ectmetrics.eeg import generate, plot
from ectmetrics.metrics import calculate_metrics

#Generating an EEG Signal
You can create an EEG signal by specifying parameters for the EEG signal.

eeg = generate(
    signal_duration=28,      # Duration of the EEG signal in seconds
    seizure_duration=21,     # Duration of the seizure in seconds
    sampling_frequency=200,  # Sampling frequency of the signal in Hz
    eeg_name='My EEG'        # Name or identifier for the EEG
)

#Visualizing the Signals
Use the plot function to visualize the generated EEG signals.

plot(eeg)

#Calculating Seizure Quality Metrics
To calculate various metrics to analyze the seizure quality, use the calculate_metrics function.

metrics_results = calculate_metrics(eeg)
print(metrics_results)


Complete workflow example
Here’s a complete example of generating an EEG signal, visualizing it, and calculating the seizure metrics.

import ectmetrics
from ectmetrics.eeg import generate, plot
from ectmetrics.metrics import calculate_metrics

eeg = generate(
    signal_duration=28,
    seizure_duration=21,
    sampling_frequency=200,
    eeg_name='My EEG'
)

metrics_results = calculate_metrics(eeg)

plot(eeg, metrics_results)

metrics_results