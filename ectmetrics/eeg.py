import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pyedflib

# Default parameters for EEG signal processing
DEFAULTS = {
    'signal_duration': 28,                  # Duration of the signal in seconds
    'sampling_frequency': 200,        # Sampling rate in Hz
    'stim_time': 0,                             # Time of stimulation in seconds
    'stim_duration_ms': 1000,          # Duration of stimulation in milliseconds
    'seizure_duration': 21,                # Duration of the seizure in seconds
    'add_noise': True,                      # Add background noise
    'immediate_seizure': True,         # Set to True to start seizure immediately
    'filters': {                                     # Specify which filters to apply
        'notch': {
            'apply': True,
            'options': {
                'notch_freq': 60
            }
        },
        'lowpass': {
            'apply': True,
            'options': {
                'cutoff_freq': 30
            }
        },
        'bandpass': {
            'apply': True,
            'options': {
                'lowcut': 1,
                'highcut': 40
            }
        }
    }
}

"""
Default settings for EEG signal generation.

This dictionary contains default parameters that can be used to generate 
simulated EEG signals. Each key represents a specific setting, which can 
be adjusted according to the needs of the user.

Keys:
- `signal_duration`: int
    Total duration of the signal in seconds.
- `sampling_frequency`: int
    Sampling rate in Hz.
- `stim_time`: int
    Time of stimulation in seconds.
- `stim_duration_ms`: int
    Duration of stimulation in milliseconds.
- `seizure_duration`: int
    Duration of the seizure in seconds.
- `add_noise`: bool
    Whether to add background noise.
- `immediate_seizure`: bool
    Whether to start the seizure immediately or after stimulation.
- `filters`: dict
    Dictionary specifying which filters to apply.
"""
# Define templates
templates = {
    'eeg': {
        'name': None,
        'signals': None,
        'channels': [],
        'x-axis': [],
        'sampling_frequency': None,
        'timepoints': {
                'stim_startpoint': None,
                'seizure_startpoint': None,
                'seizure_endpoint': None
        },
        'filters': None,
    }
}
        
def generate(signal_duration=DEFAULTS['signal_duration'], sampling_frequency=DEFAULTS['sampling_frequency'], stim_time=DEFAULTS['stim_time'], stim_duration_ms=DEFAULTS['stim_duration_ms'], seizure_duration=DEFAULTS['seizure_duration'], add_noise=DEFAULTS['add_noise'], immediate_seizure=DEFAULTS['immediate_seizure'], filters=DEFAULTS['filters'], eeg_name=None):
 
    """
    Generate a simulated EEG signal with optional stimulation, seizure, and postictal phases for two channels.

    Parameters
    ----------
    signal_duration : float, optional
        Total duration of the signal in seconds. Default is ``DEFAULTS['signal_duration']``.
    sampling_frequency : int, optional
        Sampling rate in Hz. Default is ``DEFAULTS['sampling_frequency']``.
    stim_time : float, optional
        Time of stimulation in seconds. Default is ``DEFAULTS['stim_time']``.
    stim_duration_ms : int, optional
        Duration of stimulation in milliseconds. Default is ``DEFAULTS['stim_duration_ms']``.
    seizure_duration : float, optional
        Duration of the seizure in seconds. Default is ``DEFAULTS['seizure_duration']``.
    add_noise : bool, optional
        Whether to add background noise. Default is ``DEFAULTS['add_noise']``.
    immediate_seizure : bool, optional
        Whether to start the seizure immediately or after stimulation. Default is ``DEFAULTS['immediate_seizure']``.
    filters : dict, optional
        Dictionary specifying which filters to apply. Filters are defined as follows:
        - 'bandpass': {'apply': bool, 'options': {'highcut': float, 'lowcut': float}}
        - 'lowpass': {'apply': bool, 'options': {'cutoff_freq': float}}
        - 'notch': {'apply': bool, 'options': {'notch_freq': float}}
        Default is ``DEFAULTS['filters']``.
    eeg_name : str, optional
        Name of the EEG file to save, if provided. Default is None.

    Returns
    -------
    t : numpy.ndarray
        Time array corresponding to the EEG signal.
    eeg_signals : numpy.ndarray
        2D numpy array containing the simulated EEG signals for both channels.
    stim_startpoint : int
        The index where stimulation starts.
    seizure_startpoint : int
        The index where the seizure starts.
    seizure_endpoint : int
        The index where the seizure ends.

    Notes
    -----
    The function generates an EEG signal with customizable noise, filters, and event phases (stimulation, seizure, postictal). 
    Filtering options include bandpass, lowpass, and notch filters that can be applied based on the configuration in the 
    `filters` dictionary.

    Example
    -------
    >>> eeg.generate(signal_duration=28, sampling_frequency=200, stim_time=5, 
                     stim_duration_ms=500, seizure_duration=10, add_noise=True, 
                     immediate_seizure=False, filters={'bandpass': {'apply': True, 
                     'options': {'highcut': 40, 'lowcut': 1}}, 'lowpass': {'apply': True, 
                     'options': {'cutoff_freq': 30}}, 'notch': {'apply': True, 
                     'options': {'notch_freq': 60}}}, eeg_name='example_eeg')
    """
    
    t = np.arange(0, signal_duration, 1/sampling_frequency)  # Time axis
    signal1 = np.zeros_like(t)  # Initialize signal for channel 1
    signal2 = np.zeros_like(t)  # Initialize signal for channel 2
    
    # Default values for stimulation and seizure start/end times
    stim_startpoint = None
    seizure_startpoint = None
    seizure_endpoint = None

    # Background noise (Gaussian noise) in microvolts
    if add_noise:
        noise1 = np.random.normal(0, 50, t.shape)  # Background noise for channel 1
        noise2 = np.random.normal(0, 50, t.shape)  # Background noise for channel 2
    else:
        noise1 = np.zeros_like(t)
        noise2 = np.zeros_like(t)

    if immediate_seizure:
        # Seizure simulation
        seizure_startpoint = 0
        seizure_endpoint = seizure_startpoint + int(seizure_duration * sampling_frequency)
        
        # Channel 1 seizure wave
        seizure_wave1 = (
            200 * np.sin(2 * np.pi * 20 * t[seizure_startpoint:seizure_endpoint]) +
            100 * np.sin(2 * np.pi * 40 * t[seizure_startpoint:seizure_endpoint])
        )
        signal1[seizure_startpoint:seizure_endpoint] += seizure_wave1
        
        # Channel 2 seizure wave (different frequency)
        seizure_wave2 = (
            150 * np.sin(2 * np.pi * 25 * t[seizure_startpoint:seizure_endpoint]) +
            80 * np.sin(2 * np.pi * 35 * t[seizure_startpoint:seizure_endpoint])
        )
        signal2[seizure_startpoint:seizure_endpoint] += seizure_wave2
        
        # Suppression starts immediately after the seizure
        suppression_startpoint = seizure_endpoint
        suppression_wave = 50 * np.exp(-0.01 * (t[suppression_startpoint:] - t[suppression_startpoint]))  # Exponential decay
        signal1[suppression_startpoint:] -= suppression_wave
        signal2[suppression_startpoint:] -= suppression_wave  # Same suppression effect for both channels
        
    else:
        # Stimulation duration in seconds
        stim_duration = stim_duration_ms / 1000.0

        # Simulate stimulation
        stim_startpoint = int(stim_time * sampling_frequency)
        stim_endpoint = stim_startpoint + int(stim_duration * sampling_frequency)  # Duration of stimulation in seconds
        
        # Channel 1 stimulation wave
        stim_wave1 = 100 * np.sin(2 * np.pi * 10 * (t[stim_startpoint:stim_endpoint] - stim_time))
        signal1[stim_startpoint:stim_endpoint] += stim_wave1

        # Channel 2 stimulation wave (different frequency)
        stim_wave2 = 80 * np.sin(2 * np.pi * 12 * (t[stim_startpoint:stim_endpoint] - stim_time))
        signal2[stim_startpoint:stim_endpoint] += stim_wave2

        # Simulate seizure
        seizure_startpoint = stim_endpoint + int(2 * sampling_frequency)  # Seizure starts 2 seconds after stimulation
        seizure_endpoint = seizure_startpoint + int(seizure_duration * sampling_frequency)  # Seizure lasts `seizure_duration` seconds

        # Channel 1 seizure wave
        seizure_wave1 = (
            200 * np.sin(2 * np.pi * 20 * t[seizure_startpoint:seizure_endpoint]) +
            100 * np.sin(2 * np.pi * 40 * t[seizure_startpoint:seizure_endpoint])
        )
        signal1[seizure_startpoint:seizure_endpoint] += seizure_wave1

        # Channel 2 seizure wave (different frequency)
        seizure_wave2 = (
            150 * np.sin(2 * np.pi * 25 * t[seizure_startpoint:seizure_endpoint]) +
            80 * np.sin(2 * np.pi * 35 * t[seizure_startpoint:seizure_endpoint])
        )
        signal2[seizure_startpoint:seizure_endpoint] += seizure_wave2

        # Suppression starts immediately after the seizure
        suppression_startpoint = seizure_endpoint
        suppression_wave = 50 * np.exp(-0.01 * (t[suppression_startpoint:] - t[suppression_startpoint]))  # Exponential decay
        signal1[suppression_startpoint:] -= suppression_wave
        signal2[suppression_startpoint:] -= suppression_wave  # Same suppression effect for both channels

    # Add background noise if enabled
    signal1 += noise1
    signal2 += noise2

    # Adding occasional artifacts but excluding the postictal phase
    num_artifacts = 5
    for _ in range(num_artifacts):
        artifact_startpoint = np.random.randint(0, suppression_startpoint - int(0.5 * sampling_frequency))
        artifact_endpoint = artifact_startpoint + int(0.5 * sampling_frequency)  # Artifact lasts 0.5 seconds
        artifact = 200 * np.random.normal(0, 1, int(0.5 * sampling_frequency))  # Artifact with 200 µV
        signal1[artifact_startpoint:artifact_endpoint] += artifact
        signal2[artifact_startpoint:artifact_endpoint] += artifact  # Same artifact for both channels

    # Check if filters are provided; if not, use defaults
    filters = filters if filters is not None else DEFAULTS['filters']

    # Apply filters if specified
    if filters is not None:
        # Check if notch filter should be applied
        if filters['notch']['apply']:
            options = get_filter_options(filters['notch'].get('options'), DEFAULTS['filters']['notch']['options'])
            signal1 = notch_filter(signal1, sampling_frequency, options)
            signal2 = notch_filter(signal2, sampling_frequency, options)

        # Check if lowpass filter should be applied
        if filters['lowpass']['apply']:
            options = get_filter_options(filters['lowpass'].get('options'), DEFAULTS['filters']['lowpass']['options'])
            signal1 = lowpass_filter(signal1, sampling_frequency, options)
            signal2 = lowpass_filter(signal2, sampling_frequency, options)

        # Check if bandpass filter should be applied
        if filters['bandpass']['apply']:
            options = get_filter_options(filters['bandpass'].get('options'), DEFAULTS['filters']['bandpass']['options'])
            signal1 = bandpass_filter(signal1, sampling_frequency, options)
            signal2 = bandpass_filter(signal2, sampling_frequency, options)

    # Convert EEG signals to multi-dimensional numpy array
    eeg_signals_list = [np.array(signal1), np.array(signal2)]
    eeg_signals = np.array(eeg_signals_list)
    
    # Define EEG channels variable
    eeg_channels = [0, 1]
    
    # Load EEG template
    eeg = templates['eeg'].copy()
    
    # Set EEG variables
    eeg['name'] = eeg_name
    eeg['signals'] = eeg_signals
    eeg['channels'] = eeg_channels
    eeg['x-axis'] = t
    eeg['sampling_frequency'] = sampling_frequency
    eeg['timepoints']['stim_startpoint'] = stim_startpoint
    eeg['timepoints']['seizure_startpoint'] = seizure_startpoint
    eeg['timepoints']['seizure_endpoint'] = seizure_endpoint
    eeg['filters'] = filters
    
    return eeg

# Define a function to merge user-defined filter options with defaults
def get_filter_options(user_options, default_options):
    # Create a copy of the default options to avoid mutating it
    options = default_options.copy()
    # Update with user-defined options, if any
    if user_options is not None:
        options.update(user_options)
    return options

def notch_filter(signal, sampling_frequency, options=None):
    
    #Define notch_freq variable
    if options is not None and 'notch_freq' in options:
        notch_freq = options['notch_freq']
    else:
        notch_freq  = DEFAULTS['filters']['notch']['options']['notch_freq']
        
    nyquist = 0.5 * sampling_frequency
    low = (notch_freq - 0.5) / nyquist
    high = (notch_freq + 0.5) / nyquist
    b, a = butter(2, [low, high], btype='bandstop')
    return filtfilt(b, a, signal)

def lowpass_filter(signal, sampling_frequency, options=None):

    #Define cutoff_freq variable
    if options is not None and 'cutoff_freq' in options:
        cutoff_freq = options['cutoff_freq']
    else:
        cutoff_freq  = DEFAULTS['filters']['lowpass']['options']['cutoff_freq']
        
    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

def bandpass_filter(signal, sampling_frequency, options=None):

    #Define lowcut variable
    if options is not None and 'lowcut' in options:
        lowcut = options['lowcut']
    else:
        lowcut  = DEFAULTS['filters']['bandpass']['options']['lowcut']
        
    #Define highcut variable
    if options is not None and 'highcut' in options:
        highcut = options['highcut']
    else:
        highcut  = DEFAULTS['filters']['bandpass']['options']['highcut']
        
    #Validate input data
    if not isinstance(signal, np.ndarray):
        raise ValueError("Signal must be a numpy array.")
    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")
    if sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be positive.")
    nyquist = 0.5 * sampling_frequency
    if not (0 < lowcut < nyquist) or not (0 < highcut < nyquist):
        raise ValueError(f"Cutoff frequencies must be between 0 and {nyquist} Hz.")
        
    nyquist = 0.5 * sampling_frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)

def import_eeg(file_path, name=None):
    """
    Import EEG data from an EDF file and return it as a dictionary.

    Parameters
    ----------
    file_path : str
        The path to the EDF file containing the EEG data.
    name : str, optional
        Name for the EEG data. If None, the first signal label from the file will be used as the name.

    Returns
    -------
    eeg : dict
        A dictionary containing the EEG data with the following keys:
        - 'name' : str
            The name assigned to the EEG data (either provided or the first signal label).
        - 'signals' : numpy.ndarray
            A 2D array containing the EEG signal data for all channels (shape: [n_channels, n_samples]).
        - 'channels' : list of int
            A list of channel indices corresponding to the signals.
        - 'x-axis' : numpy.ndarray
            Time vector corresponding to the EEG signal data.
        - 'sampling_frequency' : float
            The sampling frequency of the EEG signals in Hz.

    Raises
    ------
    FileNotFoundError
        If the specified `file_path` does not exist.
    ValueError
        If the EDF file is invalid or does not contain EEG data.

    Notes
    -----
    This function requires the `pyedflib` library to read EDF files. Ensure that the library is installed 
    in your environment. The time vector is generated based on the number of samples and the sampling frequency.

    Examples
    --------
    >>> eeg = import_eeg('path/to/eeg_data.edf')
    >>> print(eeg['signals'])  # Access the EEG signal data
    >>> print(eeg['sampling_frequency'])  # Access the sampling frequency
    """
    # Open the EDF file
    with pyedflib.EdfReader(file_path) as f:
        # Extract basic information
        n_channels = f.signals_in_file
        signal_labels = f.getSignalLabels()
        sampling_frequency = f.getSampleFrequency(0)  # Assuming all channels have the same frequency
        total_samples = f.getNSamples()[0]

        # Initialize arrays for signals and time
        eeg_signals = np.zeros((n_channels, total_samples))
        for i in range(n_channels):
            eeg_signals[i, :] = f.readSignal(i)
        
        # Generate time vector based on the number of samples and sampling frequency
        x_axis = np.arange(total_samples) / sampling_frequency

        # Use the provided name or take from the first signal label if name is None
        eeg_name = name if name is not None else (signal_labels[0] if signal_labels else None)
        
        # Load EEG template
        eeg = templates['eeg'].copy()
        
        # Set EEG variables
        eeg['name'] = eeg_name
        eeg['signals'] = eeg_signals
        eeg['channels'] = list(range(n_channels))
        eeg['x-axis'] = x_axis
        eeg['sampling_frequency'] = sampling_frequency

    return eeg
   

def plot(eeg, metrics=None):

    """
    Plot two-channel EEG signals with stimulation and seizure markers.

    Parameters
    ----------
    eeg : dict
        A dictionary containing EEG data with the following keys:
        - 'name' : str
            The name of the EEG dataset.
        - 'signals' : numpy.ndarray
            A 2D array containing EEG signal data for all channels (shape: [n_channels, n_samples]).
        - 'x-axis' : numpy.ndarray
            Time vector corresponding to the EEG signal data.
        - 'timepoints' : dict
            A dictionary containing the following keys:
            - 'stim_startpoint' : int
                The sample index where stimulation starts.
            - 'seizure_startpoint' : int
                The sample index where the seizure starts.
            - 'seizure_endpoint' : int
                The sample index where the seizure ends.
        - 'sampling_frequency' : float
            The sampling frequency of the EEG signals in Hz.

    metrics : list of dict, optional
        A list of metrics to plot on the EEG signals. Each metric dictionary should contain:
        - 'name' : str
            The name of the metric.
        - 'timepoints' : dict
            A dictionary with either of the following:
            - 'startpoint' : int
                The sample index where the metric starts.
            - 'endpoint' : int
                The sample index where the metric ends.
            - 'timepoint' : int
                The sample index of a specific point in time related to the metric.

    Example
    -------
    >>> eeg_data = import_eeg('path/to/eeg_data.edf')  # Assuming this function is defined
    >>> metrics = [{'name': 'Metric 1', 'timepoints': {'startpoint': 500, 'endpoint': 600}}]
    >>> plot(eeg_data, metrics)

    Notes
    -----
    This function requires the `matplotlib` library for plotting. Ensure that the library is installed 
    in your environment.
    """

    name = eeg.get('name')
    eeg_signal = eeg['signals']
    signal1 = eeg_signal[0]
    signal2 = eeg_signal[1]
    t = eeg['x-axis']
    
    stim_startpoint = eeg['timepoints']['stim_startpoint']  # Assuming these keys are directly under the eeg dictionary
    seizure_startpoint = eeg['timepoints']['seizure_startpoint']
    seizure_starttime = seizure_startpoint / eeg['sampling_frequency']
    seizure_endpoint = eeg['timepoints']['seizure_endpoint']
    seizure_endtime = seizure_endpoint / eeg['sampling_frequency']
    
    plt.figure(figsize=(12, 5)) 

    # Channel 1
    plt.subplot(2, 1, 1)
    plt.plot(t, signal1, label='EEG Signal (Channel 1)', color='#0d82d6')
    plt.axvline(x=seizure_starttime, color='orange', linestyle='--', label='Seizure Start')
    plt.axvline(x=seizure_endtime, color='purple', linestyle='--', label='Seizure End')
    plt.title(f'{name} - Channel 1' if name else 'Channel 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    # Channel 2
    plt.subplot(2, 1, 2)
    plt.plot(t, signal2, label='EEG Signal (Channel 2)', color='#0d82d6')
    plt.axvline(x=seizure_starttime, color='orange', linestyle='--', label='Seizure Start')
    plt.axvline(x=seizure_endtime, color='purple', linestyle='--', label='Seizure End')
    plt.title(f'{name} - Channel 2' if name else 'Channel 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
      
    # Plot metrics if provided
    if metrics is not None:
        for metric in metrics:
            timepoints = metric['timepoints']

            # Check for startpoint and endpoint
            if 'startpoint' in timepoints and 'endpoint' in timepoints:
                # Calculate start and end times
                start_time = timepoints['startpoint'] / eeg['sampling_frequency']
                end_time = timepoints['endpoint'] / eeg['sampling_frequency']
                
                # Plot vertical lines in Channel 1
                plt.subplot(2, 1, 1)
                plt.axvline(x=start_time, color='gray', linestyle=':', label=f'{metric["name"]} Start')
                plt.axvline(x=end_time, color='gray', linestyle=':', label=f'{metric["name"]} End')

                # Plot vertical lines in Channel 2
                plt.subplot(2, 1, 2)
                plt.axvline(x=start_time, color='gray', linestyle=':', label=f'{metric["name"]} Start')
                plt.axvline(x=end_time, color='gray', linestyle=':', label=f'{metric["name"]} End')

            # Check for timepoint
            elif 'timepoint' in timepoints:
                time_point = timepoints['timepoint'] / eeg['sampling_frequency']
                
                # Plot vertical line in Channel 1
                plt.subplot(2, 1, 1)
                plt.axvline(x=time_point, color='green', linestyle=':', label=f'{metric["name"]} Timepoint')

                # Plot vertical line in Channel 2
                plt.subplot(2, 1, 2)
                plt.axvline(x=time_point, color='green', linestyle=':', label=f'{metric["name"]} Timepoint')


    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()  # Show the plot
