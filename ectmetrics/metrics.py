import numpy as np
from scipy import signal
from scipy.fft import fft

# Default parameters
DEFAULTS = {
    'segment_length': 256
}

def calculate_asei(eeg_signals, eeg_channel, sampling_frequency, seizure_startpoint, seizure_endpoint, debug=False):
    """
    Calculate the Average Seizure Energy Index (ASEI) for a given signal.
    
    Parameters:
    eeg_signals: numpy.ndarray - A 2D numpy array containing the EEG signals.
    eeg_channel: int - The channel index to analyze.
    sampling_frequency: float - Sampling frequency of the EEG signal in Hz.
    seizure_startpoint: float - Start point of the seizure in seconds.
    seizure_endpoint: float - End point of the seizure in seconds.
    debug: bool - If True, print debug information.
    
    Returns:
    dict: A dictionary containing the calculated ASEI metric.
    """
    
    if debug:
        print("> Calculating ASEI")

    try:
        
        asei_startpoint = seizure_startpoint
        asei_endpoint = seizure_endpoint
        
        # Extract the seizure segment
        segment = eeg_signals[eeg_channel][asei_startpoint:asei_endpoint]
        
        # Calculate ASEI
        average_energy = np.mean(np.square(segment))
        asei = average_energy 
        
        if asei is not None:
            asei_result = {
                'name': 'asei', 
                'value': asei, 
                'timepoints': {
                    'startpoint': asei_startpoint,
                    'endpoint': asei_endpoint,
                },
                'unit': 'μV²',
                'description': 'Average Seizure Energy Index'
            }
            
        if debug:
            print(f"Calculated ASEI: {asei_result}")

        return asei_result

    except Exception as e:
        print(f"An error occurred during calculation of ASEI: {e}")
        return None

def calculate_sei(eeg_signals, eeg_channel, sampling_frequency, segment_length, seizure_startpoint, seizure_endpoint, asei=None, debug=False):

    """
    Calculate the Seizure Energy Index (SEI) based on the ASEI.
    
    Parameters:
    eeg_signals: numpy.ndarray - A 2D numpy array containing the EEG signals.
    eeg_channel: int - The channel index to analyze.
    sampling_frequency: float - Sampling frequency of the EEG signal in Hz.
    segment_length: float - Length of the segment in seconds.
    seizure_startpoint: float - Start point of the seizure in seconds.
    seizure_endpoint: float - End point of the seizure in seconds.
    asei: float - Average Seizure Energy Index in microvolts squared.
    debug: bool - If True, print debug information.
    
    Returns:
    dict: A dictionary containing the calculated SEI metric.
    """
    
    if debug:
        print("> Calculating SEI...")
    
    try:
        
        # Validate inputs
        if sampling_frequency is None or sampling_frequency <= 0:
            print("Error: Invalid sampling frequency.")
            return None

        if seizure_startpoint is None or seizure_endpoint is None:
            print("Error: Seizure startpoint or endpoint is None.")
            return None

        if seizure_startpoint >= seizure_endpoint:
            print("Error: Seizure startpoint must be less than seizure endpoint.")
            return None
            
        sei_startpoint = seizure_startpoint
        sei_endpoint = seizure_endpoint
    
        # Calculate ASEI if not provided
        if asei is None:
            asei_result = calculate_asei(eeg_signals, eeg_channel, sampling_frequency, seizure_startpoint, seizure_endpoint, debug)
            asei = asei_result['value']
            
        # Calculate seizure duration in seconds
        seizure_duration = (sei_endpoint - sei_startpoint) / sampling_frequency
       
        if debug:
            print(f">> Seizure duration: {seizure_duration}s")
    
        # Calculate SEI
        sei = asei * seizure_duration if asei is not None else None
        
        if sei is not None:
            sei_result = {
                'name': 'sei', 
                'value': sei, 
                'timepoints': {
                    'startpoint': sei_startpoint,
                    'endpoint': sei_endpoint,
                },
                'unit': 'μV²·s',
                'description': 'Seizure Energy Index'
            }
            
        if debug:
            print(f">> Calculated SEI: {sei_result}")
        
        return sei_result

    except Exception as e:
        print(f">> An error occurred during calculation of SEI: {e}")
        return None


def calculate_psi(eeg_signals, eeg_channel, sampling_frequency, segment_length, seizure_endpoint, debug=False):
    """
    Calculate the Postictal Suppression Index (PSI) for a given signal.

    Parameters:
    eeg_signals: numpy.ndarray - A 2D numpy array containing the EEG signals.
    eeg_channel: int - The channel index to analyze.
    sampling_frequency: float - Sampling frequency of the EEG signal in Hz.
    segment_length: int - Length of each segment in samples.
    seizure_endpoint: float - End time of the seizure in seconds.
    debug: bool - If True, print debug information.

    Returns:
    dict: A dictionary containing the calculated PSI metric.
    """
    
    if debug:
        print("Calculating PSI...")
    
    try:
        # Define sample points
        psi_ictal_start_sample = int(seizure_endpoint - (4.5 * segment_length))
        psi_ictal_end_sample = int(seizure_endpoint - (1.5 * segment_length))
        psi_nonictal_start_sample = int(seizure_endpoint + (1.5 * segment_length))
        psi_nonictal_end_sample = int(seizure_endpoint + (4.5 * segment_length))
        
        # Ensure indices are within the bounds of the EEG signal
        if (psi_ictal_start_sample < 0 or psi_ictal_end_sample > len(eeg_signals[eeg_channel]) or 
            psi_nonictal_start_sample < 0 or psi_nonictal_end_sample > len(eeg_signals[eeg_channel])):
            raise ValueError("Calculated indices are out of bounds.")
            
        # Extract segments
        psi_ictal_eeg_signal = eeg_signals[eeg_channel][psi_ictal_start_sample:psi_ictal_end_sample]
        psi_nonictal_eeg_signal = eeg_signals[eeg_channel][psi_nonictal_start_sample:psi_nonictal_end_sample]

        # Calculate absolute mean amplitudes
        psi_ictal_abs_mean = abs(psi_ictal_eeg_signal).mean()
        psi_nonictal_abs_mean = abs(psi_nonictal_eeg_signal).mean()

        # Calculate PSI
        psi = (1 - (psi_nonictal_abs_mean / psi_ictal_abs_mean)) * 100 if psi_ictal_abs_mean > 0 else np.nan
        
        if psi is not None:
            psi_result = {
                'name': 'psi', 
                'value': psi, 
                'timepoints': {
                    'startpoint': psi_ictal_start_sample,
                    'endpoint': psi_nonictal_end_sample,
                },
                'unit': '%',
                'description': 'Postictal Suppression Index'
            }
            
        if debug:
            print(f"Calculated PSI: {psi_result}")

        return psi_result

    except Exception as e:
        print(f"An error occurred during calculation of PSI: {e}")
        return np.nan


def calculate_eia(eeg_signals, eeg_channel, segment_length, seizure_startpoint, seizure_endpoint, debug=False):

    """
    Calculate the Early Ictal Amplitude (EIA)

    Parameters:
    eeg_signals: numpy.ndarray - A 2D numpy array containing the EEG signals.
    eeg_channel: int - The channel index to analyze.
    segment_length: int - Length of each segment in samples.
    seizure_startpoint: int - The startpoint of the seizure in samples.
    seizure_endpoint: float - End time of the seizure in seconds.
    debug: bool - If True, print debug information.

    Returns:
    dict: A dictionary containing the calculated EIA metric.
    """

    if debug:
        print("Calculating EIA...")
    
    try:
        # Check if the signal is long enough for 8 segments
        if (8 * segment_length) > len(eeg_signals[eeg_channel][seizure_startpoint:seizure_endpoint]):
            raise ValueError("Seizure signal is less than 8 segments long.")
        
        eia_startpoint = seizure_startpoint
        eia_endpoint = int(seizure_startpoint + 8 * segment_length)
        
        # Calculate absolute signal mean of the first 8 seizure segments
        eia = abs(eeg_signals[eeg_channel][eia_startpoint:eia_endpoint]).mean()
        
        if eia is not None:
            eia_result = {
                'name': 'eia', 
                'value': eia, 
                'timepoints': {
                    'startpoint': eia_startpoint,
                    'endpoint': eia_endpoint,
                },
                'unit': 'μV',
                'description': 'Early Ictal Amplitude'
            }
        
        if debug:
            print(f"Calculated EIA: {eia_result}")

        return eia_result

    except Exception as e:
        print(f"An error occurred during calculation of EIA: {e}")
        return np.nan


def calculate_mia(eeg_signals, eeg_channel, segment_length, seizure_startpoint, seizure_endpoint, debug=False):

    """
    Calculate the Midictal Amplitude (MIA)

    Parameters:
    eeg_signals: numpy.ndarray - A 2D numpy array containing the EEG signals.
    eeg_channel: int - The channel index to analyze.
    segment_length: int - Length of each segment in samples.
    seizure_startpoint: int - The startpoint of the seizure in samples.
    seizure_endpoint: float - End time of the seizure in seconds.
    debug: bool - If True, print debug information.

    Returns:
    dict: A dictionary containing the calculated MIA metric.
    """
    
    if debug:
        print("Calculating MIA...")
    
    try:
        midictal_amplitudes_list = []

        # Check if the signal is long enough for 8 segments
        if (8 * segment_length) > len(eeg_signals[eeg_channel][seizure_startpoint:seizure_endpoint]):
            raise ValueError("Signal is less than 8 segments long.")

        # Loop through sequences of 8 segments within the EEG seizure segment
        for i in range((len(eeg_signals[eeg_channel][seizure_startpoint:seizure_endpoint]) // segment_length) - 7):
            midictal_amplitudes_list.append(
                abs(eeg_signals[eeg_channel][seizure_startpoint + i * segment_length : 
                               seizure_startpoint + (i + 8) * segment_length]).mean()
            )

        if midictal_amplitudes_list:
            mia = np.max(midictal_amplitudes_list)
            mia_index = np.argmax(midictal_amplitudes_list)
            mia_startpoint = seizure_startpoint + mia_index * segment_length
            mia_endpoint = seizure_startpoint + (mia_index + 8) * segment_length 
            
            if mia is not None:
                mia_result = {
                    'name': 'mia', 
                    'value': mia, 
                    'timepoints': {
                        'startpoint': mia_startpoint,
                        'endpoint': mia_endpoint,
                    },
                'unit': 'μV',
                'description': 'Midictal Amplitude'
                }
            
            if debug:
                print(f"Calculated MIA: {mia}, Startpoint: {mia_startpoint}, Endpoint: {mia_endpoint}")

            return mia_result
        else:
            raise ValueError("No absolute mean could be calculated.")

    except Exception as e:
        print(f"An error occurred during calculation of MIA: {e}")
        return np.nan, np.nan, np.nan


def calculate_msp(eeg_signals, eeg_channel, sampling_frequency, segment_length, seizure_startpoint, seizure_endpoint, frequency_bands, debug=False):

    """
    Calculate the Maximum Sustained Power (MSP)

    Parameters:
    eeg_signals: numpy.ndarray - A 2D numpy array containing the EEG signals.
    eeg_channel: int - The channel index to analyze.
    sampling_frequency: int, optional - The sampling frequency of the signal.
    segment_length: int - Length of each segment in samples.
    seizure_startpoint: int - The startpoint of the seizure in samples.
    seizure_endpoint: float - End time of the seizure in seconds.
    frequency_bands: dict - Dictionary containing frequency bands for FFT power calculation.
    debug: bool - If True, print debug information.

    Returns:
    dict: A dictionary containing the calculated MSP metric.
    """

    if debug:
        print("Calculating MSP...")
    
    maximum_sustained_power_list = []

    try:
        # Check if the signal is long enough for 8 segments
        if (8 * segment_length) > len(eeg_signals[eeg_channel][seizure_startpoint:seizure_endpoint]):
            raise ValueError("Signal is less than 8 segments long.")

        # Loop through consecutive segments
        for i in range((len(eeg_signals[eeg_channel][seizure_startpoint:seizure_endpoint]) // segment_length) - 7):
            segment = eeg_signals[eeg_channel][seizure_startpoint + i * segment_length : seizure_startpoint + (i + 8) * segment_length]

            # Calculate FFT and PSD
            fft_result = fft(segment)
            psd = np.abs(fft_result) ** 2  # Calculate power from FFT results

            # Create frequency array
            freq = np.fft.fftfreq(len(segment), d=1/sampling_frequency)

            # Get frequency indices for defined bands
            frequency_indices = [np.logical_and(freq >= band[0], freq <= band[1]) for band in frequency_bands.values()]
            mean_power_per_band = [np.sum(psd[frequency_index]) for frequency_index in frequency_indices]

            # Total mean power for the 'total' band
            mean_power_total = mean_power_per_band[list(frequency_bands).index('total')]
            maximum_sustained_power_list.append(mean_power_total)

        if maximum_sustained_power_list:
            msp = np.max(maximum_sustained_power_list)
            msp_index = np.argmax(maximum_sustained_power_list)
            msp_startpoint = seizure_startpoint + msp_index * segment_length
            msp_endpoint = seizure_startpoint + (msp_index + 8) * segment_length
            
            if msp is not None:
                msp_result = {
                    'name': 'msp', 
                    'value': msp, 
                    'timepoints': {
                        'startpoint': msp_startpoint,
                        'endpoint': msp_endpoint,
                    },
                'unit': 'μV²/Hz',
                'description': 'Maximum Sustained Power'
                }
                
            if debug:
                print(f"Calculated MSP: {msp}, Startpoint: {msp_startpoint}, Endpoint: {msp_endpoint}")

            return msp_result
        else:
            raise ValueError("No sustained power could be calculated.")

    except Exception as e:
        print(f"An error occurred during calculation of MSP: {e}")
        return np.nan, np.nan, np.nan


def calculate_ttpp(eeg_signals, eeg_channel, sampling_frequency, segment_length, seizure_startpoint, seizure_endpoint, frequency_bands, msp_startpoint=None,  msp_endpoint=None,debug=False):
    
    """
    Calculate the Maximum Sustained Power (MSP)

    Parameters:
    eeg_signals: numpy.ndarray - A 2D numpy array containing the EEG signals.
    eeg_channel: int - The channel index to analyze.
    sampling_frequency: int, optional - The sampling frequency of the signal.
    segment_length: int - Length of each segment in samples.
    seizure_startpoint: int - The startpoint of the seizure in samples.
    seizure_endpoint: float - End time of the seizure in seconds.
    frequency_bands: dict - Dictionary containing frequency bands for FFT power calculation.
    msp_startpoint: int - Startpoint of maximum sustained supression sample
    msp_endpoint: int - Endpoint of maximum sustained supression sample
    debug: bool - If True, print debug information.

    Returns:
    dict: A dictionary containing the calculated TTPP metric.
    """

    if debug:
        print("Calculating TPPP...")
    
    try:
    
        # Validate msp_startpoint and msp_endpoint
        if (msp_startpoint is None or msp_endpoint is None or np.isnan(msp_startpoint) or np.isnan(msp_endpoint)):
            msp_result = calculate_msp(eeg_signals, eeg_channel, sampling_frequency, segment_length, seizure_startpoint, seizure_endpoint, frequency_bands)
            
            msp_startpoint = msp_result['timepoints']['startpoint']
            msp_endpoint = msp_result['timepoints']['endpoint']
            
        # Ensure msp_startpoint and msp_endpoint are valid
        if msp_startpoint < 0 or msp_endpoint < 0 or msp_startpoint >= len(eeg_signals[eeg_channel]) or msp_endpoint >= len(eeg_signals[eeg_channel]):
            raise ValueError("Invalid MSP startpoint or endpoint values.")
    
        # Calculate TTPP
        ttpp = ((msp_endpoint - msp_startpoint) / 2 + msp_startpoint) / sampling_frequency
            
        if ttpp is not None:
            ttpp_result = {
                'name': 'ttpp', 
                'value': ttpp, 
                'timepoints': {
                    'timepoint': int(ttpp * sampling_frequency)
                },
                'unit': 's',
                'description': 'Time to Peak Power'
            }
        
        if debug:
            print(f"Calculated TPPP: {ttpp}s")

        return ttpp_result

    except Exception as e:
        print(f"An error occurred during calculation of TPPP: {e}")
        return np.nan


def calculate_coh(eeg_signals, eeg_channel, sampling_frequency, segment_length, seizure_startpoint, n_consecutive_segments, debug=False):
 
    """
    Calculate the Maximum Sustained Coherence (COH)

    Parameters:
    eeg_signals: numpy.ndarray - A 2D numpy array containing the EEG signals.
    eeg_channel: int - The channel index to analyze.
    sampling_frequency: float - Sampling frequency of the EEG signal in Hz.
    segment_length: int - Length of the segment in samples.
    seizure_startpoint: int - The start index of the seizure segment.
    n_consecutive_segments: int - Number of consecutive segments to consider.
    debug: bool - If True, print debug information.

    Returns:
    dict: A dictionary containing the calculated COH metric.
    """
    
    if debug:
        print("Calculating COH...")
    
    maximum_sustained_coherence_list = []
    
    
    #Check if both signals equal length
    #
    #
    #
    
    
    # Check signal length
    if (n_consecutive_segments * segment_length) > len(eeg_signals[eeg_channel[0]][seizure_startpoint:]):
        raise ValueError("Signal is less than {} segments long.".format(n_consecutive_segments))

    # Loop through consecutive segments
    i = 0
    while (i + n_consecutive_segments) * segment_length <= len(eeg_signals[eeg_channel[0]][seizure_startpoint:]):
        # Get COH signal segments for both channels
        eeg_signal_channel_1 = eeg_signals[eeg_channel[0]][seizure_startpoint + i * segment_length: seizure_startpoint + (i + n_consecutive_segments) * segment_length]
        eeg_signal_channel_2 = eeg_signals[eeg_channel[1]][seizure_startpoint + i * segment_length: seizure_startpoint + (i + n_consecutive_segments) * segment_length]

        # Calculate coherence
        freqs, Cxy = signal.coherence(
            eeg_signal_channel_1,
            eeg_signal_channel_2,
            fs=sampling_frequency,
            nperseg=segment_length,
            window='hann',
            noverlap=segment_length // 2,
            detrend='constant'
        )

        # Retain delta band (0.78 - 3.5 Hz) only
        df_coh_x = (Cxy[(freqs >= 0.78) & (freqs <= 3.5)].mean()) * 100
        maximum_sustained_coherence_list.append(df_coh_x)

        i += 1

    # Calculate COH
    if maximum_sustained_coherence_list:
        coh = np.array(maximum_sustained_coherence_list).max()
        coh_index = maximum_sustained_coherence_list.index(coh)
        
        coh_startpoint = seizure_startpoint + (coh_index + 1) * segment_length
        coh_endpoint = seizure_startpoint + (coh_index + 1 + 4) * segment_length
        
        if coh is not None:
            coh_results = {
                'name': 'coh', 
                'value': coh, 
                'timepoints': {
                        'startpoint': coh_startpoint,
                        'endpoint': coh_endpoint,
                },
                'unit': '%',
                'description': 'Maximum Sustained Coherence'
            }

        return coh_results
    else:
        raise ValueError("COH cannot be calculated.")
        

def calculate_ttpc(eeg_signals, eeg_channel, sampling_frequency, segment_length, seizure_startpoint, n_consecutive_segments, coh_startpoint=None, coh_endpoint=None, debug=False):

    """
    Calculate the Time to Peak Coherence (TTPC)

    Parameters:
    eeg_signals: numpy.ndarray - A 2D numpy array containing the EEG signals.
    eeg_channel: int - The channel index to analyze.
    sampling_frequency: float - Sampling frequency of the EEG signal in Hz.
    segment_length: int - Length of the segment in samples.
    seizure_startpoint: int - The start index of the seizure segment.
    seizure_endpoint: int - The end index of the seizure segment.
    n_consecutive_segments: int - Number of consecutive segments to consider.
    coh_startpoint: int - The startpoint of coherence measurement.
    coh_endpoint: int - The endpoint of coherence measurement.
    debug: bool - If True, print debug information (default is False).

    Returns:
    dict: A dictionary containing the calculated TTPC metric.
    """
    try:
    
        # Validate coh_startpoint and coh_endpoint
        if (coh_startpoint is None or coh_endpoint is None or np.isnan(coh_startpoint) or np.isnan(coh_endpoint)):
            coh_result = calculate_coh(eeg_signals, eeg_channel, sampling_frequency, segment_length, seizure_startpoint, n_consecutive_segments, debug)
 
            coh_startpoint = coh_result['timepoints']['startpoint']
            coh_endpoint = coh_result['timepoints']['endpoint']
        
        # Ensure coh_startpoint and coh_endpoint are valid
        for eeg_channel_index in eeg_channel:
            if coh_startpoint < 0 or coh_endpoint < 0 or coh_startpoint >= len(eeg_signals[eeg_channel[eeg_channel_index]]) or coh_endpoint >= len(eeg_signals[eeg_channel[eeg_channel_index]]):
                raise ValueError("Invalid COH startpoint or endpoint values.")
        
        # Calculate TTPP
        ttpc = ((coh_endpoint - coh_startpoint) / 2 + coh_startpoint) / sampling_frequency
            
        if ttpc is not None:
            ttpc_result = {
                'name': 'ttpc', 
                'value': ttpc, 
                'timepoints': {
                    'timepoint': int(ttpc * sampling_frequency)
                },
                'unit': 's',
                'description': 'Time to Peak Coherence'
            }
        
        if debug:
            print(f"Calculated TPPP: {ttpc}s")

        return ttpc_result 
        
    except Exception as e:
        print(f"An error occurred during calculation of TTPC: {e}")
        return np.nan
        
def metrics(eeg, segment_length=DEFAULTS['segment_length'], metrics_list=None, seizure_duration=None, frequency_bands=None, debug=False):
    """
    Main function to calculate various seizure metrics based on the provided indices.

    Parameters:
    eeg: EEG
    segment_length: int - Length of segments for calculation.
    metrics_list: list - List specifying which metrics to calculate, can contain metric names or dictionaries.
    sampling_frequency: int, optional - The sampling frequency of the signal.
    seizure_duration: int, optional - The duration of the seizure.
    frequency_bands: dict, optional - Frequency bands for calculations.

    Returns:
    results: dict - Dictionary of calculated metrics.
    """
    
    if debug:
        print("Start calculating seizure quality metrics")
    
    # Set default frequency bands if not provided
    if frequency_bands is None:
        frequency_bands = {
            'delta': [0.5, 4],
            'theta': [4, 8],
            'alpha': [8, 12],
            'beta': [12, 30],
            'gamma': [30, 100],
            'total': [0.5, 100]  # Example of a total band
        }
        
    eeg_signals = eeg['signals']
    sampling_frequency = eeg['sampling_frequency']
    #segment_length = eeg['']
    seizure_startpoint = eeg['timepoints']['seizure_startpoint']
    seizure_endpoint = eeg['timepoints']['seizure_endpoint']
    
    #Validate input data
    if not isinstance(eeg_signals, np.ndarray):
        raise ValueError("EEG signals must be a numpy array.")
    if eeg_signals.size == 0:
        raise ValueError("EEG signals cannot be empty.")
    if sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be a positive number.")
    if segment_length <= 0:
        raise ValueError("Segment length must be a positive number.")
    #if not all(isinstance(ch, int) for ch in eeg_channel):
    #    raise ValueError("EEG channels must be integers.")
    if seizure_startpoint < 0 or seizure_startpoint >= eeg_signals.shape[1]:
        raise ValueError("Invalid seizure start point.")
    if seizure_endpoint < 0 or seizure_endpoint >= eeg_signals.shape[1] or seizure_endpoint < seizure_startpoint:
        raise ValueError("Invalid seizure end point.")

    # Default metrics as a list of dictionaries
    default_metrics_list = [
        {
            'name': 'asei',
            'calculate': True,
            'channel': 0
        },
        {
            'name': 'sei',
            'calculate': True,
            'channel': 0
        },
        {
            'name': 'psi',
            'calculate': True,
            'channel': 0
        },
        {
            'name': 'eia',
            'calculate': True,
            'channel': 0
        },
        {
            'name': 'mia',
            'calculate': True,
            'channel': 0
        },
        {
            'name': 'msp',
            'calculate': True,
            'channel': 0
        },
        {
            'name': 'ttpp',
            'calculate': True,
            'channel': 0
        },
        {
            'name': 'coh',
            'calculate': True,
            'channels': [0, 1]
        },
        {
            'name': 'ttpc',
            'calculate': True,
            'channels': [0, 1]
        }
    ]
        
    # If metrics_list is None, use the default metrics list to calculate all
    if metrics_list is None:
        metrics_list = [metric for metric in default_metrics_list if metric['calculate']]
    else:
        # Create a list for the specified metrics, initializing with default params
        indices = []
        if isinstance(metrics_list, list):
            # Check if it's a mix of names and dicts
            for item in metrics_list:
                if isinstance(item, str):
                    # If it's a name, get the default params
                    default_metric = next((metric for metric in default_metrics_list if metric['name'] == item), None)
                    if default_metric is not None:
                        indices.append(default_metric)
                elif isinstance(item, dict):
                    # If it's a dict, add it with the specified name
                    metric_name = item.get('name')
                    default_metric = next((metric for metric in default_metrics_list if metric['name'] == metric_name), None)
                    if default_metric is not None:
                        # Update the default parameters with the provided ones
                        updated_metric = {**default_metric, **item}
                        indices.append(updated_metric)

        metrics_list = indices

    results = []

    for metric in metrics_list:
        metric_name = metric['name']
        if debug:
            print(f"Calculating {metric_name}...")
        

        if metric_name == 'asei':
            results.append(calculate_asei(eeg_signals, metric['channel'], segment_length, seizure_startpoint, seizure_endpoint, debug))

        elif metric_name == 'sei':
            
            # Initialize asei variable
            asei = None
            
            # Check if ASEI is in results
            for results_metric in results:
                if results_metric['name'] == 'asei':
                    asei = results_metric['value']
                    
            results.append(calculate_sei(eeg_signals, metric['channel'], sampling_frequency, segment_length, seizure_startpoint, seizure_endpoint, asei, debug))

        elif metric_name == 'psi':
            results.append(calculate_psi(eeg_signals, metric['channel'], sampling_frequency, segment_length, seizure_endpoint, debug))
        
        elif metric_name == 'eia':
            results.append(calculate_eia(eeg_signals, metric['channel'], segment_length, seizure_startpoint, seizure_endpoint, debug))

        elif metric_name == 'mia':
            results.append(calculate_mia(eeg_signals, metric['channel'], segment_length, seizure_startpoint, seizure_endpoint, debug))

        elif metric_name == 'msp':
            results.append(calculate_msp(eeg_signals, metric['channel'], sampling_frequency, segment_length, seizure_startpoint, seizure_endpoint, frequency_bands, debug))

        elif metric_name == 'ttpp':
        
            # Initialize variables to hold start and endpoint for MSP
            msp_startpoint = None
            msp_endpoint = None
            
            # Check if MSP is in results
            for results_metric in results:
                if results_metric['name'] == 'msp':
                    msp_startpoint = results_metric['timepoints'].get('startpoint', None)
                    msp_endpoint = results_metric['timepoints'].get('endpoint', None)
                    
                    # Validate msp_startpoint and msp_endpoint
                    if msp_startpoint is None or msp_endpoint is None:
                        msp_startpoint = None
                        msp_endpoint = None
                        
            results.append(calculate_ttpp(eeg_signals, metric['channel'], sampling_frequency, segment_length, seizure_startpoint, seizure_endpoint, frequency_bands, msp_startpoint,  msp_endpoint, debug))

        elif metric_name == 'coh':
            #!
            #
            metric['n_consecutive_segments'] = 8
            metric['channel'] = [0, 1]
            
            results.append(calculate_coh(eeg_signals, metric['channel'], sampling_frequency, segment_length, seizure_startpoint, metric['n_consecutive_segments'], debug))
 
 
        elif metric_name == 'ttpc':
        
            # Initialize variables to hold start and endpoint for COH
            coh_startpoint = None
            coh_endpoint = None
            
            # Check if COH is in results
            for results_metric in results:
                if results_metric['name'] == 'coh':
                    coh_startpoint = results_metric['timepoints'].get('startpoint', None)
                    coh_endpoint = results_metric['timepoints'].get('endpoint', None)
                    
                    # Validate coh_startpoint and coh_endpoint
                    if coh_startpoint is None or coh_endpoint is None:
                        coh_startpoint = None
                        coh_endpoint = None
                        
            #!
            #
            metric['n_consecutive_segments'] = 8
            metric['channel'] = [0, 1]
            results.append(calculate_ttpc(eeg_signals, metric['channels'], sampling_frequency, segment_length, seizure_startpoint, metric['n_consecutive_segments'], debug))

    return results
