.. ECTMetrics documentation master file, created by
   sphinx-quickstart on Thu Sep 26 20:53:38 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ECTMetrics
==========================

ECTMetrics is a Python library for analyzing EEG signals, particularly focusing on
electroconvulsive therapy (ECT) seizure metrics. It provides functionalities to
generate synthetic EEG signals, visualize them, and calculate various metrics related
to seizure activity.

Features
--------

- Generate synthetic ECT-specific EEG signals with customizable parameters.
- Visualize EEG signals for better understanding and analysis.
- Import EEG data from GPD (Elektrika Inc®). *currently under development*
- Calculate various `ECT seizure metrics <modules.html#ectmetrics.metric.metric>`_ including:

  - `Seizure Energy Index (SEI) <modules.html#ectmetrics.metric.metric_sei>`_
  - `Average Seizure Energy Index (ASEI) <modules.html#ectmetrics.metric.metric_asei>`_
  - `Postictal Suppression Index (PSI) <modules.html#ectmetrics.metric.metric_psi>`_
  - `Earlyictal Amplitude (EIA) <modules.html#ectmetrics.metric.metric_eia>`_
  - `Midictal Amplitude (MIA) <modules.html#ectmetrics.metric.metric_mia>`_
  - `Maximum Sustained Power (MSP) <modules.html#ectmetrics.metric.metric_msp>`_
  - `Time to Peak Power (TTPP) <modules.html#ectmetrics.metric.metric_ttpp>`_
  - `Maximum Sustained Coherence (COH) <modules.html#ectmetrics.metric.metric_coh>`_
  - `Time to Peak Coherence (TTPC) <modules.html#ectmetrics.metric.metric_ttpc>`_

.. note::
   ECTMetrics is under active development, and contributions are welcome. 
   If you encounter issues or have suggestions for improvements, please consult 
   the `contact` section or open an issue on our repository.
   

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   contact
   citations
   license


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
