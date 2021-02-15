"""Configuration values for MNE-HFO."""
BIDS_VERSION = "1.4.0"

DOI = """10.5281/zenodo.4485036"""

EPHY_ALLOWED_DATATYPES = ['ieeg']

REFERENCES = {
    'mne-bids':
        'Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., '
        'Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., '
        'Tavabi, K., Höchenberger, R., Welke, D., Brunner, C., '
        'Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). '
        'MNE-BIDS: Organizing electrophysiological data into the '
        'BIDS format and facilitating their analysis. Journal of '
        'Open Source Software 4: (1896). '
        'https://doi.org/10.21105/joss.01896',
    'ieeg':
        'Holdgraf, C., Appelhoff, S., Bickel, S., Bouchard, K., '
        'D\'Ambrosio, S., David, O., … Hermes, D. (2019). iEEG-BIDS, '
        'extending the Brain Imaging Data Structure specification '
        'to human intracranial electrophysiology. Scientific Data, '
        '6, 102. https://doi.org/10.1038/s41597-019-0105-7'
}
MINIMUM_SUGGESTED_SFREQ = 2000
ACCEPTED_BAND_METHODS = ['linear', 'log']

# conversion of time scales to seconds
TIME_SCALE_TO_SECS = {'s': 1., 'm': 60., 'h': 60. * 60, 'd': 60. * 60 * 24}
EVENT_COLUMNS = ['onset', 'duration', 'sample', 'trial_type']
ANNOT_COLUMNS = ['onset', 'duration', 'label', 'channels']
