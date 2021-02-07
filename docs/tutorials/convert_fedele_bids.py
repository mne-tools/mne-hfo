from  mne_bids import read_raw_bids, write_raw_bids, BIDSPath, get_entities_from_fname
from pathlib import Path


import scipy.io


class MatReader:
    """
    Object to read mat files into a nested dictionary if need be.
    Helps keep strucutre from matlab similar to what is used in python.
    """

    def __init__(self, filename=None):
        self.filename = filename

    def loadmat(self, filename):
        """
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        """
        try:
            data = scipy.io.loadmat(
                filename, struct_as_record=False, squeeze_me=True, chars_as_strings=True
            )
        except NotImplementedError as e:
            print(e)
            data = hdf5storage.loadmat(filename)
        return self._check_keys(data)

    def _check_keys(self, dict):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        print(f"Found these keys {dict.keys()}")
        for key in dict:
            if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
                dict[key] = self._todict(dict[key])
        return dict

    def _todict(self, matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                dict[strg] = self._todict(elem)
            elif isinstance(elem, np.ndarray):
                dict[strg] = self._tolist(elem)
            else:
                dict[strg] = elem
        return dict

    def _tolist(self, ndarray):
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(self._todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(self._tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

def _load_mat_file(mat_fname):
    mat_dict = MatReader().loadmat(mat_fname)['data']
    return mat_dict


def _write_bids(mat_dict, bids_path):
    eeg_data = mat_dict['x']
    ch_names = mat_dict['label']
    sfreq = mat_dict['fs']

    meas_date = mat_dict['Rec_timestamp']

    print(sfreq, meas_date)

    raise Exception('hi')

def convert_fedele_bids(source_path, root):
    # get all patients
    patients = sorted([x for x in Path(source_path).glob('pat*') if x.is_dir()])

    for subject_path in patients:
        data_path = subject_path / 'mat'
        mat_fpaths = sorted(data_path.glob('*.mat'))
        subject = subject_path.name

        for fpath in mat_fpaths:
            subj, _, _, nightnum, _, run = fpath.name.split('_')

            # load the matlab data
            mat_dict = _load_mat_file(fpath)

            # create BIDS path
            bids_path = BIDSPath(subject=subject, session='presurgery',
                                 task='interictal', run=run, datatype='ieeg',
                                 suffix='ieeg', extension='.vhdr')

            _write_bids(mat_dict, bids_path)

if __name__ == '__main__':
    root = Path('/Users/adam2392/OneDrive - Johns Hopkins/epilepsy_bids/')
    source_path = root / 'sourcedata' / 'hfo_ieegdata'

    convert_fedele_bids(source_path, root)