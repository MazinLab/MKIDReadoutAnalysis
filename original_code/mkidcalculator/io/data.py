import os
import yaml
import glob
import fnmatch
import logging
import pathlib
import numpy as np
from scipy.io import loadmat
from functools import partial

from mkidcalculator.io.utils import (_loaded_npz_files, offload_data, ev_nm_convert, load_legacy_binary_data,
                                     structured_to_complex)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class NoData:
    def __getitem__(self, item):
        raise AttributeError("This object has no pre-processed data.")


def analogreadout_temperature(metadata):
    """
    Returns the average temperature across all the temperatures taken during
    the data set.
    Args:
        metadata: dictionary
            The metadata dictionary from the analogreadout procedure.
    Returns:
        temperature: float
            The average temperature during the data set.
    """
    temperatures = [metadata[key]['thermometer']['temperature'][0]
                    for key in metadata.keys() if key not in ('parameters', 'file_name')]
    temperature = np.mean(temperatures)
    return temperature


def analogreadout_sample_rate(metadata):
    """
    Returns the sample rate in Hz from the metadata.
    Args:
        metadata: dictionary
            The metadata dictionary from the analogreadout procedure.
    Returns:
        sample_rate: float
            The sample rate in Hz.
    """
    sample_rate = metadata['parameters']['sample_rate'] * 1e6
    return sample_rate


def analogreadout_trace(array, npz, quad, channel):
    if isinstance(array, str):  # array is a string
        directory = os.path.dirname(npz)
        # casts all path types into Windows format and grabs the base name
        file_name = pathlib.PureWindowsPath(array).name
        data = np.load(os.path.join(directory, file_name))[quad][channel]
    else:
        try:  # array is a real array
            data = array[quad]
        except IndexError:  # old format with complex values
            if quad == "I":
                data = array.real
            elif quad == "Q":
                data = array.imag
            else:
                raise ValueError("invalid quad")
    return data


class AnalogReadoutABC:
    """
    Abstract base class for handling data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    def __init__(self, npz_handle=None, channel=None, index=None):
        self.channel = channel
        self.index = index
        if isinstance(npz_handle, str):
            self._npz = os.path.abspath(npz_handle)
        elif isinstance(npz_handle, np.lib.npyio.NpzFile):
            self._npz = os.path.abspath(npz_handle.fid.name)
        else:
            self._npz = None

    def __getstate__(self):
        return offload_data(self)

    def __getitem__(self, item):
        # get conversion values
        try:
            convert = self.CONVERT[item]
        except KeyError:
            raise KeyError("allowed keys for this data structure are in {}".format(list(self.CONVERT.keys())))
        try:
            # get the result from the npz file
            result = _loaded_npz_files[self._npz][convert[0] if isinstance(convert, tuple) else convert]
            if result.dtype.kind in ["S", "U", "O"]:
                # if it's an object unpack it
                result = result.item()
            else:
                # else get the channel and index
                if self.index is not None:
                    result = result[:, self.index, ...]
                if self.channel is not None:
                    result = result[self.channel]
            # more conversion
            if isinstance(convert, tuple) and len(convert) > 1:
                # try the first conversion
                if not callable(convert[1]):
                    try:
                        if isinstance(convert[1], (tuple, list)):
                            for c in convert[1]:
                                result = result[c]
                        else:
                            result = result[convert[1]]
                    except IndexError:  # didn't work try the second
                        # if that failed run a function (for complex formatted data)
                        result = convert[2](result)
                else:
                    result = convert[1](result)
            return result
        except TypeError:
            raise KeyError("no data has been loaded.")

    def free_memory(self):
        """Frees memory from the wrapped npz file."""
        _loaded_npz_files.free_memory(self._npz)


class AnalogReadoutLoop(AnalogReadoutABC):
    """
    Class for handling loop data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    CONVERT = {"f": "freqs", "z": "z", "imbalance": ("calibration", structured_to_complex), "offset": "z_offset",
               "metadata": "metadata", "attenuation": ("metadata", ("parameters", "attenuation")),
               "field": ("metadata", ("parameters", "field")), "temperature": ("metadata", analogreadout_temperature)}


class AnalogReadoutNoise(AnalogReadoutABC):
    """
    Class for handling noise data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    # "i_psd": ("psd", "I"), "q_psd": ("psd", "Q"), "f_psd": "f_psd" not using these from file but they are there
    CONVERT = {"f_bias": "freqs", "i_trace": ("noise", "I", np.real), "q_trace": ("noise", "Q", np.imag),
               "metadata": "metadata", "attenuation": ("metadata", ("parameters", "attenuation")),
               "sample_rate": ("metadata", analogreadout_sample_rate)}

    def __init__(self, npz_handle=None, channel=None, index=0):
        super().__init__(npz_handle=npz_handle, channel=channel, index=index)


class AnalogReadoutPulse(AnalogReadoutABC):
    """
    Class for handling pulse data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    CONVERT = {"f_bias": "freqs", "offset": "zero", "metadata": "metadata",
               "attenuation": ("metadata", ("parameters", "attenuation")),
               "sample_rate": ("metadata", analogreadout_sample_rate)}

    def __init__(self, *args, energies=(), wavelengths=(), **kwargs):
        super().__init__(*args, **kwargs)
        if energies != ():
            self._energies = tuple(np.atleast_1d(energies))
        elif wavelengths != ():
            self._energies = tuple(ev_nm_convert(np.atleast_1d(wavelengths)))
        else:
            self._energies = ()
        self._add_pulses()

    def __setstate__(self, state):
        self.__dict__ = state
        self._add_pulses()

    def __getitem__(self, item):
        if item == 'energies':
            if self._energies:
                result = self._energies
            else:
                metadata = super().__getitem__("metadata")
                if 'laser' in metadata['parameters'].keys():
                    key = 'laser'  # Legacy
                else:
                    key = 'source'  # New

                try:
                    source_state = np.array(metadata['parameters'][key])
                    if source_state.size == 1:  # no laser box
                        pass
                    elif source_state.size == 5:  # when laser box is used
                        source_state *= np.array([813.7, 916.8, 978.6, 1110,
                                                 1310])
                    elif source_state.size == 8:  # no longer used
                        source_state *= np.array([254, 405.9, 663.1, 813.7,
                                                 916.8, 978.6, 1120, 1310])
                    else:
                        raise ValueError(
                            f"Unrecognized source state: {source_state}")
                    source_state = source_state[source_state != 0]
                    self._energies = tuple(ev_nm_convert(source_state))
                except KeyError:
                    pass
                result = self._energies
        else:
            result = super().__getitem__(item)
        return result

    def _add_pulses(self):
        self.CONVERT.update(
            {"i_trace": ("pulses", partial(analogreadout_trace,
                                           npz=self._npz, quad="I",
                                           channel=self.channel)),
             "q_trace": ("pulses", partial(analogreadout_trace,
                                           npz=self._npz, quad="Q",
                                           channel=self.channel))})


def analogreadout_resonator(file_name, channel=None):
    """
    Class for loading in analogreadout resonator data.
    Args:
        file_name: string
            The resonator configuration file name.
        channel: integer
            The resonator channel for the data.
    Returns:
        loop_kwargs: list of dictionaries
            A list of keyword arguments to send to Loop.from_file().
    """
    directory = os.path.dirname(file_name)
    if os.path.splitext(file_name)[-1] == ".npz":  # old format
        config = np.load(file_name, allow_pickle=True)
        parameter_dict = config['parameter_dict'].item()
    else:  # new format
        with open(file_name, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        try:  # old format (barely used)
            parameter_dict = config['parameter_dict']
        except KeyError:
            parameter_dict = config['parameters']
    loop_kwargs = []
    pattern = "sweep_*_*_*_{:d}_*".format(int(channel // 2))
    for loop_name, parameters in parameter_dict.items():
        loop_file_name = os.path.join(directory, loop_name)
        match = fnmatch.filter([loop_name], pattern)
        if not os.path.isfile(loop_file_name) and match:
            log.warning("Could not find '{}'".format(loop_file_name))
        elif match:
            loop_kwargs.append({"loop_file_name": loop_file_name, "channel": int(channel % 2)})
            if parameters['noise'][0]:
                # 3 is boolean, 5 is number of off resonance
                n_noise = 1 + int(parameters['noise'][5]) * int(parameters['noise'][3])
                noise_name = "_".join(["noise", *loop_name.split("_")[1:]])
                noise_file_name = os.path.join(directory, noise_name)
                if os.path.isfile(noise_file_name):
                    noise_names = [noise_file_name] * n_noise
                    noise_kwargs = [{'index': ii} for ii in range(n_noise)]
                    loop_kwargs[-1].update({"noise_file_names": noise_names, "noise_kwargs": noise_kwargs,
                                            "channel": int(channel % 2)})
                else:
                    log.warning("Could not find '{}'".format(noise_file_name))
    return loop_kwargs


def analogreadout_sweep(file_name, unique=True):
    """
    Class for loading in analogreadout sweep data.
    Args:
        file_name: string
            The sweep configuration file name.
        unique: boolean
            If True, data taken under the same conditions as previous data is
            not loaded. If False, all data is loaded. The default is True.
    Returns:
        resonator_kwargs: list of dictionaries
            A list of keyword arguments to send to Resonator.from_file().
    """
    # Get the first config in the directory if a file isn't specified.
    if not os.path.isfile(file_name) and os.path.isdir(file_name):
        try:
            file_name = glob.glob(file_name + "/config_*")[0]
        except IndexError:
            raise OSError(f"There are no config files in {file_name}")

    # Open the config file
    if os.path.splitext(file_name)[-1] == ".npz":  # old format
        config = np.load(file_name, allow_pickle=True)
        parameter_dict = config['parameter_dict'].item()
        sweep_dict = config['sweep_dict'].item()

    else:  # new format
        with open(file_name, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        try:  # old format (barely used)
            parameter_dict = config['parameter_dict']
            sweep_dict = config['sweep_dict']
        except KeyError:
            parameter_dict = config['parameters']
            sweep_dict = config['sweep']
    file_names = list(parameter_dict.keys())
    if 'frequencies' in sweep_dict.keys():
        multiple = 1
        lengths = [len(sweep_dict['frequencies'])]
    else:
        multiple = 2
        lengths = [len(sweep_dict['frequencies1']),
                   len(sweep_dict['frequencies2'])]

    resonator_kwargs = []
    for index in range(multiple * len(file_names)):
        if unique:
            repeat = index // multiple + 1 > lengths[index % multiple]
        else:
            repeat = index // multiple + 1 > max(lengths)
        if repeat:
            continue
        resonator_kwargs.append({"resonator_file_name": file_name, "channel": index})

    return resonator_kwargs


class LegacyABC:
    """
    Abstract base class for handling data from the Legacy matlab code.
    Args:
        config_file: string
            A file path to the config file for the measurement.
        channel: integer
            An integer specifying which channel to load.
        index: tuple of integers (optional)
            An integer specifying which temperature and attenuation index to
            load. The default is None.
    """
    def __init__(self, config_file, channel, index=None):
        self.channel = channel
        self.index = index
        self._empty_fields = []
        self._do_not_clear = ['metadata']
        # load in data to the configuration file
        self._data = {'metadata': {}}
        config = loadmat(config_file, squeeze_me=True)
        for key in config.keys():
            if not key.startswith("_"):
                for name in config[key].dtype.names:
                    try:
                        self._data['metadata'][name] = float(config[key][name].item())
                    except ValueError:
                        self._data['metadata'][name] = config[key][name].item()
                    except TypeError:
                        try:
                            self._data['metadata'][name] = config[key][name].item().astype(float)
                        except ValueError:
                            self._data['metadata'][name] = config[key][name].item()  # spec_settings exception

    def __getitem__(self, item):
        value = self._data[item]
        if value is None and item not in self._empty_fields:
            self._load_data()
            value = self._data[item]
        return value

    def __getstate__(self):
        __dict__ = self.__dict__.copy()
        __dict__['_data'] = {}
        for key in self.__dict__['_data'].keys():
            if key not in self._do_not_clear:
                __dict__['_data'][key] = None
            else:
                __dict__['_data'][key] = self.__dict__['_data'][key]
        return __dict__

    def free_memory(self):
        """Frees memory from the wrapped data."""
        for key in self._data.keys():
            if key not in self._do_not_clear:
                self._data[key] = None

    def _load_data(self):
        raise NotImplementedError


class LegacyLoop(LegacyABC):
    """
    Class for handling loop data from the legacy Matlab code. Use legacy_loop()
    instead.
    """
    def __init__(self, config_file, channel, index):
        super().__init__(config_file, channel, index=index)
        # load in the loop data
        time = os.path.basename(config_file).split('_')[2:]
        directory = os.path.dirname(config_file)
        mat_file = "sweep_data.mat" if not time else "sweep_data_" + "_".join(time)
        self._mat = os.path.abspath(os.path.join(directory, mat_file))
        self._empty_fields += ["imbalance"]
        self._data.update({"f": None, "z": None, "imbalance": None, "offset": None, "field": None, "temperature": None,
                           "attenuation": None})  # defer loading

    def _load_data(self):
        data = loadmat(self._mat, struct_as_record=False)['IQ_data'][0, 0]
        res = data.temps[0, self.index[0]].attens[0, self.index[1]].res[0, self.channel]
        self._data.update({"f": res.freqs.squeeze(), "z": res.z.squeeze(), "imbalance": None,
                           "offset": res.zeropt.squeeze(), "field": 0,
                           "temperature": data.temprange[0, self.index[0]] * 1e-3,
                           "attenuation": data.attenrange[0, self.index[1]]})


class LegacyNoise(LegacyABC):
    """
    Class for handling noise data from the legacy Matlab code. Use
    legacy_noise() instead.
    """
    def __init__(self, config_file, channel, index=None, on_res=True, offset=0, chunk=None):
        super().__init__(config_file, channel, index=index)
        self.offset = offset
        self.chunk = chunk
        # figure out the file specifics
        directory = os.path.dirname(os.path.abspath(config_file))
        self._sweep_gui = os.path.basename(config_file).split("_")[0] == "sweep"
        if self._sweep_gui:
            if self.index is None:
                raise ValueError("The index (temperature, attenuation) must be specified for Sweep GUI data.")
            temps = np.arange(self._data['metadata']['starttemp'],
                              self._data['metadata']['stoptemp'] + self._data['metadata']['steptemp'] / 2,
                              self._data['metadata']['steptemp'])
            attens = np.arange(self._data['metadata']['startatten'],
                               self._data['metadata']['stopatten'] + self._data['metadata']['stepatten'] / 2,
                               self._data['metadata']['stepatten'])
            label = "a" if on_res else "b"
            label += "{:g}".format(index[2]) + "-" if len(index) > 2 and index[2] != 0 else "-"
            file_name = ("{:g}".format(temps[index[0]]) + "-" + "{:g}".format(channel // 2 + 1) + label +
                         "{:g}".format(attens[index[1]]) + ".ns")
            n_points = (self._data['metadata']['adtime'] * self._data['metadata']['noiserate'] /
                        self._data['metadata']['decfac'])
            self._data['attenuation'] = attens[index[1]]
            self._data['sample_rate'] = self._data['metadata']['noiserate']
        else:
            time = os.path.basename(config_file).split('.')[0].split('_')[2:]
            file_name = "pulse_data.ns" if not time else "pulse_data_" + "_".join(time) + ".ns"
            n_points = self._data['metadata']['noise_adtime'] * self._data['metadata']['samprate']
            self._data['attenuation'] = self._data['metadata']['atten1'] + self._data['metadata']['atten2']
            self._data['sample_rate'] = self._data['metadata']["samprate"]
        self._do_not_clear += ['attenuation', 'sample_rate']
        # load the data
        assert n_points.is_integer(), "The noise adtime and sample rate do not give an integer number of data points"
        self._n_points = int(n_points)
        self._bin = os.path.abspath(os.path.join(directory, file_name))
        self._data.update({"i_trace": None, "q_trace": None, "f_bias": None})  # defer loading

    def _load_data(self):
        # % 2 for resonator data
        i_trace, q_trace, f = load_legacy_binary_data(self._bin, self.channel % 2, self._n_points, offset=self.offset,
                                                      chunk=self.chunk)
        self._data.update({"i_trace": i_trace, "q_trace": q_trace, 'f_bias': f})


class LegacyPulse(LegacyABC):
    """
    Class for handling pulse data from the legacy Matlab code. Use
    legacy_pulse() instead.
    """
    def __init__(self, config_file, channel, energies=(), wavelengths=(), offset=0, chunk=None):
        channel = channel % 2  # channels can't be > 1
        super().__init__(config_file, channel=channel)
        self.offset = offset
        self.chunk = chunk
        # record the photon energies
        if energies != ():
            self._data["energies"] = tuple(np.atleast_1d(energies))
        elif wavelengths != ():
            self._data["energies"] = tuple(ev_nm_convert(np.atleast_1d(wavelengths)))
        else:
            self._data["energies"] = ()
        # get the important parameters from the metadata
        self._data["f_bias"] = self._data['metadata']["f0" + "{:g}".format(channel + 1)]
        self._data["offset"] = None
        self._data["attenuation"] = self._data['metadata']['atten1'] + self._data['metadata']['atten2']
        self._data['sample_rate'] = self._data['metadata']["samprate"]
        self._do_not_clear += ['f_bias', 'attenuation', 'offset', 'energies', 'sample_rate']
        self._empty_fields += ["offset"]

        directory = os.path.dirname(os.path.abspath(config_file))
        time = os.path.basename(config_file).split('.')[0].split('_')[2:]
        file_name = "pulse_data.dat" if not time else "pulse_data_" + "_".join(time) + ".dat"
        self._bin = os.path.abspath(os.path.join(directory, file_name))
        self._n_points = int(self._data['metadata']['numpts'])
        self._data.update({"i_trace": None, "q_trace": None})  # defer loading

    def _load_data(self):
        i_trace, q_trace, _ = load_legacy_binary_data(self._bin, self.channel, self._n_points, noise=False,
                                                      offset=self.offset, chunk=self.chunk)
        self._data.update({"i_trace": i_trace, "q_trace": q_trace})


def legacy_noise(config_file, channel, index=None, on_res=True, chunk=None):
    """
    Function for handling noise data from the legacy Matlab code.
    Args:
        config_file: string
            A file path to the config file for the measurement.
        channel: integer
            An integer specifying which channel to load.
        index: tuple of integers (optional)
            An integer specifying which temperature and attenuation index to
            load. An additional third index may be included in the tuple to
            specify additional noise points. This is only needed if the data
            is from a sweep config.
        on_res: boolean (optional)
            A boolean specifying if the noise is on or off resonance. This is
            only used when the noise comes from the Sweep GUI. The default is
            True.
        chunk: integer (optional)
            Chunk the data objects into a list of objects each with this many
            traces. The last object may have less traces. The default is None
            and only one object is returned.
    Returns:
        data: object or list of objects
            A data object used in Noise.from_file(). If chunk is used, a list
            of objects is returned.
    """
    if chunk is None:
        return LegacyNoise(config_file, channel, index=index, on_res=on_res)
    else:
        # get the number of triggers in the binary file
        ln = LegacyNoise(config_file, channel)
        bin_file = np.memmap(ln._bin, dtype=np.int16, mode='r')
        n_triggers = bin_file[4 * 12:].size / ln._n_points / 4.0
        data = []
        # loop over chunks
        offset = 0
        while offset < n_triggers:
            data.append(LegacyNoise(config_file, channel, index=index, on_res=on_res, offset=offset, chunk=chunk))
            offset += chunk
        return data


def legacy_pulse(config_file, channel, energies=(), wavelengths=(), chunk=None):
    """
    Function for handling pulse data from the legacy Matlab code.
    Args:
        config_file: string
            A file path to the config file for the measurement.
        channel: integer
            An integer specifying which channel to load.
        energies: number or iterable of numbers (optional)
            The known energies in the pulse data. The default is an empty
            tuple.
        wavelengths: number or iterable of numbers (optional)
            If energies is not specified, wavelengths can be specified instead
            which are internally converted to energies. The default is an empty
            tuple.
        chunk: integer (optional)
            Chunk the data objects into a list of objects each with this many
            traces. The last object may have less traces. The default is None
            and only one object is returned.
    Returns:
        data: object or list of objects
            A data object used in Pulse.from_file(). If chunk is used, a list
            of objects is returned.
    """
    if chunk is None:
        return LegacyPulse(config_file, channel, energies=energies, wavelengths=wavelengths)
    else:
        # get the number of triggers in the binary file
        lp = LegacyPulse(config_file, channel, energies=energies, wavelengths=wavelengths)
        bin_file = np.memmap(lp._bin, dtype=np.int16, mode='r')
        n_triggers = bin_file[4 * 14:].size / lp._n_points / 4.0
        data = []
        # loop over chunks
        offset = 0
        while offset < n_triggers:
            data.append(LegacyPulse(config_file, channel, energies=energies, wavelengths=wavelengths, offset=offset,
                                    chunk=chunk))
            offset += chunk
        return data


def legacy_loop(config_file, channel, index):
    """
    Function for handling loop data from the legacy Matlab code.
    Args:
        config_file: string
            A file path to the config file for the measurement.
        channel: integer
            An integer specifying which channel to load. The default is None
            which will raise an error forcing the user to directly specify the
            channel.
        index: tuple of integers
            An integer specifying which temperature and attenuation index to
            load. The default is None will raise an error forcing the user to
            directly specify the index.
    Returns:
        data: object
            A data object used in Loop.from_file().
    """
    return LegacyLoop(config_file, channel, index)


def legacy_resonator(config_file, channel=None, noise=True):
    """
    Function for loading in legacy matlab resonator data.
    Args:
        config_file: string
            The resonator configuration file name.
        channel: integer
            The resonator channel for the data.
        noise: boolean
            If False, ignore the noise data. The default is True.
    Returns:
        loop_kwargs: list of dictionaries
            A list of keyword arguments to send to Loop.from_file().
    """
    directory = os.path.dirname(config_file)
    config = loadmat(config_file, squeeze_me=True)['curr_config']
    temperatures = np.arange(config['starttemp'].astype(float),
                             config['stoptemp'].astype(float) + config['steptemp'].astype(float) / 2,
                             config['steptemp'].astype(float))
    attenuations = np.arange(config['startatten'].astype(float),
                             config['stopatten'].astype(float) + config['stepatten'].astype(float) / 2,
                             config['stepatten'].astype(float))

    loop_kwargs = []
    for t_index, temp in enumerate(temperatures):
        for a_index, atten in enumerate(attenuations):
            loop_kwargs.append({"loop_file_name": config_file, "index": (t_index, a_index), "data": legacy_loop,
                                "channel": channel})
            if config['donoise'] and noise:
                group = channel // 2 + 1
                # on resonance file names
                on_res = glob.glob(os.path.join(directory, "{:g}-{:d}a*-{:g}.ns".format(temp, group, atten)))
                noise_kwargs = []
                for file_name in on_res:
                    # collect the index for the file name
                    base_name = os.path.basename(file_name)
                    index2 = base_name.split("a")[1].split("-")[0]
                    index = (t_index, a_index, int(index2)) if index2 else (t_index, a_index)
                    noise_kwargs.append({"index": index, "on_res": True, "data": legacy_noise, "channel": channel})
                # off resonance file names
                off_res_names = glob.glob(os.path.join(directory, "{:g}-{:d}b*-{:g}.ns".format(temp, group, atten)))
                for file_name in off_res_names:
                    # collect the index for the file name
                    base_name = os.path.basename(file_name)
                    index2 = base_name.split("b")[1].split("-")[0]
                    index = (t_index, a_index, int(index2)) if index2 else (t_index, a_index)
                    noise_kwargs.append({"index": index, "on_res": False, "data": legacy_noise, "channel": channel})
                loop_kwargs[-1].update({"noise_file_names": [config_file] * len(on_res + off_res_names),
                                        "noise_kwargs": noise_kwargs})
                if not noise_kwargs:
                    log.warning("Could not find noise files for '{}'".format(config_file))
    return loop_kwargs


def legacy_sweep(config_file, noise=True):
    """
    Function for loading in analogreadout sweep data.
    Args:
        config_file: string
            The sweep configuration file name.
        noise: boolean
            If False, ignore the noise data. The default is True.
    Returns:
        resonator_kwargs: list of dictionaries
            A list of keyword arguments to send to Resonator.from_file().
    """
    config = loadmat(config_file, squeeze_me=True)['curr_config']
    channels = np.arange(len(config['f0list'].item()))

    resonator_kwargs = []
    for channel in channels:
        resonator_kwargs.append({'resonator_file_name': config_file, 'channel': channel, 'noise': noise,
                                 'data': legacy_resonator})
    return resonator_kwargs


def labview_segmented_widesweep(file_name, field=np.nan, temperature=np.nan):
    """
    Function for loading data from the Mazin Lab widesweep LabView GUI.
    Args:
        file_name: string
            The file name with the data.
        field: float (optional)
            The field at the time of the data taking. np.nan is used if not
            provided.
        temperature: float (optional)
            The temperature at the time of the data taking. np.nan is used if
            not provided.
    Returns:
        f: np.ndarray
            The frequency data in GHz.
        z: np.ndarray
            The complex scattering parameter data.
        attenuation: float
            The attenuation for the data for a 0 dBm power calibration.
        field: float
            The field for the data.
        temperature: float
            The temperature for the data.
    """
    with open(file_name, "r") as f:
        header = f.readline()
    attenuation = -float(header.strip().split("\t")[1])
    f, zr, zi = np.loadtxt(file_name, skiprows=3, unpack=True)
    z = zr + 1j * zi
    return f, z, attenuation, field, temperature


def copper_mountain_c1220_widesweep(file_name, attenuation=np.nan, field=np.nan, temperature=np.nan):
    """
    Function for loading data from the Copper Mountain C1220 VNA output file.
    Args:
        file_name: string
            The file name with the data.
        attenuation: float (optional)
            The attenuation at the time of the data taking. np.nan is used if
            not provided.
        field: float (optional)
            The field at the time of the data taking. np.nan is used if not
            provided.
        temperature: float (optional)
            The temperature at the time of the data taking. np.nan is used if
            not provided.
    Returns:
        f: np.ndarray
            The frequency data in GHz.
        z: np.ndarray
            The complex scattering parameter data.
        attenuation: float
            The attenuation for the data for a 0 dBm power calibration.
        field: float
            The field for the data.
        temperature: float
            The temperature for the data.
    """
    f, zr, zi = np.loadtxt(file_name, skiprows=3, unpack=True, delimiter=",")
    z = zr + 1j * zi
    return f * 1e-9, z, attenuation, field, temperature


def mkidreadout2_widesweep(file_name, field=np.nan, temperature=np.nan):
    """
    Function for loading data from the Mazin Lab generation 2 digital readout
    widesweep file. Frequencies are ordered by segment and may be overlapping.
    Args:
        file_name: string
            The file name with the data.
        field: float (optional)
            The field at the time of the data taking. np.nan is used if not
            provided.
        temperature: float (optional)
            The temperature at the time of the data taking. np.nan is used if
            not provided.

    Returns:
        f: np.ndarray
            The frequency data in GHz.
        z: np.ndarray
            The complex scattering parameter data.
        attenuation: np.ndarray
            The attenuation for the data for a 0 dBm power calibration.
        field: float
            The field for the data.
        temperature: float
            The temperature for the data.
    """
    npz = np.load(file_name)
    shape = npz['I'].shape
    if len(shape) == 2:  # only one attenuation so we add the dimension back in
        z = npz['I'].reshape((1, *shape)) + 1j * npz['Q'].reshape((1, *shape))
    else:
        z = npz['I'] + 1j * npz['Q']
    z = z.reshape((z.shape[0], z.shape[1] * z.shape[2]))
    f = np.broadcast_to(npz['freqs'].ravel() * 1e-9, z.shape)
    return f, z, npz['atten'], field, temperature


def mkidreadout2_widesweep_indices(f, z, metadata_file=None, fr=None):
    """
    Returns an array of indices that correspond to the resonator locations from
    the data returned by mkidreadout2_widesweep().

    Args:
        f: numpy.ndarray
            The frequency data in GHz.
        z: numpy.ndarray
            The complex scattering parameter data.
        metadata_file: string
            The file name for the widesweep metadata file. If not supplied, the
            resonator frequencies are taken from fr.
        fr: iterable of floats
            The resonant frequencies in GHz for which to output the indices. If
            not supplied, the resonator frequencies are loaded from the
            metadata file.

    Returns:
        indices: numpy.ndarray
            The indices of f corresponding to resonator locations.
    """
    if z.base is None:
        message = ("z must be a view of the original array loaded by mkidreadout2_widesweep(). "
                   "Don't copy it before using it in this function.")
        raise ValueError(message)

    # load in the metadata
    if metadata_file is None and fr is None:
        raise ValueError("Supply a metadata file or frequency list.")
    if fr is not None:
        fr = np.array(fr)
    else:
        metadata = np.loadtxt(metadata_file)
        if metadata.shape[1] == 3:
            fr = metadata[:, 1] * 1e-9
        elif metadata.shape[1] == 9:
            fr = metadata[:, 5] * 1e-9
        else:
            raise IOError("Unknown file format for {}".format(metadata_file))

    # find the indices
    window_f_centers = f.reshape(z.base.shape)[0, :, z.base.shape[-1] // 2]
    window_index = np.argmin(np.abs(fr - window_f_centers[:, np.newaxis]), axis=0)  # index of closest window to fr
    f_windows = f.reshape(z.base.shape)[0, window_index, :]  # (len(fr), window size)
    f_index = np.argmin(np.abs(fr[:, np.newaxis] - f_windows), axis=1)  # index of closest frequency to fr in the window
    indices = f_index + window_index * z.base.shape[-1]
    return indices


def sonnet_touchstone_loop(file_name):
    """
    Function for handling Loop data from sonnet. The file_name must
    point to a version 1 or 2 touchstone file.
    """
    with open(file_name) as fid:
        values = []
        while True:
            line = fid.readline()

            # Exit the while loop if we run out of lines.
            if not line:
                break

            # Remove the comments, leading or trailing whitespace, and make
            # everything lowercase.
            line = line.split('!', 1)[0].strip().lower()

            # Skip the line if it was only comments.
            if len(line) == 0:
                continue

            # Skip the version line.
            if line.startswith('[version]'):
                continue

            # Skip the number of ports line.
            if line.startswith('[number of ports]'):
                continue

            # Skip the data order line since it's the same for all Sonnet
            # outputs.
            if line.startswith('[two-port data order]'):
                continue

            # Skip the number of frequencies line.
            if line.startswith('[number of frequencies]'):
                continue

            # skip the network data line.
            if line.startswith('[network data]'):
                continue

            # Skip the end line.
            if line.startswith('[end]'):
                continue

            # Note the options.
            if line[0] == '#':
                options = line[1:].strip().split()
                # fill the option line with the missing defaults
                options.extend(['ghz', 's', 'ma', 'r', '50'][len(options):])
                unit = options[0]
                parameter = options[1]
                if parameter != 's':
                    raise ValueError("The file must contain the S parameters.")
                data_format = options[2]
                continue

            # Collect all of the values making sure they are the right length.
            data = [float(v) for v in line.split()]
            if len(data) != 9:
                raise ValueError("The data does not come from a two port "
                                 "network.")
            values.extend(data)

    # Reshape into rows of f, s11, s21, s12, s22, s11, ...
    values = np.asarray(values).reshape((-1, 9))

    # Extract the frequency in GHz.
    multiplier = {'hz': 1.0, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9}[unit]
    f = values[:, 0] * multiplier / 1e9  # always in GHz

    # Extract the S21 parameter.
    if data_format == "ri":
        values_complex = values[:, 1::2] + 1j * values[:, 2::2]
        z = values_complex[:, 1]  # S21
    elif data_format == "ma":
        mag = values[:, 1::2]
        angle = np.pi / 180 * values[:, 2::2]
        values_complex = mag * np.exp(1j * angle)
        z = values_complex[:, 1]  # S21
    else:  # == "db"
        mag_db = values[:, 1::2]
        angle = np.pi / 180 * values[:, 2::2]
        values_complex = 10**(mag_db / 20.0) * np.exp(1j * angle)
        z = values_complex[:, 1]  # S21

    data = {"f": f, "z": z, "attenuation": np.nan, "field": np.nan,
            "temperature": np.nan, "imbalance": None, "offset": None,
            "metadata": {}}
    return data
