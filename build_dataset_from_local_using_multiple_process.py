# Import necessary libraries
import seisbench  # A benchmarking framework for seismology
import seisbench.models as sbm  # For seismic models
import pandas as pd  # Data manipulation library
import os  # Library to interact with the operating system
import glob  # Library to retrieve files/pathnames matching a specified pattern
from obspy import read  # ObsPy function for reading seismic data
from obspy import UTCDateTime  # ObsPy function for handling date and time information
import matplotlib.pyplot as plt  # Library for creating static, animated, and interactive visualizations
import obspy as obs  # A Python framework for processing seismological data
import numpy as np  # Numerical computations library
import matplotlib.pyplot as plt  # A plotting library
import warnings  # Library to handle warnings
from tqdm import tqdm  # Library to show progress bar
import concurrent.futures  # Library to execute function calls asynchronously
import time  # Library to handle time-related tasks
import multiprocessing  # Library to create processes
import seisbench.data as sbd  # For handling seismic data
import seisbench.util as sbu  # For seismic utilities
from obspy import UTCDateTime  # For handling date and time information
from obspy.clients.fdsn import Client  # Client to interact with FDSN web services
from obspy.clients.fdsn.header import FDSNException  # Exception handling for FDSN client
from pathlib import Path  # Library to handle filesystem paths
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore warnings of category 'UserWarning'
pd.options.mode.chained_assignment = None  # Avoids warning for chained assignments in pandas

# Load the dataset
folder_path = '/Users/hao/Downloads/MacKenzie_Mountains/'

df = pd.read_csv(folder_path + "NEDB_catalog_MacKenzie_Mountains.csv", index_col=0)

# Function to read filenames from a directory
def read_filenames(root_folder):
    filenames = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames
# Get the filenames of the waveforms
mseed_folder = folder_path + "mseed_2017_2019"

# Load the dataset again 
filenames = read_filenames(mseed_folder)
df = pd.read_csv(folder_path + "NEDB_catalog_MacKenzie_Mountains.csv", index_col=0)
base_path = Path(".")
metadata_path = base_path / "metadata.csv"
waveforms_path = base_path / "waveforms.hdf5"

# Function to process an individual file
def process_file(filename):
    if filename[-5:] == 'mseed' and os.path.isfile(filename):
        # load waveform from local drive
        st = read(filename)
        if len(st) == 0:
            # No waveform data available
            return
        # pre-processing
        # st = st.detrend('demean')
        st = st.detrend('linear')
        st = st.taper(max_percentage=0.05, type='cosine')
        st = st.filter('bandpass', freqmin=3.0, freqmax=18.0, corners=2, zerophase=True)
        st = st.resample(100)
        # read phase
        if filename[-7:-6] != 'S' and filename[-7:-6] != 'P':
            phase_hint = filename[-8:-6]
        else:
            phase_hint = filename[-7:-6]
        # read event information
        mask = (df['station'] == st[0].stats.station) & (df['phase'] == phase_hint) & (
                df['network'] == st[0].stats.network) \
               & (df['arrival_time'] > st[0].stats.starttime) & (df['arrival_time'] < st[0].stats.endtime)
        if mask.sum() == 0:
            return
        # event exists
        # 获取文件名
        basename = os.path.basename(filename)
        # 分离文件名和扩展名
        name, ext = os.path.splitext(basename)
        # 分割文件名并获取ID部分
        id = name.split('_', 1)[0]
        # create event_params
        event_params = {}
        event_params['source_id'] = id
        event_params['source_origin_time'] = df[mask]['origin time'].values[0]
        event_params['source_origin_uncertainty_sec'] = None
        event_params['source_latitude_deg'] = df[mask]['srcLat (deg)'].values[0]
        event_params['source_latitude_uncertainty_km'] = None
        event_params['source_longitude_deg'] = df[mask]['srcLon (deg)'].values[0]
        event_params['source_longitude_uncertainty_km'] = None
        event_params['source_depth_km'] = df[mask]['srcDepth (km)'].values[0]
        event_params['source_depth_uncertainty_km'] = None
        event_params['source_magnitude'] = df[mask]['mag Sol'].values[0]
        event_params['source_magnitude_uncertainty'] = None
        event_params['source_magnitude_type'] = df[mask]['mag Sol tp'].values[0]
        event_params['source_magnitude_author'] = 'NRCan'
        event_params['split'] = 'train'
        # create trace_params
        trace_params = {}
        trace_params['station_network_code'] = st[0].stats.network
        trace_params['station_code'] = st[0].stats.station
        trace_params['trace_channel'] = df[mask]['channel'].values[0][:-1]
        trace_params['station_location_code'] = None
        # parse information from mseed stream
        sampling_rate = st[0].stats.sampling_rate
        # Check that the traces have the same sampling rate
        assert all(trace.stats.sampling_rate == sampling_rate for trace in st)

        actual_t_start, data, _ = sbu.stream_to_array(
            st,
            component_order='ZNE',
        )


        trace_params["trace_sampling_rate_hz"] = sampling_rate
        trace_params["trace_start_time"] = str(actual_t_start)

        sample = (UTCDateTime(df[mask].iloc[0]['arrival_time']) - actual_t_start) * sampling_rate
        trace_params[f"trace_{phase_hint}_arrival_sample"] = int(sample)
        trace_params[f"trace_{phase_hint}_status"] = 'manual'

        # detect other phase in same trace
        new_mask = (df['station'] == st[0].stats.station) & (df['phase'] != phase_hint) & (
                df['network'] == st[0].stats.network) \
                   & (df['arrival_time'] > st[0].stats.starttime) & (df['arrival_time'] < st[0].stats.endtime)
        if new_mask.any():
            # add new phase
            new_phase = df[new_mask]['phase'].values[0]
            new_sample = (UTCDateTime(df[new_mask].iloc[0]['arrival_time']) - actual_t_start) * sampling_rate
            trace_params[f"trace_{new_phase}_arrival_sample"] = int(new_sample)
            trace_params[f"trace_{new_phase}_status"] = 'manual'
        return [event_params, trace_params, data]

# Main execution
if __name__ == '__main__':
    # read the csv file
    start_index = 0  # start index of the dataframe
    end_index = len(filenames)
    print("working on the index from " + str(start_index) + " to " + str(end_index))
    with multiprocessing.Pool() as pool:
        results = pool.map(process_file, filenames)
    # Iterate over mseed files, write to SeisBench format
    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:

        # Define data format
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }
        for result in results:
            if result is None:
                continue
            event_params, trace_params, data = result
            writer.add_trace({**event_params, **trace_params}, data)
    print("finished!")

