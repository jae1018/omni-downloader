#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import requests
import pandas as pd
import numpy as np
from io import StringIO




class OMNIDownloader:
    """
    Downloads OMNI 1-min data from NASA GSFC OMNIWeb using POST requests.
    
    Use the VAR_ID_MAP class dict to determine which
    variable you'd like to download. Variables related to time (e.g.
    year, day [as in day-of-year, or DOY], hour, and/or minute) are always
    downloaded. (minute only for high res, otherwise ignored)
    
    HIGH RES info:
    --------------
        The VAR_ID_MAP_HIGH_RES was determined from the text information
        at the following address:
            https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/hroformat.txt
        The repository for all the high-res OMNI data (5min and 1min) can
        in general be found at:
            https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/
    
    LOW RES info:
    -------------
        The VAR_ID_MAP_LOW_RES was determined from a separate text file 
        for the low res data:
            https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
    """

    VAR_ID_MAP_HIGH_RES = {
        "year":              0,   # Year
        "day":               1,   # Day of Year
        "hour":              2,   # Hour
        "minute":            3,   # Minute
        "imf_sc_id":         4,   # ID for IMF spacecraft
        "plasma_sc_id":      5,   # ID for SW Plasma spacecraft
        "n_imf_pts":         6,   # Number of points in IMF averages
        "n_plasma_pts":      7,   # Number of points in Plasma averages
        "pct_interp":        8,   # Percent interpolated data
        "timeshift_s":       9,   # Timeshift, seconds
        "rms_timeshift":    10,   # RMS of timeshift
        "rms_pfn":          11,   # RMS of phase front normal
        "dbot1_s":          12,   # Time between observations, seconds (DBOT1)
        "b_avg":            13,   # Field magnitude average, nT
        "bx":               14,   # Bx, nT (GSE, GSM)
        "by_gse":           15,   # By, nT (GSE)
        "bz_gse":           16,   # Bz, nT (GSE)
        "by_gsm":           17,   # By, nT (GSM)
        "bz_gsm":           18,   # Bz, nT (GSM)
        "rms_b_scalar":     19,   # RMS SD of B scalar, nT
        "rms_b_vec":        20,   # RMS SD of field vector, nT
        "flow_speed":       21,   # Flow speed, km/s
        "vx":               22,   # Vx, km/s (GSE)
        "vy":               23,   # Vy, km/s (GSE)
        "vz":               24,   # Vz, km/s (GSE)
        "density":          25,   # Proton density, n/cc
        "temp":             26,   # Temperature, K
        "flow_pressure":    27,   # Flow pressure, nPa
        "electric_field":   28,   # Electric field, mV/m
        "plasma_beta":      29,   # Plasma beta
        "alfven_mach":      30,   # Alfven Mach number
        "x_sc_re":          31,   # X spacecraft (GSE), Re
        "y_sc_re":          32,   # Y spacecraft (GSE), Re
        "z_sc_re":          33,   # Z spacecraft (GSE), Re
        "bsn_x_re":         34,   # BSN Xgse location, Re
        "bsn_y_re":         35,   # BSN Ygse location, Re
        "bsn_z_re":         36,   # BSN Zgse location, Re
        "ae":               37,   # AE-index, nT
        "al":               38,   # AL-index, nT
        "au":               39,   # AU-index, nT
        "sym_d":            40,   # SYM/D index, nT
        "sym_h":            41,   # SYM/H index, nT
        "asy_d":            42,   # ASY/D index, nT
        "asy_h":            43,   # ASY/H index, nT
        "pcn":              44,   # PC(N) index
        "m_ms":             45    # Magnetosonic Mach number
    }

    VAR_ID_MAP_LOW_RES = {
        "year":              0,   # Year (e.g., 1963, 1964, ...)
        "day":               1,   # Decimal Day of Year (1 = Jan 1)
        "hour":              2,   # Hour (0–23)
        "imf_sc_id":         4,   # IMF spacecraft ID
        "plasma_sc_id":      5,   # SW plasma spacecraft ID
        "n_imf_pts":         6,   # Number of points in IMF average
        "n_plasma_pts":      7,   # Number of points in plasma average
        "b_avg":             8,   # |B| magnitude average (nT)
        "b_vec_mag":         9,   # Magnitude of average vector field (nT)
        "b_lat_angle":      10,   # Latitudinal angle of average field (°)
        "b_long_angle":     11,   # Longitudinal angle of average field (°)
        "bx":               12,   # Bx component (GSE/GSM) (nT)
        "by_gse":           13,   # By GSE (nT)
        "bz_gse":           14,   # Bz GSE (nT)
        "by_gsm":           15,   # By GSM (nT)
        "bz_gsm":           16,   # Bz GSM (nT)
        "rms_b_scalar":     17,   # RMS SD of |B| magnitude (nT)
        "rms_b_vec":        18,   # RMS SD of B vector (nT)
        "rms_bx":           19,   # RMS SD of Bx component (nT)
        "rms_by":           20,   # RMS SD of By component (nT)
        "rms_bz":           21,   # RMS SD of Bz component (nT)
        "temp":             22,   # Proton temperature (K)
        "density":          23,   # Proton density (cm⁻³)
        "flow_speed":       24,   # Plasma flow speed (km/s)
        "flow_long_angle":  25,   # Plasma flow longitude angle (°)
        "flow_lat_angle":   26,   # Plasma flow latitude angle (°)
        "alpha_ratio":      27,   # Alpha/proton ratio (Na/Np)
        "flow_pressure":    28,   # Flow pressure (nPa)
        "sigma_temp":       29,   # RMS SD of temperature (K)
        "sigma_density":    30,   # RMS SD of density (cm⁻³)
        "sigma_speed":      31,   # RMS SD of speed (km/s)
        "sigma_phi_v":      32,   # RMS SD of flow longitude angle (°)
        "sigma_theta_v":    33,   # RMS SD of flow latitude angle (°)
        "sigma_alpha_ratio":34,   # RMS SD of alpha/proton ratio
        "electric_field":   35,   # Electric field (mV/m)
        "plasma_beta":      36,   # Plasma beta (dimensionless)
        "alfven_mach":      37,   # Alfven Mach number (dimensionless)
        "kp":               38,   # Kp index (scaled, e.g., 3+ = 33)
        "sunspot_number":   39,   # Sunspot number (version 2)
        "sym_d":            40,   # Dst index (SYM/D), nT
        "ae":               41,   # AE index (nT)
        "flux_gt_1MeV":     42,   # Proton flux >1 MeV (1/cm²·s·sr)
        "flux_gt_2MeV":     43,   # Proton flux >2 MeV (1/cm²·s·sr)
        "flux_gt_4MeV":     44,   # Proton flux >4 MeV (1/cm²·s·sr)
        "flux_gt_10MeV":    45,   # Proton flux >10 MeV (1/cm²·s·sr)
        "flux_gt_30MeV":    46,   # Proton flux >30 MeV (1/cm²·s·sr)
        "flux_gt_60MeV":    47,   # Proton flux >60 MeV (1/cm²·s·sr)
        "flag":             48,   # Data quality flag (-1 to 6)
        "ap":               49,   # ap index (nT)
        "f107":             50,   # F10.7 solar radio flux index (sfu)
        "pcn":              51,   # PC(N) index (dimensionless)
        "al":               52,   # AL index (nT)
        "au":               53,   # AU index (nT)
        "m_ms":             54    # Magnetosonic Mach number (dimensionless)
    }

    
    # made into dict due to annoyance of 'min' for '1min' but '5min' for '5min'
    TIME_RES_STR = {
        '1min' : 'min',
        '5min' : '5min',
        'hour' : 'hour'
    }
    
    # The Year marker also changes depending on the time resolution (wtf, why?!?)
    YEAR_MARKER = {
        '1min' : 'YYYY',
        '5min' : 'YYYY',
        'hour' : 'YEAR'
    }



    def __init__(
            self,
            save_dir: str         = "omniweb_data",
            variables: list[str]  = ['sym_h'],
            time_res: str         = '1min'
    ):
        self.url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
        self.save_dir  = save_dir
        
        # check that time res is legal
        if time_res not in self.TIME_RES_STR:
            raise ValueError(f"Given time_res '{time_res}' not among allowed values:"
                             f" {list(self.TIME_RES_STR.keys())}")
        self.time_res  = time_res
        
        # check that variables are legal
        var_dict_for_time_res = self._get_var_id_dict()
        for variable in variables:
            if variable not in var_dict_for_time_res:
                raise ValueError(f"Given variable '{variable}' not among allowed values:"
                                 f" {list(var_dict_for_time_res.keys())}")
        self.variables = variables
        
        # make path to would-be output files
        os.makedirs(save_dir, exist_ok=True)
    
    
    
    def get_allowed_low_res_vars(self):
        """
        Return list of strs of vars that can be downloaded at low res
        """
        return self.VAR_ID_MAP_LOW_RES.keys()
    
    
    
    def get_allowed_high_res_vars(self):
        """
        Return list of strs of vars that can be downloaded at high res
        """
        return self.VAR_ID_MAP_HIGH_RES.keys()
    
    
    
    def _get_var_id_dict(
                self
    ) -> dict[str, int]:
        """
        Retrieves the VAR_ID dict based on the time res given in the
        OMNIWebDownloader constructor

        PARAMETERS
        ----------
        None

        Returns
        -------
        dict[str, int]
            The correct VAR_ID dict for use with the given time res
        """
        
        var_dict_for_time_res = None
        if self.time_res in ['1min', '5min']:
            var_dict_for_time_res = self.VAR_ID_MAP_HIGH_RES
        else:
            var_dict_for_time_res = self.VAR_ID_MAP_LOW_RES
        
        return var_dict_for_time_res
        
    
        
    def _construct_payload(
                self,
                start_time: str,
                end_time: str,
    ) -> dict[str, str]:
        """
        Creates the "payload" (body of the link) for the download request
        
        Parameters
        ----------
        start_time: str
            start_time for download as str (e.g. 2012010100 [YYYYMMDDHH])
        end_time: str
            end_time for download as str
        
        Returns
        -------
        Dict with strs mapping to strs
            To be used with requests.post
        """
        
        ### determine correct link info based on vars and time res
        
        # get correct IDs for vars based on class dict
        var_dict_for_time_res = self._get_var_id_dict()
        var_ids = [str(var_dict_for_time_res[v]) for v in self.variables if v in var_dict_for_time_res]
        
        # get correct spacecraft name based on time res
        sc_str = 'omni'
        if self.time_res == 'hour':
            sc_str += '2'
        if (self.time_res == '1min') or (self.time_res == '5min'):
            sc_str += '_' + self.TIME_RES_STR[self.time_res]
            
        # get right time res str
        res_str = self.TIME_RES_STR[self.time_res]
        
        # if 1min / 5min, then need hour info ...
        # ... but only need YYYYMMDD info if hourly
        start_date_str = start_time
        end_date_str = end_time
        if self.time_res == 'hour':
            start_date_str = start_date_str[:8]
            end_date_str = end_date_str[:8]
        
        # construct dict
        payload = {
            "activity": "retrieve",
            "res": res_str,
            "spacecraft": sc_str,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "scale": "Linear",
            "table": "1",  # Request plain-text table
        }
        
        # get the vars as a list of var ids (or empty if given None)
        for v in var_ids:
            payload["vars"] = payload.get("vars", []) + [v]
        
        return payload
        



    def _download(
                self, 
                start_time: str, 
                end_time: str, 
                filepath: str
    ) -> str:
        """
        Downloads data from OMNIWeb between start and end in YYYYMMDDHH format.
        Saves to a local file and returns the filepath.
        
        The way to correctly prepare the link was determined from:
        https://omniweb.gsfc.nasa.gov/html/command_line_sample_high.txt
        
        Note that the curl command example they show, specifically:
        For curl command:
        > curl -d  "activity=retrieve&res=min&spacecraft=omni_min
                   &start_date=1998050100&end_date=1998050111&vars=13&vars=21
                   &scale=Linear&ymin=&ymax=&view=0&charsize=&xstyle=0&ystyle=0
                   &symbol=0&symsize=&linestyle=solid&table=0&imagex=640
                   &imagey=480&color=&back=" https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi > test_curl_high.txt
        (note that newlines are just for human readability)
        
        can be made using requests.post(url, data=payload_dict) where
          url = https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi
        and
          payload_dict = payload (shown below)
          
        PARAMETERS
        ----------
        start_time: str
            start_time for download as str (e.g. 2012010100 [YYYYMMDDHH])
        end_time: str
            end_time for download as str
        filepath: str
            The path of the file to save the downloaded data to (folder + filename).
        
        RETURNS
        -------
        Path to saved file (str)
        """
        
        # create payload for link
        payload = self._construct_payload(start_time, end_time)

        # download data
        response = requests.post(self.url, data=payload)

        # raise Error if download not successfull
        if not response.ok:
            raise RuntimeError(f"Request failed with status {response.status_code}\n{response.text[:500]}")

        # Otherwise, save downlaoded file contents to file at filepath
        with open(filepath, "w") as f:
            f.write(response.text)
        print(f"Saved to {filepath}")
        return filepath
    
    
    
    def _convert_omni_999s_to_nans(
            df: pd.DataFrame, 
            ignore_cols: list[str] = None
    ) -> pd.DataFrame:
        """
        (Abstract Method - called like OMNIWebDownloader.<func_name>)
        
        This function converts the hard-coded fill values of 999, 9999.99
        (and so on) to numpy nans. NaNs in OMNI data have usually all 9's 
        depending on their Fortran number type (e.g. I4 NaN would be 9999 
        and F6.2 would be 9999.99)
        
        Overall idea is to convert all columns of interest to rounded-down
        ints (so that floats like 999.9 and 9999.99 become just 999 and 9999)
        then check for each 999 (and higher) instance individually.
        
        NOTE:
            numpy.nan cannot be specified in integer format!!! So integer
            columns that have a fill value replaced with NaN are converted
            to float type!!!
        
        Parameters
        ----------
        df - pandas dataframe
            dataframe of downloaded OMNI data
        ignore_cols (optional) - list of str (default empty list)
            The list of columns which should not be parsed for NaNs
        
        Returns
        -------
        pandas dataframe with 999 fill values (and higher) replaced with np.nans
        """
        
        nan_vals_as_ints = [ int('9'*i) for i in range(3,8) ]
        if ignore_cols is None: ignore_cols = []
        
        nan_masks = []
        for col in df:
            
            
            # if col is in ignore_cols, just make mask of all False ...
            if col in ignore_cols:
                nan_masks_for_col = np.full(df.shape[0], False)
                
                
            # ... otherwise, determine mask
            else:
                
                # convert data to ints so that only need to check 999, 9999,
                # and so on (then no need to check for 999.99, etc)
                col_as_ints = df[col].astype(int)
                
                nan_masks_for_col = []
                for nan_val_int in nan_vals_as_ints:
                    nan_masks_for_col.append( (col_as_ints == nan_val_int).values )
                
                # convert to row x nan_vals_as_ints array and if row had pt
                # that corresponded to nan_val for any of the nan_vals, assign
                # that point as true for overall mask
                nan_masks_for_col = np.array(nan_masks_for_col).T.any(axis=1)
            
            
            nan_masks.append( nan_masks_for_col )
        
        # make row x cols array of masks
        nan_masks = np.array(nan_masks).T
        
        # any value with True set to np.nan
        df[nan_masks] = np.nan
        
        return df



    def _parse(
            self, 
            filepath: str
    ) -> pd.DataFrame:
        """
        Parses the downloaded OMNIWeb plain-text file into a pandas DataFrame.
        Looks for the 'YYYY' header and '</pre>' footer to extract the data block.
        
        An example downloaded file (without dashes) looks like this:
        
        ----------
        <HTML>
        <HEAD><TITLE>OMNIWeb Results</TITLE></HEAD>
        <BODY>
        <center><font size=5 color=red>OMNIWeb Plus Browser Results </font></center><br>
        <B>Listing for omni_min data from 2012010100 to 2012020100</B><hr><pre>Selected parameters:
         1 SYM/H, nT
         2 AE-index, nT
        
        YYYY DOY HR MN    1     2 
        2012   1  0  0   -12   158
        2012   1  0  1   -12   162
        2012   1  0  2   -11   163
           .... lots of data ....
        2012  32  0 57    12    21
        2012  32  0 58    12    24
        2012  32  0 59    13    22
        </pre><hr><HR>
        <ADDRESS>
        <table>
        <tr>
        <td>
        If you have any questions about OMNIWeb Plus Interface,contact:<br>
        <A HREF="mailto:Natalia.E.Papitashvili@nasa.gov">
        Dr. Natalia Papitashvili</A>, Code 672, Greenbelt, MD 20771.
        </ADDRESS>
        </td></tr>
        </table>
        <HR>
        <H6>Last Update: June 16, 2025, NEP.</H6>
        </BODY>
        </HTML>
        ----------
        
        So the data is extracted by looking for the YYYY year signifier first
        and the </pre> string second. The column order (here, SYM/H and AE-index)
        are the same as specified in the constructor with class attr variables.
        """
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Find start: line with 'YYYY'
        year_marker = self.YEAR_MARKER[self.time_res]
        try:
            start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith(year_marker))
        except StopIteration:
            raise ValueError(f"Could not find data header (line starting with '{year_marker}')")

        # Find end: line with '</pre>'
        try:
            end_idx = next(i for i, line in enumerate(lines) if "</pre>" in line.lower())
        except StopIteration:
            end_idx = len(lines)

        # Extract only data block
        data_block = lines[start_idx:end_idx]

        # Join and parse
        df = pd.read_csv(StringIO("".join(data_block)), sep='\s+')
        
        
        ## replace numeric columns titles with vars
        # 3 or 4 columns are for time (Year, DOY, hr, and sometime MN), so 
        # remaining cols are ints to replace
        num_orig_time_cols = (
            (year_marker in df)
            + ('DOY' in df)
            + ('HR' in df)
            + ('MN' in df)
        )
        num_col_names_to_replace = df.shape[1] - num_orig_time_cols
        col_rename_dict = dict(zip(
                # ints start from 1... but they're actually strs!
                (1+np.arange(num_col_names_to_replace)).astype(str), 
                self.variables
        ))
        df.rename(columns=col_rename_dict, inplace=True)
        
            
        ### make the timestamp column
        # Year (either as YYYY or YEAR), DOY, and HR always included ...
        timestamps = pd.to_datetime(
            df[year_marker].astype(str) + df["DOY"].astype(str).str.zfill(3),
            format="%Y%j"
        ) + pd.to_timedelta(df["HR"], unit="h")
        # ... however, MN not always included!
        if 'MN' in df:
            timestamps += pd.to_timedelta(df["MN"], unit="m")
        # then assign to df
        df["time"] = timestamps
        
        
        ## delete the original columns and set timestamp as index
        cols_to_drop = [year_marker,'DOY','HR']
        if 'MN' in df: cols_to_drop.append('MN')
        df.drop(columns=cols_to_drop, inplace=True)
        df.set_index("time", inplace=True)
        
        
        # Replace 999s of various kinds with NaNs
        df = OMNIDownloader._convert_omni_999s_to_nans(df, ignore_cols=['timestamp'])
        
        return df
    
    
    
    def fetch_range(self, start_time: str, end_time: str) -> pd.DataFrame:
        """
        Downloads and parses OMNI data over multiple monthly blocks.
        
        Parameters:
            start_time (str): Start time, e.g. '2013-01-01'
            end_time (str): End time, e.g. '2013-03-01'

        Returns:
            pd.DataFrame: Concatenated DataFrame across all months
        """
        start = pd.Timestamp(start_time)
        end = pd.Timestamp(end_time)
        all_dfs = []

        # Generate 1st-of-month timestamps from start to end
        monthly_starts = pd.date_range(
                            start = start.normalize().replace(day=1),  # floor to 1st of the month, 
                            end   = end.normalize(), 
                            freq  = "MS"
        )

        for t0 in monthly_starts:
            # make it so that end date is end-of-month (and at last possible second)
            # seems pedantic..., but necessary to get download correct without
            # repeated info
            t1 = t0 + pd.offsets.MonthEnd() + pd.Timedelta('23:59:59')

            # Clip to user-specified bounds
            start_str = max(t0, start).strftime("%Y%m%d%H")
            end_str   = min(t1, end).strftime("%Y%m%d%H")

            var_string = "_".join(self.variables)
            yyyymm = t0.strftime("%Y%m")
            filename = f"omni_{var_string}_{yyyymm}.txt"
            filepath = os.path.join(self.save_dir, filename)
            print(f"Fetching {start_str} to {end_str}...")

            # check if file exists, and if it does not, then download ...
            if not os.path.exists(filepath):
                filepath = self._download(
                                    start_time = start_str, 
                                    end_time   = end_str, 
                                    filepath   = filepath
                )

            # ... otherwise, use local file
            else:
                print("Preexisting download file found - retrieving local file")
            
            df = self._parse(filepath)
            all_dfs.append(df)

        print(f"Fetched and parsed {len(all_dfs)} monthly files.")
        df = pd.concat(all_dfs)
        
        # last, slice df according to start_time / end_time (in case mid range)
        time_mask = ( (df.index >= pd.Timestamp(start_time)) 
                      & (df.index < pd.Timestamp(end_time)) )
        return df[time_mask]
        
        



if __name__ == "__main__":

    

    
    # 11111 Example of downloading at 1min resolution 11111
    
    ## Example of downloading BZ_GSM, density, and flow_speed at 1min
    # instantiate OMNIWebDownloader object
    omni_1min = OMNIDownloader(
                    variables = ["bz_gsm", "density", "flow_speed"],
                    time_res = '1min'
    )
    # fetch data for particular range (downloaded in 1 month chunks)
    df_1min = omni_1min.fetch_range("2012-01-01", "2012-03-01")
    # convert to xarray and save (if you like xarray)
    #df.to_xarray().to_netcdf( os.path.join(os.getcwd(),'omni_symh_ae_1min.nc') )
    
    # 11111111111111111111111111111111111111111111111111111
    
    
    
    
    # 22222 Example of downloading at 5min resolution 22222
    
    ## Example of downloading SYM/H and AE at 5min
    #omni = OMNIDownloader(
    #            variables = ["sym_h", "density", "flow_speed"],
    #            time_res = '5min'
    #)
    #df = omni.fetch_range("2012-01-01", "2013-01-01")
    
    # 22222222222222222222222222222222222222222222222222222
    
    
    
    
    # 33333 Example of downloading at 1hour resolution 33333
    
    ## Example of downloading Kp and AE at 1hour
    omni_hour = OMNIDownloader(
                    variables = ["kp", "ae"],
                    time_res = 'hour'
    )
    df_hour = omni_hour.fetch_range("2012-01-01", "2012-03-01")
    
    # 33333333333333333333333333333333333333333333333333333
    
    
    
    
    # 44444 How to merge pandas dataframes at different time resolutions 44444
    
    ## if you wanted to merge the dataframes of the 1min data and 1 hour
    ## data (just repeating hourly values at 1min resolution), then you
    ## can use this
    # Reindex to 1-min index, forward fill
    df_hour_to_1min_aligned = df_hour.reindex(df_1min.index, method='ffill')
    # Then join to the 1-min dataframe
    df_merged = df_1min.join(df_hour_to_1min_aligned)
    
    # 444444444444444444444444444444444444444444444444444444444444444444444444




    # 55555 Vars that can be downloaded can be inferred from class 55555
    # 55555 dicts or by calling couple of functions                55555
    
    # find out high res vars
    omni_1min.get_allowed_high_res_vars()
    # find out low res vars
    omni_1min.get_allowed_low_res_vars()
    
    # 555555555555555555555555555555555555555555555555555555555555555555
        