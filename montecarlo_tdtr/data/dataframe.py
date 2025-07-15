import logging
import os

import pandas as pd
import numpy as np

import scipy.io as sio
import scipy.optimize as optimize
import scipy.interpolate as interpolate

import seaborn as sns
import matplotlib.pyplot as plt

class TDTR:
    def __init__(self, df):
        self.df     = df
    
    def extract_interior(self, t_min, t_max):
        cond = np.logical_and(
            self.df.tdelay > t_min * 1e12,
            self.df.tdelay < t_max * 1e12
        )
        return TDTR(self.df[cond])
    
    def phase_shift(self, phase):
        V_comp  = self.df.Vin + 1.j * self.df.Vout
        V_shift = V_comp * np.exp(1.j * phase / 180 * np.pi)

        df          = self.df.copy()
        df.Vin      = np.real(V_shift)
        df.Vout     = np.imag(V_shift)
        df.Ratio    = -df.Vin / df.Vout
        return TDTR(df)
        
    def export_to_mat(self, path_to_file):
        data  = self.df.to_dict('series')
        mdata = {m: data[m].to_numpy() for m in data.keys()}
        
        sio.savemat(path_to_file, {'Data' : mdata}, oned_as = 'column')
    
    def copy(self):
        return TDTR(self.df)

    def plot(self, t_min = None, t_max = None):
        """ Plots the TDTR data
        Args:
        - t_min (double) : minimum time delay (ps)
        - t_max (double) : maximum time delay (ps)
        """
        fig, ax = plt.subplots()
        sns.lineplot(self.df, x = 'tdelay', y = 'Vin',
                     label = 'in-phase voltage', ax = ax)
        sns.lineplot(self.df, x = 'tdelay', y = 'Vout',
                     label = 'out-of-phase voltage', ax = ax)
        ax.set_ylabel('Voltage [$\mu$V]')
        ax.set_xlabel('tdelay [ps]')
        if t_min is not None and t_max is not None:
            ax.set_xlim([t_min * 1e12, t_max * 1e12])
        ax.grid(which = 'major', linestyle = '-')
        ax.grid(which = 'minor', linestyle = '--')
        ax.legend()
        fig.show()

    def plot_ratio(self, t_min = 100e-12, t_max = 4000e-12):
        """ Plots the Ratio data
        Args:
        - t_min (double) : minimum time delay (ps)
        - t_max (double) : maximum time delay (ps)
        """
        fig, ax = plt.subplots()
        sns.lineplot(self.df, x = 'tdelay', y = 'Ratio',
                     label = 'Ratio', ax = ax)
        ax.set_ylabel('Ratio [a.u.]')
        ax.set_xlabel('tdelay [ps]')
        ax.set_xlim([t_min * 1e12, t_max * 1e12])
        ax.set_ylim([1, self.df.Ratio.max()])
        ax.set_xscale('log')
        ax.grid(which = 'both')
        ax.legend()
        fig.show()

    def out_phase_statistics(self, t_min = -20e-12, t_max = 100e-12,
                             plot = False, verbose = False):
        out_phase = self.extract_interior(t_min, t_max)
        mean_vout = out_phase.df.Vout.mean()
        std_vout = out_phase.df.Vout.std()
        
        SNR = 20 * np.log10(np.abs(mean_vout / std_vout))
        
        if verbose:
            print('SNR %.1f dB' % SNR)

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (6.4, 7.2))
            
            out_phase.df.plot.scatter(x = 'tdelay', y = 'Vout', 
                              grid = True, ax = ax1)
            out_phase.df.Vout.plot.kde(ax = ax2)
            ax1.axhline(mean_vout, color = 'red')
            ax1.axhline(mean_vout + 2 * std_vout, color = 'red', linestyle = '--')
            ax1.axhline(mean_vout - 2 * std_vout, color = 'red', linestyle = '--')

            ax1.set_ylabel('Voltage [µV]')
            ax1.set_xlabel('tdelay [ps]')
            ax1.set_xlim(t_min * 1e12, t_max * 1e12)

            fig.show()
        
        return SNR
    
    def picosecond_acoustics(self, h = 80e-9, polarity = True,
                             use_ratio = True, plot = False,
                             multiple = True):
        c_Al  = 6260 # speed of sound in Aluminium m/s

        # Time delay range to search for the first Al reflection peak
        t_mid = 2 * h / c_Al
        picosecond = self.extract_interior(
            t_mid - 1e-12, t_mid + 1e-12
        )

        if use_ratio:
            if polarity:
                t_peak = picosecond.df.tdelay[picosecond.df.Ratio.idxmax()]
            else:
                t_peak = picosecond.df.tdelay[picosecond.df.Ratio.idxmin()]
        else:
            if polarity:
                t_peak = picosecond.df.tdelay[picosecond.df.Vin.idxmax()]
            else:
                t_peak = picosecond.df.tdelay[picosecond.df.Vin.idxmin()]

        h_Al = t_peak * 1e-12 * c_Al / 2

        if plot:
            if multiple:
                fig, ax = plt.subplots(1, 1, figsize = (9.6, 4.8))
                short = self.extract_interior(t_mid - 15e-12, 2 * t_mid + 15e-12)
            else:
                fig, ax = plt.subplots(1, 1, figsize = (6.4, 4.8))
                short = self.extract_interior(t_mid - 15e-12, t_mid + 15e-12)
            if use_ratio:
                short.df.plot(x = 'tdelay', y = 'Ratio', marker = 's', ax = ax)
                ax.set_ylabel('Ratio [a.u.]')
            else:
                short.df.plot(x = 'tdelay', y = 'Vin', marker = 's', ax = ax)
                ax.set_ylabel('Voltage [µV]')
            
            if multiple:
                ax.set_xlim(t_mid * 1e12 - 10, 2 * t_mid * 1e12 + 10)
                ax.axvline(2 * t_peak, color = 'red', linestyle = '--')
            else:
                ax.set_xlim(t_mid * 1e12 - 10, t_mid * 1e12 + 10)

            ax.set_xlabel('tdelay [ps]')
            ax.axvline(t_peak, color = 'red', linestyle = '--')
            ax.grid()
            fig.show()
        
        return h_Al
    
class Measurement(TDTR):
    def __init__(self, path_to_file):
        # Get file name from path
        self.file_orig = os.path.basename(path_to_file)
        self.path_name = os.path.dirname(path_to_file)

        # Convert .mat file to dictionary
        data    = sio.loadmat(path_to_file)['Data']
        ndata   = {n: data[n][0,0] for n in data.dtype.names}
                
        df = pd.DataFrame.from_dict(
            {n: ndata[n].reshape(len(ndata[n])) for n in ndata.keys()}
        )

        # Assign the dataframe to TDTR object
        super().__init__(df.dropna())
    
    def set_t0(self, t0 = None, t_min = -20e-12, t_max = 20e-12, plot = False,
               use_approximate = False):
        # Finding t0 based on numerical differentiation -- done already in DAQ
        if use_approximate:
            az_idx = np.argmax(np.abs(np.diff(self.df.Vin)))
            self.df.tdelay = (self.df.stagePosition - self.df.stagePosition[az_idx]) * \
                (2 * 12.5e-6 * 1e12) / 3e8

        # t0 is roughly the half-rise in-phase voltage jump
        # ref. Aaron J. Schmidt's thesis
        near_t0     = self.extract_interior(t_min, t_max)
        normalize   = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

        drift = self.df.Vdet / self.df.Vdet.head(1).to_numpy()

        near_t0.df['Vin_norm']   = normalize(near_t0.df.Vin * drift)
        near_t0.df['Ratio_norm'] = normalize(near_t0.df.Ratio)

        interp_func = interpolate.interp1d(
            x = near_t0.df.tdelay,
            y = near_t0.df.Vin_norm
        )

        # Rescale tdelay according to t0
        if t0 is None:
            find_t0 = optimize.bisect(lambda xx:interp_func(xx) - 0.5, t_min / 2 * 1e12, t_max / 2 * 1e12)
            self.df.tdelay = self.df.tdelay - find_t0
        else:
            self.df.tdelay = self.df.tdelay - t0

        # Plot the rising-edge of in-phase voltage
        if plot:
            fig, ax = plt.subplots()
            sns.lineplot(near_t0.df, x = 'tdelay', y = 'Vin_norm', 
                        label = 'Normalized Vin', ax = ax)
            sns.lineplot(near_t0.df, x = 'tdelay', y = 'Ratio_norm',
                        label = 'Normalized ratio', ax = ax)
            try:
                t0 = find_t0
                ax.plot(find_t0, interp_func(find_t0), 'ro')
                ax.annotate("%.1f ps" % find_t0, (find_t0 + 0.1, 0.45))
                ax.axhline(0.5, ls = '--', color = 'red')
            except:
                ax.plot(t0, interp_func(t0), 'ro')
                ax.axvline(t0, ls = '--', color = 'red')
            ax.set_xlim([t_min / 2 * 1e12, t_max/2 * 1e12])
            ax.grid(which = 'major', linestyle = '-')
            ax.grid(which = 'minor', linestyle = '--')
            ax.legend()
            fig.show()
        
        try:
            return find_t0
        except:
            return None
        
    def auto_phase_shift(self, t_min = -20e-12, t_short = 80e-12, 
                         use_residuals = True, method = 'Nelder-Mead'):
        data_n = self.extract_interior(t_min, 0)
        data_p = self.extract_interior(0, t_short)

        if use_residuals:
            res = optimize.minimize(
                self.__phase_shift_residuals,
                0.0,
                args   = (data_n, data_p),
                method = method 
            )
        else:
            res = optimize.minimize(
                self.__phase_shift_del_phase,
                0.0,
                args = (data_n, data_p),
                method = method
            )

        self.phase_sol = res.x[0]
        return self.phase_shift(self.phase_sol, t_min, t_short)
    
    def phase_shift(self, phase, t_min = -20, t_short = 80):
        data_shift = super().phase_shift(phase)

        data_n = data_shift.extract_interior(t_min, 0)
        data_p = data_shift.extract_interior(0, t_short)
        self.__phase_shift_stats(data_n, data_p)

        return data_shift
        
    def __phase_shift_del_phase(self, x, data_n, data_p):
        data_ps = data_p.phase_shift(x)
        data_ns = data_n.phase_shift(x)

        noise_y = np.std(
            np.concatenate((data_ps.df.Vout, data_ns.df.Vout), axis = None)
        )
        del_x   = np.max(data_ps.df.Vin) - np.min(data_ns.df.Vin)
        return (2 * noise_y / del_x * 180.0 / np.pi)

    def __phase_shift_residuals(self, x, data_n, data_p):
        data_ps = data_p.phase_shift(x)
        data_ns = data_n.phase_shift(x)
        
        return np.abs(np.mean(data_ps.df.Vout) -
                      np.mean(data_ns.df.Vout))
    
    def __phase_shift_stats(self, data_n, data_p):
        self.del_phase = self.__phase_shift_del_phase(0.0, data_n, data_p)
        self.residuals = self.__phase_shift_residuals(0.0, data_n, data_p)