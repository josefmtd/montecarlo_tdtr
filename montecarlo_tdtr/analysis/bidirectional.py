import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

class SysParam:
    def __init__(self, r_pump, r_probe, P_pump, P_probe, tau_rep = 1 / 75.8e6):
        self.r_pump     = r_pump
        self.r_probe    = r_probe
        self.P_pump     = P_pump
        self.P_probe    = P_probe
        self.tau_rep    = tau_rep

class DutParam:
    def __init__(self, Label, Layer, Lambda, C, h):
        self.Label  = Label
        self.Layer  = Layer
        self.Lambda = Lambda
        self.C      = C
        self.h      = h
        self.eta    = np.ones(len(Lambda))

        # Hard-coded aluminium transducer
        self.X_heat  = np.arange(1, 36, 5) * 1e-9
        self.AbsProf = np.exp(-4 * np.pi * 4.9 * self.X_heat / 400e-9)
        self.X_temp  = 1e-9

class Bidirectional:
    def __init__(self, SysParam):
        self.r_pump     = np.array([SysParam.r_pump])
        self.r_probe    = np.array([SysParam.r_probe])
        self.tau_rep    = SysParam.tau_rep
        self.P_pump     = SysParam.P_pump
        self.P_probe    = SysParam.P_probe

        self.f          = []
        self.data_raw   = []
        self.tdelay     = []
        self.ratio      = []

        self.__tdelay_raw = []
        self.__ratio_raw  = []
        self.__vin_raw    = []
        self.__vout_raw   = []

    def set_parameters(self, i_Lambda, i_C, i_h,
            tdelay_min = 100e-12, tdelay_max = 4e-9):
        self.i_Lambda = np.array(i_Lambda)
        self.i_C      = np.array(i_C)
        self.i_h      = np.array(i_h)

        self.tdelay_min = tdelay_min
        self.tdelay_max = tdelay_max
    
    def set_sample_parameters(self, DutParam):
        self.Label    = DutParam.Label
        self.Layer    = DutParam.Layer
        self.Lambda   = np.array(DutParam.Lambda)
        self.C        = np.array(DutParam.C)
        self.h        = np.array(DutParam.h)
        self.eta      = np.array(DutParam.eta)

        self.X_heat   = np.array(DutParam.X_heat)
        self.AbsProf  = np.array(DutParam.AbsProf)
        self.X_temp   = DutParam.X_temp

    def add_measurement_data(self, f, Measurement):
        self.f.append(f)
        self.data_raw.append(Measurement)
        data = Measurement.extract_interior(
            self.tdelay_min, self.tdelay_max
        )
        
        # NumPy variables
        self.ratio.append(data.df.Ratio.to_numpy())
        self.tdelay.append(data.df.tdelay.to_numpy() * 1e-12)
        
        self.__tdelay_raw.append(Measurement.df.tdelay.to_numpy() * 1e-12)
        self.__ratio_raw.append(Measurement.df.Ratio.to_numpy())
        self.__vin_raw.append(Measurement.df.Vin.to_numpy())
        self.__vout_raw.append(Measurement.df.Vout.to_numpy())
    
    def __split_layers(self, X_heat, X_temp):
        hsum    = np.cumsum(self.h)
        I_heatA = np.sum(np.ceil(hsum) / X_heat == 1)
        I_temp  = np.sum(np.ceil(hsum) / X_temp == 1)

        if I_heatA == 0:
            X_heatL = X_heat
        else:
            X_heatL = X_heat - hsum[I_heatA - 1]

        if I_temp == 0:
            X_tempL = X_temp
        else:
            X_tempL = X_temp - hsum[I_heatA - 1]
        
        split_layer = False
        if X_heatL == 0:
            I_heat = I_heatA
        elif X_heatL == self.h[I_heatA]:
            I_heat = I_heatA + 1
        else:
            split_layer = True
            Lambda = np.concatenate(
                (self.Lambda[:I_heatA], self.Lambda[I_heatA], self.Lambda[I_heatA:]),
                axis = None
            )
            C = np.concatenate(
                (self.C[:I_heatA], self.C[I_heatA], self.C[I_heatA:]),
                axis = None
            )
            eta = np.concatenate(
                (self.eta[:I_heatA], self.eta[I_heatA], self.eta[I_heatA:]),
                axis = None
            )
            h = np.concatenate(
                (self.h[:I_heatA], X_heatL, self.h[I_heatA] - X_heatL, self.h[I_heatA + 1:]),
                axis = None
            )
            I_heat  = I_heatA + 1

        if X_temp >= X_heat and split_layer:
            I_temp  = I_temp + 1
            X_tempL = X_tempL - X_heatL
        
        LCTE        = (Lambda, C, h, eta)
        Coordinates = (I_heat, X_heatL, I_temp, X_tempL)
        return LCTE, Coordinates

    def __calculate_down_matrix(self, start_layer, end_layer, Lambda,
                                D, h, eta, kterm2, omega):
        q2 = np.outer(np.ones(self.Nint), 1j * omega / D[start_layer])
        un = np.sqrt(eta[start_layer] * kterm2 + q2)
        gamman = Lambda[start_layer] * un

        Bplus   = np.zeros((self.Nint, self.Nfreq))
        Bminus  = np.ones((self.Nint, self.Nfreq))

        for n in np.arange(start_layer, end_layer, -1):
            q2 = np.outer(np.ones(self.Nint), 1j * omega / D[n - 1])
            unminus = np.sqrt(eta[n - 1] * kterm2 + q2)
            gammanminus = Lambda[n - 1] * unminus

            # Temporary variables
            AA = gammanminus + gamman
            BB = gammanminus - gamman
            temp1 = AA * Bplus + BB * Bminus
            temp2 = BB * Bplus + AA * Bminus

            # Calculate upper layer B matrix
            expterm = np.exp(unminus * h[n - 1])
            Bplus = (0.5 / (gammanminus * expterm)) * temp1
            Bminus = 0.5 / (gammanminus) * expterm * temp2

            # For numerical stability if one of the layer is very thick or resistive
            # FIX: if penetration depth is smaller than layer set to semi-inf
            penetration_logic = np.greater(h[n - 1] * np.abs(unminus), 100)
            Bplus[penetration_logic] = 0
            Bminus[penetration_logic] = 1

            gamman = gammanminus
        
        return (gamman, Bplus, Bminus)

    def __calculate_up_matrix(self, start_layer, end_layer, Lambda,
                              D, h, eta, kterm2, omega):
        q2 = np.outer(np.ones(self.Nint), 1j * omega / D[start_layer])
        un = np.sqrt(eta[start_layer] * kterm2 + q2)
        gamman = Lambda[start_layer] * un

        Bplus = np.exp(-2 * un * h[start_layer])
        Bminus = np.ones((self.Nint, self.Nfreq))


        for n in np.arange(start_layer, end_layer, 1):
            q2 = np.outer(np.ones(self.Nint), 1j * omega / D[n])
            unplus = np.sqrt(eta[n] * kterm2 + q2)
            gammanplus = Lambda[n] * unplus

            # Temporary variables
            AA = gammanplus + gamman
            BB = gammanplus - gamman
            temp1 = AA * Bplus + BB * Bminus
            temp2 = BB * Bplus + AA * Bminus
            expterm = np.exp(un * h[n])
            Bplus = (0.5 / (gammanplus * expterm)) * temp1
            Bminus = 0.5 / (gammanplus) * expterm * temp2

            # For numerical stability if one of the layers is very thick or resistive
            # FIX: if penetration depth is smaller than the layer, set to semi-inf
            penetration_logic = np.greater(h[n] * np.abs(unplus), 100)
            Bplus[penetration_logic] = 0
            Bminus[penetration_logic] = 1

            gamman = gammanplus
            un = unplus
        
        return (gamman, Bplus, Bminus)
        
    def __calculate_integrand(self, kvectin, freq, X_heat):
        LCTE, Coordinates = self.__split_layers(X_heat, self.X_temp)
        
        Lambda  = LCTE[0]
        C       = LCTE[1]
        h       = LCTE[2]
        eta     = LCTE[3]

        I_heat  = Coordinates[0]
        X_heatL = Coordinates[1]
        I_temp  = Coordinates[2]
        X_tempL = Coordinates[3]

        Nlayers = len(Lambda)

        self.Nfreq  = len(freq)
        self.Nint   = len(kvectin)
        kvect       = np.outer(kvectin, np.ones(self.Nfreq))

        # Calculate D, omega, kterm2
        D       = Lambda / C
        omega   = 2 * np.pi * freq
        kvect2  = kvect ** 2
        kterm2  = 4 * np.pi ** 2 * kvect2

        # Calculating down matrix
        gamma_Iheat, alpha_down, beta_down = self.__calculate_down_matrix(
            Nlayers - 1, I_heat, Lambda, D, h, eta, kterm2, omega
        )

        _, Bplus, Bminus = self.__calculate_up_matrix(
            0, I_heat - 1, Lambda, D, h, eta, kterm2, omega
        )

        if I_heat == 0:
            gamma_Iheatminus = np.nan
            alpha_up         = np.nan
            beta_up          = np.nan
            print('Program does not work for X_heat = 0! Use "Surfaceheating" model instead!!')
        else:
            q2 = np.outer(np.ones(self.Nint), 1j * omega / D[I_heat - 1])
            un = np.sqrt(eta[I_heat - 1] * kterm2 + q2)
            gamma_Iheatminus = Lambda[I_heat - 1] * un
            alpha_up = Bplus
            beta_up = Bminus

        # Calculate B1 and BN
        BNminus = -(alpha_up + beta_up) / (
            gamma_Iheat * (alpha_down - beta_down) * (alpha_up + beta_up) +
            gamma_Iheatminus * (alpha_down + beta_down) * (alpha_up - beta_up)
        )

        B1minus = (alpha_down + beta_down) * BNminus / (alpha_up + beta_up)

        q2 = np.outer(np.ones(self.Nint), 1j * omega / D[0])
        u1 = np.sqrt(eta[0] * kterm2 + q2)
        B1plus = B1minus * np.exp(-2 * u1 * h[0])

        if I_temp == 0:
            BTplus = B1plus
            BTminus = B1minus
        elif I_temp == Nlayers - 1:
            BTplus = np.zeros(self.Nint, self.Nfreq)
            BTminus = BNminus
        elif I_temp < I_heat:
            _, Bplus, Bminus = self.__calculate_up_matrix(
                0, I_temp, Lambda, D, h, eta, kterm2, omega
            )            
            BTplus  = Bplus  * B1minus
            BTminus = Bminus * BNminus
        elif I_temp >= I_heat:            
            _, Bplus, Bminus = self.__calculate_down_matrix(
                Nlayers - 1, I_temp, Lambda, D, h, eta, kterm2, omega
            )
            BTplus  = Bplus  * BNminus
            BTminus = Bminus * BNminus

        q2 = np.outer(np.ones(self.Nint), 1j * omega / D[I_temp])
        un = np.sqrt(eta[I_temp] * kterm2 + q2)

        G = BTplus * np.exp(un * X_tempL) + BTminus * np.exp(-un * X_tempL)

        Nk, Nf = kvect.shape
        Nt = len(self.r_probe)

        arg1 = -np.pi ** 2 * (self.r_pump ** 2 + self.r_probe ** 2) / 2
        arg2 = kvect2
        
        expterm = np.exp(np.outer(arg2.ravel(), arg1))

        if Nt == 1:
            expterm = expterm.reshape(Nk, Nf)
            Kernal  = 2 * np.pi * self.P_pump * expterm * kvect
            Integrand = G * Kernal
        else:
            expterm = expterm.reshape(Nk, Nf, -1)
            Kernal = 2 * np.pi * self.P_pump * np.matmul(expterm, np.tile(kvect, (1, 1, Nt)))
            Integrand = np.tile(G, (1, 1, Nt)) * Kernal

        return Integrand

    def __lgwt(self, n, a, b):
        xu, w = np.polynomial.legendre.leggauss(n)
        x = 0.5 * (b - a) * xu + 0.5 * (a + b)
        w = 0.5 * w * (b - a)
        return x, w

    def surface_temperature(self, tdelay, f, freq_coef = 10, n_nodes = 35):
        # Spatial integration limits
        fmax = freq_coef / np.min(np.abs(tdelay))
        kmax = 1 / np.sqrt(np.min(self.r_pump) ** 2 + np.min(self.r_probe) ** 2) * 2
        kvect, weights = self.__lgwt(n_nodes, 0, kmax)

        # Fourier maximum frequencies
        M = 20 * np.ceil(self.tau_rep / np.min(np.abs(tdelay)))
        mvect = np.arange(-M, M + 1)

        fudgep = np.exp(-np.pi * ((mvect / self.tau_rep + f) / fmax) ** 2)
        fudgem = np.exp(-np.pi * ((mvect / self.tau_rep - f) / fmax) ** 2)

        # Initialize T and Ratio
        T       = np.zeros((len(tdelay), len(self.X_heat)), dtype=complex)
        Ratio   = np.zeros((len(tdelay), len(self.X_heat)))

        for i in range(len(self.X_heat)):
            # Calculate integrands
            Ip = self.__calculate_integrand(
                kvect, mvect / self.tau_rep + f, self.X_heat[i]
            )

            Im = self.__calculate_integrand(
                kvect, mvect / self.tau_rep - f, self.X_heat[i]
            )
            
            expterm = np.exp(1j * 2 * np.pi / self.tau_rep * np.outer(tdelay, mvect))

            dTp = np.dot(weights.T, Ip)
            dTm = np.dot(weights.T, Im)

            NNt = np.ones((len(tdelay), 1))
            Retemp = (NNt * (dTp * fudgep + dTm * fudgem)) * expterm
            Imtemp = (NNt * (dTp - dTm)) * expterm * (-1j)

            Resum = np.sum(Retemp, axis=1)  # Sum over all Fourier series components
            Imsum = np.sum(Imtemp, axis=1)

            Tm = Resum + 1j * Imsum  #

            T[:, i] = Tm  # Reflectance Fluxation (Complex)
            Ratio[:, i] = -np.real(T[:, i]) / np.imag(T[:, i])

        return T, Ratio
    
    def plot_fitted_values(self, tdelay_model, Ratio_model, Tin_model, Tout_model,
                             t_norm = 100, path_to_file_noext = None):
        for i in range(len(self.data_raw)):
            if t_norm is None:
                fig, ax1 = plt.subplots(1,1)
            else:
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, sharex = True, figsize = (6.4, 7.2)
                )
            data = self.data_raw[i].extract_interior(self.tdelay_min, self.tdelay_max)

            ax1.semilogx(data.df.tdelay, data.df.Ratio, 'ro', markersize = 4)
            ax1.semilogx(tdelay_model * 1e12, Ratio_model[i], 'k', linewidth = 2)

            ax1.set_ylabel('Ratio -Vin/Vout (a.u.)')
            ax1.set_xlim([self.tdelay_min * 1e12, self.tdelay_max * 1e12])
            ax1.grid(which = 'major', linestyle = '-')
            ax1.grid(which = 'minor', linestyle = '--')
            ax1.set_xlabel('time delay (ps)')

            if t_norm is not None:
                # Normalization of in-phase / out-of-phase temperature changes
                Vin_norm = np.interp(t_norm, data.df.tdelay * 1e12, data.df.Vin)
                Tin_model_norm = np.interp(t_norm, tdelay_model * 1e12, Tin_model[i])
                Tin_data = data.df.Vin / Vin_norm * Tin_model_norm
                Vout_norm = np.interp(t_norm, data.df.tdelay * 1e12, data.df.Vout)
                Tout_model_norm = np.interp(t_norm, tdelay_model * 1e12, Tout_model[i])
                Tout_data = data.df.Vout / Vout_norm * Tout_model_norm

                ax2.semilogx(data.df.tdelay, -Tout_data, 'bo', markersize=4)
                ax2.semilogx(data.df.tdelay, Tin_data, 'ro', markersize=4)
                ax2.grid(which = 'major', linestyle = '-')
                ax2.grid(which = 'minor', linestyle = '--')
                ax2.semilogx(tdelay_model * 1e12, Tin_model[i], 'k', linewidth=2)
                ax2.semilogx(tdelay_model * 1e12, -Tout_model[i], 'k', linewidth=2)

                ax2.set_xlabel('Time delay (ps)')
                ax2.set_ylabel('$\Delta$ T (K)')

            if path_to_file_noext is not None:
                fig.savefig(path_to_file_noext + '_FIT_%d.png' %i)
            else:
                fig.show()

    def plot(self):
        X0 = np.zeros(len(self.i_Lambda) + len(self.i_C) + len(self.i_h))

        for i in range(len(self.i_Lambda)):
            X0[i] = self.Lambda[self.i_Lambda[i]]
        
        for i in range(len(self.i_C)):
            X0[len(self.i_Lambda) + i] = self.C[self.i_C[i]]

        for i in range(len(self.i_h)):
            X0[len(self.i_Lambda) + len(self.i_C) + i] = self.h[self.i_h[i]]
        
        self.tdelay_model = np.logspace(
            np.log10(self.tdelay_min), np.log10(self.tdelay_max), 130
        )

        Tin_model   = []
        Tout_model  = []
        Ratio_model = []

        for i in range(len(self.data_raw)):
            Ts, _ = self.surface_temperature(self.tdelay_model, self.f[i])
            Ts    = np.dot(Ts, self.AbsProf) / np.sum(self.AbsProf)
            Tin_model.append(np.real(Ts))
            Tout_model.append(np.imag(Ts))
            Ratio_model.append(-np.real(Ts) / np.imag(Ts))

        self.plot_fitted_values(self.tdelay_model, Ratio_model, Tin_model, Tout_model, t_norm = None)
            
    def calculate_residuals(self, X):
        for i in range(len(self.i_Lambda)):
            self.Lambda[self.i_Lambda[i]] = X[i]

        for i in range(len(self.i_C)):
            self.C[self.i_C[i]] = X[len(self.i_Lambda) + i]

        for i in range(len(self.i_h)):
            self.h[self.i_h[i]] = X[len(self.i_Lambda) + len(self.i_C) + i]

        sum = 0
        num = 0

        for i in range(len(self.data_raw)):
            Ts, _       = self.surface_temperature(self.tdelay[i], self.f[i])
            Tin_model   = np.dot(np.real(Ts), self.AbsProf) / np.sum(self.AbsProf)
            Tout_model  = np.dot(np.imag(Ts), self.AbsProf) / np.sum(self.AbsProf)
            Ratio_model = -Tin_model / Tout_model

            res = ((Ratio_model - self.ratio[i]) / Ratio_model) ** 2
            sum = sum + np.sum(res)
            num = num + len(res)

        return np.sqrt(sum) / num
    
    def least_squares(self, bounds = None, output_filepath = None,
                      plot = False, verbose = False):
        X0 = np.zeros(len(self.i_Lambda) + len(self.i_C) + len(self.i_h))
        
        for i in range(len(self.i_Lambda)):
            X0[i] = self.Lambda[self.i_Lambda[i]]
        for i in range(len(self.i_C)):
            X0[len(self.i_Lambda) + i] = self.C[self.i_C[i]]
        for i in range(len(self.i_h)):
            X0[len(self.i_Lambda) + len(self.i_C) + i] = self.h[self.i_h[i]]
        
        if bounds is None:
            OptRes  = optimize.least_squares(self.calculate_residuals, X0)
        else:
            OptRes  = optimize.least_squares(self.calculate_residuals, X0, 
                                             bounds = bounds)
        Xsol    = OptRes.x

        self.tdelay_model = np.logspace(
            np.log10(self.tdelay_min), np.log10(self.tdelay_max), 130
        )

        Tin_model   = []
        Tout_model  = []
        Ratio_model = []
        
        for i in range(len(self.data_raw)):
            Ts, _ = self.surface_temperature(self.tdelay_model, self.f[i])
            Ts    = np.dot(Ts, self.AbsProf) / np.sum(self.AbsProf)
            Tin_model.append(np.real(Ts))
            Tout_model.append(np.imag(Ts))
            Ratio_model.append(-np.real(Ts) / np.imag(Ts))

        if verbose:
            print("Fitted thermal conductivities (W m^-1 K^-1): ")
            for i in range(len(self.i_Lambda)):
                n = self.i_Lambda[i]
                if self.Layer[n] == True:
                    print('%d. %s: %.2f' % (n + 1, self.Label[n], Xsol[i]))

            print("Calculated (lambda / h) thermal conductances (MW m^-2 K^-1): ")
            for i in range(len(self.i_Lambda)):
                n = self.i_Lambda[i]
                if self.Layer[n] == False:
                    Gs = Xsol[i] / self.h[n] / 1e6
                    print('%d. %s: %.2f' % (n + 1, self.Label[n], Gs))

        if plot:
            self.plot_fitted_values(self.tdelay_model, Ratio_model, Tin_model, Tout_model,
                                    t_norm = self.tdelay_min * 1e12, 
                                    path_to_file_noext = output_filepath)
        return OptRes

    def fit(self, method = 'Nelder-Mead', bounds = None, output_filepath = None, 
            plot = False, verbose = False, fatol = None, xatol = None, 
            return_all = False):
        # Initial guess
        X0 = np.zeros(len(self.i_Lambda) + len(self.i_C) + len(self.i_h))

        for i in range(len(self.i_Lambda)):
            X0[i] = self.Lambda[self.i_Lambda[i]]
        
        for i in range(len(self.i_C)):
            X0[len(self.i_Lambda) + i] = self.C[self.i_C[i]]

        for i in range(len(self.i_h)):
            X0[len(self.i_Lambda) + len(self.i_C) + i] = self.h[self.i_h[i]]
        
        OptRes  = optimize.minimize(self.calculate_residuals, X0, 
                                    method = method, bounds = bounds,
                                    options = {'fatol' : fatol, 'xatol' : xatol,
                                               'return_all' : return_all})
        Xsol    = OptRes.x

        self.tdelay_model = np.logspace(
            np.log10(self.tdelay_min), np.log10(self.tdelay_max), 130
        )

        Tin_model   = []
        Tout_model  = []
        Ratio_model = []
        
        for i in range(len(self.data_raw)):
            Ts, _ = self.surface_temperature(self.tdelay_model, self.f[i])
            Ts    = np.dot(Ts, self.AbsProf) / np.sum(self.AbsProf)
            Tin_model.append(np.real(Ts))
            Tout_model.append(np.imag(Ts))
            Ratio_model.append(-np.real(Ts) / np.imag(Ts))

        if verbose:
            print("Fitted thermal conductivities (W m^-1 K^-1): ")
            for i in range(len(self.i_Lambda)):
                n = self.i_Lambda[i]
                if self.Layer[n] == True:
                    print('%d. %s: %.2f' % (n + 1, self.Label[n], Xsol[i]))

            print("Calculated (lambda / h) thermal conductances (MW m^-2 K^-1): ")
            for i in range(len(self.i_Lambda)):
                n = self.i_Lambda[i]
                if self.Layer[n] == False:
                    Gs = Xsol[i] / self.h[n] / 1e6
                    print('%d. %s: %.2f' % (n + 1, self.Label[n], Gs))

        if plot:
            self.plot_fitted_values(self.tdelay_model, Ratio_model, Tin_model, Tout_model,
                                    t_norm = self.tdelay_min * 1e12, 
                                    path_to_file_noext = output_filepath)
            plt.close()
        return OptRes

    def calculate_sensitivity(self):
        Tin_0   = []
        Tout_0  = []
        Ratio_0 = []
        for n in range(len(self.data_raw)):
            T0, _ = self.surface_temperature(self.tdelay_model, self.f[n])
            T0    = np.dot(T0, self.AbsProf) / np.sum(self.AbsProf)
            Tin_0.append(np.real(T0))
            Tout_0.append(np.imag(T0))
            Ratio_0.append(-np.real(T0) / np.imag(T0))

        S_C    = np.zeros((len(self.tdelay_model), len(self.Lambda)))
        S_Cx   = S_C.copy()
        S_L    = S_C.copy()
        S_Lx   = S_C.copy()
        S_h    = S_C.copy()
        S_hx   = S_C.copy()
        
        Lambda  = self.Lambda.copy()
        C       = self.C.copy()
        h       = self.h.copy()
        X_temp  = self.X_temp
        r_pump  = self.r_pump
        r_probe = self.r_probe

        S  = []
        Sx = []
        for n in range(len(self.data_raw)):
            for i in range(len(Lambda)):
                ## Specific heat
                self.C[i] = C[i] * 1.01
                T_C, _    = self.surface_temperature(self.tdelay_model, self.f[n])
                T_C       = np.dot(T_C, self.AbsProf) / np.sum(self.AbsProf)
                Ratio_C   = -np.real(T_C) / np.imag(T_C)
                delta_C   = Ratio_C - Ratio_0[n]
                Num       = np.log(Ratio_C) - np.log(Ratio_0[n])
                Denom     = np.log(self.C[i]) - np.log(C[i])
                S_C[:,i]  = Num / Denom
                NumX      = np.log(np.real(T_C)) - np.log(Tin_0[n])
                DenomX    = np.log(self.C[i]) - np.log(C[i])
                S_Cx[:,i] = NumX / DenomX
                self.C[i] = C[i]

                ## Thermal conductivity
                self.Lambda[i] = Lambda[i] * 1.01
                T_L, _    = self.surface_temperature(self.tdelay_model, self.f[n])
                T_L       = np.dot(T_L, self.AbsProf) / np.sum(self.AbsProf)
                Ratio_L   = -np.real(T_L) / np.imag(T_L)
                delta_L   = Ratio_L - Ratio_0[n]
                Num       = np.log(Ratio_L) - np.log(Ratio_0[n])
                Denom     = np.log(self.Lambda[i]) - np.log(Lambda[i])
                S_L[:,i]  = Num / Denom
                NumX      = np.log(np.real(T_L)) - np.log(Tin_0[n])
                DenomX    = np.log(self.Lambda[i]) - np.log(Lambda[i])
                S_Lx[:,i] = NumX / DenomX
                self.Lambda[i] = Lambda[i]

                ## Layer thickness
                self.h[i] = h[i] * 1.01
                hsum    = np.cumsum(self.h)
                I_temp  = np.sum(np.ceil(hsum) / self.X_temp == 1)
                if i < I_temp:
                    self.X_temp = X_temp + self.h[i] - h[i]
                else:
                    self.X_temp = X_temp
                T_h, _    = self.surface_temperature(self.tdelay_model, self.f[n])
                T_h       = np.dot(T_h, self.AbsProf) / np.sum(self.AbsProf)
                Ratio_h   = -np.real(T_h) / np.imag(T_h)
                delta_h   = Ratio_h - Ratio_0[n]
                Num       = np.log(Ratio_h) - np.log(Ratio_0[n])
                Denom     = np.log(self.h[i]) - np.log(h[i])
                S_h[:,i]  = Num / Denom
                NumX      = np.log(np.real(T_h)) - np.log(Tin_0[n])
                DenomX    = np.log(self.h[i]) - np.log(h[i])
                S_hx[:,i] = NumX / DenomX
                self.h[i] = h[i]
            
            self.r_pump  = r_pump  * 1.01
            self.r_probe = r_probe * 1.01
            
            T_r, _    = self.surface_temperature(self.tdelay_model, self.f[n])
            T_r       = np.dot(T_r, self.AbsProf) / np.sum(self.AbsProf)
            Ratio_r   = - np.real(T_r) / np.imag(T_r)
            delta_r   = Ratio_r - Ratio_0[n]
            Num       = np.log(Ratio_r) - np.log(Ratio_0[n])
            Denom     = np.log(self.r_pump) - np.log(r_pump)
            S_r       = Num / Denom
            NumX      = np.log(np.real(T_r)) - np.log(Tin_0[n])
            DenomX    = np.log(self.r_pump) - np.log(r_pump)
            S_rx      = NumX / DenomX

            self.r_pump  = r_pump
            self.r_probe = r_probe
            
            S.append((S_C, S_L, S_h, S_r))
            Sx.append((S_Cx, S_Lx, S_hx, S_rx))
        
        return (S, Sx)
    
    def plot_sensitivity(self, S, i_C = None, i_Lambda = None, i_h = None, path_to_file_noext = None,
                         legend_loc = 'best'):
        for i in range(len(self.data_raw)):
            fig = plt.figure()
            ax  = fig.add_subplot(111)
            
            colors = ['red', 'blue', 'green', 'magenta', 'black', 'yellow']

            for n in range(len(self.Lambda)):
                if i_C is None or n in i_C:
                    ax.plot(self.tdelay_model * 1e12, S[i][0][:,n], 
                            linestyle = '-', color = colors[n], 
                            linewidth = 1, label = '$C_{%s}$' % self.Label[n])
                
                if i_Lambda is None or n in i_Lambda:
                    if self.Layer[n]:
                        ax.plot(self.tdelay_model * 1e12, S[i][1][:,n],
                                linestyle = '--', color = colors[n],
                                linewidth = 1, label = '$\Lambda_{%s}$' % self.Label[n])
                    else:
                        ax.plot(self.tdelay_model * 1e12, S[i][1][:,n],
                                linestyle = '--', color = colors[n],
                                linewidth = 1, label = '$G_{%s}$' % self.Label[n])
                
                if i_h is None or n in i_h:
                    ax.plot(self.tdelay_model * 1e12, S[i][2][:,n],
                            linestyle = '-.', color = colors[n],
                            linewidth = 1, label = '$h_{%s}$' % self.Label[n])
                
            ax.plot(self.tdelay_model * 1e12, S[i][3], label = '$r_{pump}$')
            ax.set_xlim([self.tdelay_min * 1e12, self.tdelay_max * 1e12])
            ax.set_xscale('log')
            ax.set_ylabel('Sensitivity (a.u.)')
            ax.grid(which='major', linestyle='-')
            ax.grid(which='minor', linestyle='--')
            ax.legend(loc = legend_loc)

            ax.set_xscale('log')
            ax.set_xlabel('t (ps)')
            
            if path_to_file_noext is not None:
                fig.savefig(path_to_file_noext + '_SENS_%d.png' %i)
            else:
                fig.show()
