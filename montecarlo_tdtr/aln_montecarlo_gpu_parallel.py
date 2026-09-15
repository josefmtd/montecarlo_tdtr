import click
import logging

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from montecarlo_tdtr.data import dataframe
from montecarlo_tdtr.analysis import bidirectional_gpu as bidirectional

import monaco as mc
from scipy.stats import norm
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless: pool workers have no display
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.figsize': (12.8, 6.4)})


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('pump_radius', type=float)
@click.argument('probe_radius', type=float)
@click.argument('frequency', type=float)
@click.argument('film_thickness', type=float)
@click.option('-k', '--initial_guess', 'initial_guess', type=float, default=100.0)
@click.option('-t', '--thickness', 'thickness', type=int, default=80)
@click.option('-N', '--num_iter', 'num_iter', type=int, default=256)
@click.option('-P', '--prefix', 'prefix', type=str, default='aln-film')
@click.option('-w', '--n_workers', 'n_workers', type=int,
              default=int(os.environ.get('MC_N_WORKERS', 8)),
              help='Number of pool worker processes oversubscribing the GPU.')
@click.option('--debug/--no-debug', 'debug', default=True,
              help='debug=True raises on failed cases instead of skipping.')
def main(input_filepath, pump_radius, probe_radius,
         frequency, film_thickness, initial_guess, thickness,
         num_iter, prefix, n_workers, debug):
    logger = logging.getLogger(__name__)
    logger.info('Fitting values for AlN thin film')

    data = dataframe.Measurement(input_filepath)
    data_shift = data.auto_phase_shift()

    phase_sol = data.phase_sol
    del_phase = data.del_phase

    logger.info('Data shifted by %.1f degrees' % phase_sol)
    logger.info('Phase uncertainties: %.2f degrees' % del_phase)

    r_pump = pump_radius
    r_probe = probe_radius

    SNR = data_shift.out_phase_statistics(plot=False)
    logger.info('SNR: %.1f dB' % SNR)

    try:
        h = data_shift.picosecond_acoustics(
            h=thickness * 1e-9, use_ratio=True) * 1e9
    except Exception:
        h = thickness

    h = 80

    logger.info('Thickness: %.1f nm' % h)

    # Initializes CUDA in the PARENT process. Safe because we request
    # multiprocessing_method='spawn' from monaco: workers are fresh
    # interpreters, not forked copies of this CUDA-initialized parent.
    test_passed = run_diagnostic_test(
        data=data_shift, h=h, r_pump=r_pump, r_probe=r_probe,
        film_thickness=film_thickness, frequency=frequency,
        initial_guess=initial_guess
    )
    if not test_passed:
        logger.error("Diagnostic test failed. Exiting script before starting Monaco.")
        return

    fcns = {
        'run':         aluminium_nitride_run,
        'preprocess':  aluminium_nitride_preprocess,
        'postprocess': aluminium_nitride_postprocess
    }

    sim = mc.Sim(
        name=prefix, ndraws=num_iter, fcns=fcns,
        singlethreaded=(n_workers == 1),
        usedask=False,
        ncores=n_workers,                  # monaco default None -> os.cpu_count()
        multiprocessing_method='spawn',    # fork would inherit a live CUDA parent
        verbose=True, firstcaseismedian=True, debug=debug,
        resultsdir='../out/mc-cases/%s/' % prefix,
        savesimdata=True, savecasedata=True)

    sim.addInVar(name='h_Al', dist=norm, distkwargs={'loc': h, 'scale': 1})
    sim.addInVar(name='phase', dist=norm,
                 distkwargs={'loc': phase_sol, 'scale': del_phase / 2})
    sim.addInVar(name='C_Al', dist=norm, distkwargs={'loc': 2.42, 'scale': 0.024})
    sim.addInVar(name='C_AlN', dist=norm, distkwargs={'loc': 1.938, 'scale': 0.019})
    sim.addInVar(name='C_Si', dist=norm, distkwargs={'loc': 1.6, 'scale': 0.016})
    sim.addInVar(name='r_pump', dist=norm,
                 distkwargs={'loc': r_pump, 'scale': r_pump / 10})
    sim.addConstVal(name='r_probe', val=r_probe)
    sim.addConstVal(name='h_AlN', val=film_thickness)
    sim.addConstVal(name='frequency', val=frequency)
    # With spawn, constvals are PICKLED to each worker: 'data' must be
    # picklable (no open file handles / live CuPy arrays inside Measurement).
    sim.addConstVal(name='data', val=data)
    sim.addConstVal(name='k_AlN_init', val=initial_guess)

    sim.runSim()

    sim.vars['k_AlN'].addVarStat('mean')
    sim.vars['k_AlN'].addVarStat('median')
    sim.vars['k_AlN'].addVarStat('percentile', {'p': [0.025, 0.975]})

    sim.vars['G_AlN_Si'].addVarStat('mean')
    sim.vars['G_AlN_Si'].addVarStat('median')
    sim.vars['G_AlN_Si'].addVarStat('percentile', {'p': [0.025, 0.975]})

    sim.vars['G_Al_AlN'].addVarStat('mean')
    sim.vars['G_Al_AlN'].addVarStat('median')
    sim.vars['G_Al_AlN'].addVarStat('percentile', {'p': [0.025, 0.975]})

    os.makedirs(os.path.join("../out", prefix), exist_ok=True)

    fig1, ax1 = plt.subplots()
    mc.plot(sim.vars['k_AlN'], ax=ax1)
    plt.savefig('../out/%s/%s_k.png' % (prefix, prefix))

    fig2, ax2 = plt.subplots()
    mc.plot(sim.vars['G_AlN_Si'], ax=ax2)
    plt.savefig('../out/%s/%s_G.png' % (prefix, prefix))

    fig3, ax3 = plt.subplots()
    mc.plot(sim.vars['G_Al_AlN'], ax=ax3)
    plt.savefig('../out/%s/%s_G_Al.png' % (prefix, prefix))

    fig, ax = sim.plot()
    plt.savefig('../out/%s/%s_spread.png' % (prefix, prefix))

    logger.info(sim.outvars['k_AlN'].varstats[0].vals)
    logger.info(sim.outvars['k_AlN'].varstats[1].vals)
    logger.info(sim.outvars['k_AlN'].varstats[2].vals)
    var_k_AlN = sim.outvars['k_AlN'].stats().variance
    logger.info('Standard deviation (k_AlN): %.2f' % np.sqrt(var_k_AlN))

    logger.info(sim.outvars['G_AlN_Si'].varstats[0].vals)
    logger.info(sim.outvars['G_AlN_Si'].varstats[1].vals)
    logger.info(sim.outvars['G_AlN_Si'].varstats[2].vals)
    var_G_AlN = sim.outvars['G_AlN_Si'].stats().variance
    logger.info('Standard deviation (G_AlN/Si): %.2f' % np.sqrt(var_G_AlN))

    logger.info(sim.outvars['G_Al_AlN'].varstats[0].vals)
    logger.info(sim.outvars['G_Al_AlN'].varstats[1].vals)
    logger.info(sim.outvars['G_Al_AlN'].varstats[2].vals)
    var_G_Al = sim.outvars['G_Al_AlN'].stats().variance
    logger.info('Standard deviation (G_Al_AlN): %.2f' % np.sqrt(var_G_Al))


def _pin_gpu(tag=''):
    """Select and activate a CUDA device for this process.

    Runs inside each spawned pool worker. Round-robins devices by PID
    and caps the CuPy pool so oversubscribed workers cannot OOM the GPU.
    """
    import cupy as cp
    try:
        ngpu = cp.cuda.runtime.getDeviceCount()
    except Exception:
        return None
    if ngpu == 0:
        return None
    dev = os.getpid() % ngpu
    cp.cuda.Device(dev).use()
    cp.get_default_memory_pool().set_limit(size=int(1.5 * 1024**3))  # 1.5 GiB/worker
    # print(f'{tag} GPU {dev} pinned (pid {os.getpid()}, {ngpu} devices)')
    return dev


def run_diagnostic_test(data, h, r_pump, r_probe, film_thickness,
                        frequency, initial_guess):
    """
    Manually runs a single preprocessing and fitting step to test the
    Bidirectional GPU module, bypassing Monaco's pool wrappers to
    expose the exact error in the parent process.
    """
    print("\n" + "=" * 50)
    print("      STARTING BIDIRECTIONAL DIAGNOSTIC TEST")
    print("=" * 50)

    class ValMock:
        def __init__(self, value):
            self.val = value
        def __float__(self):
            return float(self.val)
        def __repr__(self):
            return str(self.val)

    class MockCase:
        def __init__(self):
            self.invals = {
                'h_Al': ValMock(h),
                'phase': ValMock(0),
                'C_Al': ValMock(2.42),
                'C_Si': ValMock(1.6),
                'C_AlN': ValMock(1.938),
                'r_pump': ValMock(r_pump)
            }
            self.constvals = {
                'h_AlN': film_thickness,
                'r_probe': r_probe,
                'frequency': frequency,
                'data': data,
                'k_AlN_init': initial_guess
            }

    try:
        print("1. Constructing Mock Case...")
        case = MockCase()

        print("2. Calling `aluminium_nitride_preprocess`...")
        bidirectional_obj = aluminium_nitride_preprocess(case)
        print("   [SUCCESS] Preprocessing completed.")
        print(f"   [INFO] Created Object: {bidirectional_obj}")

        print("3. Checking System & Sample Parameters...")
        print(f"   [INFO] Pump Radius: {r_pump} m, Probe Radius: {r_probe} m")
        print(f"   [INFO] Sample Thickness Array: "
              f"{np.array([h, 1, film_thickness, 1, 1e6]) * 1e-9}")

        print("4. Calling `aluminium_nitride_run` (GPU Solver)...")
        results = aluminium_nitride_run(bidirectional_obj, quiet=False)
        print("   [SUCCESS] Solver execution completed.")
        print(f"   [RESULT] G_Al_AlN: {results[0]:.3e}")
        print(f"   [RESULT] k_AlN:    {results[1]:.3f}")
        print(f"   [RESULT] G_AlN_Si: {results[2]:.3e}")
        print(f"   [RESULT] RMSE:     {results[3]:.5f}")
        print("=" * 50)
        print("   DIAGNOSTIC TEST PASSED: Bidirectional GPU is working perfectly!")
        print("=" * 50 + "\n")
        return True

    except Exception:
        print("\n" + "!" * 50)
        print("   DIAGNOSTIC TEST FAILED!")
        print("!" * 50)
        import traceback
        traceback.print_exc()
        print("=" * 50 + "\n")
        return False


def aluminium_nitride_preprocess(case):
    _pin_gpu(tag='[worker]')  # runs in the spawned worker process

    h_Al    = case.invals['h_Al'].val
    phase   = case.invals['phase'].val
    C_Al    = case.invals['C_Al'].val
    C_Si    = case.invals['C_Si'].val
    C_AlN   = case.invals['C_AlN'].val
    r_pump  = case.invals['r_pump'].val
    h_AlN   = case.constvals['h_AlN']
    r_probe = case.constvals['r_probe']
    frequency = case.constvals['frequency']
    data    = case.constvals['data']
    k_AlN   = case.constvals['k_AlN_init']

    SystemParameters = bidirectional.SysParam(
        r_pump, r_probe, P_pump=15e-3, P_probe=5e-3
    )

    SampleParameters = bidirectional.DutParam(
        Lambda=np.array([237, 0.15, k_AlN, 0.15, 140]),
        Label=['Al', 'Al/AlN', 'AlN', 'AlN/Si', 'Si'],
        Layer=[True, False, True, False, True],
        C=np.array([C_Al, 0.1, C_AlN, 0.1, C_Si]) * 1e6,
        h=np.array([h_Al, 1, h_AlN, 1, 1e6]) * 1e-9,
    )

    data_shift = data.phase_shift(phase)

    Bidirectional = bidirectional.Bidirectional(SystemParameters)
    Bidirectional.set_parameters(
        i_Lambda=np.array([1, 2, 3]),
        i_C=np.array([]),
        i_h=np.array([]),
        tdelay_min=100e-12,
        tdelay_max=3500e-12
    )
    Bidirectional.set_sample_parameters(SampleParameters)
    Bidirectional.add_measurement_data(frequency, data_shift)
    return Bidirectional


def aluminium_nitride_run(Bidirectional, quiet=True):
    bounds = [
        (1e-3, 500e-3), # G_Al/AlN
        (0.1, 150),     # k_AlN for 185
        (1e-3, 500e-3), # G_AlN/Si
    ]
    
    OptRes = Bidirectional.fit(
        method='Nelder-Mead', verbose=False, plot=False,
        fatol=0.05, xatol=0.5, bounds=bounds
    )

    G_Al_AlN = OptRes.x[0] / 1e-9
    k_AlN    = OptRes.x[1]
    G_AlN_Si = OptRes.x[2] / 1e-9
    RMSE     = OptRes.fun    
    print(f'{k_AlN=:.1f}, {G_Al_AlN=:.4e}, {G_AlN_Si=:.4e}, {RMSE=:.4f}, {OptRes.nit=}')
    return (G_Al_AlN, k_AlN, G_AlN_Si, RMSE)


def aluminium_nitride_postprocess(case, G_Al_AlN, k_AlN, G_AlN_Si, RMSE):
    case.addOutVal(name='G_Al_AlN', val=G_Al_AlN)
    case.addOutVal(name='k_AlN', val=k_AlN)
    case.addOutVal(name='G_AlN_Si', val=G_AlN_Si)
    case.addOutVal(name='RMSE', val=RMSE)

if __name__ == '__main__':
    # Load-bearing: spawned workers re-import this module; the guard
    # stops them from re-firing the Click CLI and re-running main().
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())

    main()
