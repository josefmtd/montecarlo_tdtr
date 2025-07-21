import click
import logging

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data import dataframe
from src.analysis import bidirectional_gpu as bidirectional

import monaco as mc
from scipy.stats import norm
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.figsize': (12.8, 6.4)})

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('pump_radius', type = float)
@click.argument('probe_radius', type= float)
@click.argument('frequency', type = float)
@click.argument('film_thickness', type = float)
@click.option('-k', '--initial_guess', 'initial_guess',
              type = int, default = 20)
@click.option('-t', '--thickness', 'thickness',
              type = int, default = 80)
@click.option('-N', '--num_iter', 'num_iter',
              type = int, default = 256)
@click.option('-P', '--prefix', 'prefix',
              type = str, default = 'aln-film')
def main(input_filepath, pump_radius, probe_radius, 
         frequency, film_thickness, initial_guess, thickness, 
         num_iter, prefix):
    logger = logging.getLogger(__name__)
    logger.info('Fitting values for AlN thin film')

    data = dataframe.Measurement(input_filepath)
    data_shift = data.auto_phase_shift()    
    logger.info('Data shifted by %.1f degrees' % data.phase_sol)
    logger.info('Phase uncertainties: %.2f degrees' % data.del_phase)
    
    # r_pump, r_probe = beam.characterize_beam(
    #     red_beam = redbeam_filepath,
    #     blue_beam = bluebeam_filepath,
    #     plot = False, use_center = False
    # )
    r_pump  = pump_radius
    r_probe = probe_radius
    
    SNR = data_shift.out_phase_statistics(plot = False)
    logger.info('SNR: %.1f dB' % SNR)
    print(prefix)

    #try:
    #    h = data_shift.picosecond_acoustics(
    #    h = thickness * 1e-9, use_ratio = True) * 1e9
    #except:
    #    h = thickness

    h = 80

    logger.info('Thickness: %.1f nm' % h)

    fcns = {
        'run'           : aluminium_nitride_run,
        'preprocess'    : aluminium_nitride_preprocess,
        'postprocess'   : aluminium_nitride_postprocess
    }
    
    sim = mc.Sim(name = prefix, ndraws = num_iter, fcns = fcns,
                 singlethreaded = False, verbose = False,
                 firstcaseismedian = True, resultsdir = './montecarlo-cases/%s/' % prefix,
                 savesimdata = True, savecasedata = True)
    
    sim.addInVar(name = 'h_Al', dist = norm, distkwargs = {'loc' : h, 'scale' : 1})
    sim.addInVar(name = 'phase', dist = norm, distkwargs = {'loc' : data.phase_sol, 'scale' : data.del_phase/2})
    sim.addInVar(name = 'C_Al', dist = norm, distkwargs = {'loc' : 2.42, 'scale' : 0.024})
    sim.addInVar(name = 'C_AlN', dist = norm, distkwargs = {'loc' : 2.7, 'scale' : 0.027})
    sim.addInVar(name = 'C_Si', dist = norm, distkwargs = {'loc' : 1.6, 'scale' : 0.016})
    sim.addInVar(name = 'r_pump', dist = norm, distkwargs = {'loc' : r_pump, 'scale' : r_pump/10})
    sim.addConstVal(name = 'r_probe', val = r_probe)
    sim.addConstVal(name = 'h_AlN', val = film_thickness)
    sim.addConstVal(name = 'frequency', val = frequency)
    sim.addConstVal(name = 'data', val = data)
    sim.addConstVal(name = 'k_AlN', val = initial_guess)

    sim.runSim()

    sim.vars['k_AlN'].addVarStat('mean')
    sim.vars['k_AlN'].addVarStat('median')
    sim.vars['k_AlN'].addVarStat('percentile', {'p':[0.025, 0.975]})
    
    sim.vars['G_AlN_Si'].addVarStat('mean')
    sim.vars['G_AlN_Si'].addVarStat('median')
    sim.vars['G_AlN_Si'].addVarStat('percentile', {'p':[0.025, 0.975]})

    sim.vars['G_Al_AlN'].addVarStat('mean')
    sim.vars['G_Al_AlN'].addVarStat('median')
    sim.vars['G_Al_AlN'].addVarStat('percentile', {'p':[0.025, 0.975]})
        
    os.makedirs(os.path.join("out", prefix), exist_ok=True)

    fig1, ax1 = plt.subplots()
    mc.plot(sim.vars['k_AlN'], ax = ax1)
    plt.savefig('out/%s/%s_k.png' % (prefix, prefix))
    
    fig2, ax2 = plt.subplots()
    mc.plot(sim.vars['G_AlN_Si'], ax = ax2)
    plt.savefig('out/%s/%s_G.png' % (prefix, prefix))

    fig3, ax3 = plt.subplots()
    mc.plot(sim.vars['G_Al_AlN'], ax = ax3)

    fig, ax = sim.plot()
    plt.savefig('out/%s/%s_spread.png' % (prefix, prefix))
    
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

def aluminium_nitride_preprocess(case):
    h_Al        = case.invals['h_Al'].val
    phase       = case.invals['phase'].val
    C_Al        = case.invals['C_Al'].val
    C_Si        = case.invals['C_Si'].val
    C_AlN       = case.invals['C_AlN'].val
    r_pump      = case.invals['r_pump'].val
    h_AlN       = case.constvals['h_AlN']
    r_probe     = case.constvals['r_probe']
    frequency   = case.constvals['frequency']
    data        = case.constvals['data']
    k_AlN       = case.constvals['k_AlN']
    
    SystemParameters = bidirectional.SysParam(
        r_pump, r_probe, P_pump = 15e-3, P_probe = 5e-3
    )

    SampleParameters = bidirectional.DutParam(
        Lambda  = np.array([237, 0.15, k_AlN, 0.2, 140]),
        Label   = ['Al', 'Al/AlN', 'AlN', 'AlN/Si', 'Si'],
        Layer   = [True, False, True, False, True],
        C       = np.array([C_Al, 0.1, C_AlN, 0.1, C_Si]) * 1e6,
        h       = np.array([h_Al, 1, h_AlN, 1, 1e6]) * 1e-9,
    )

    data_shift = data.phase_shift(phase)
    Bidirectional = bidirectional.Bidirectional(SystemParameters)
    Bidirectional.set_parameters(
        i_Lambda    = np.array([1, 2, 3]),
        i_C         = np.array([]),
        i_h         = np.array([]),
        tdelay_min  = 60e-12,
        tdelay_max  = 3500e-12
    )
    Bidirectional.set_sample_parameters(SampleParameters)
    Bidirectional.add_measurement_data(frequency, data_shift)
    return Bidirectional

def aluminium_nitride_run(Bidirectional):
    OptRes = Bidirectional.fit(
        method = 'Nelder-Mead', verbose = False, plot = False, fatol = 0.05, xatol = 0.5
    )
    
    G_Al_AlN = OptRes.x[0] / 1e-9
    k_AlN    = OptRes.x[1]
    G_AlN_Si = OptRes.x[2] / 1e-9
    RMSE     = OptRes.fun
    
    print(f'{k_AlN=:.1f}, {G_Al_AlN=:.4e}, {G_AlN_Si=:.4e}, {RMSE=:.4f}, {OptRes.nit=}')    
    return (G_Al_AlN, k_AlN, G_AlN_Si, RMSE)

def aluminium_nitride_postprocess(case, G_Al_AlN, k_AlN, G_AlN_Si, RMSE):
    case.addOutVal(name = 'G_Al_AlN', val = G_Al_AlN)
    case.addOutVal(name = 'k_AlN', val = k_AlN)
    case.addOutVal(name = 'G_AlN_Si', val = G_AlN_Si)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()