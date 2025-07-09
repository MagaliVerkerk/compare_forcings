# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:49:35 2025

@author: mv393
"""

from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
from fair.earth_params import seconds_per_year
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



# emissions_csv = './data/rcmip-emissions-annual-means-v5-1-0.csv'
# concentration_csv = './data/rcmip-concentrations-annual-means-v5-1-0.csv'
# forcing_csv = './data/rcmip-radiative-forcing-annual-means-v5-1-0.csv'
start_date = 1850
end_date = 2004
stochastic_behavior = False



f = FAIR(ch4_method='thornhill2021')

f.define_time(start_date,end_date, 1)

# Define SSP scenarios, only one kept, as we look into the past not the future
scenarios = ['rcp60']
f.define_scenarios(scenarios)


df_configs = pd.read_csv("./data/calibrated_constrained_parameters.csv",index_col=0,)

configs = []
for k in range(df_configs.shape[0]):
    name_ind = (df_configs.index[k])
    configs.append(f"{name_ind}")

f.define_configs(configs)


species, properties = read_properties('./data/species_configs_properties_CMIP5.csv')
f.define_species(species, properties)
f.allocate()

f.fill_species_configs('./data/species_configs_properties_CMIP5.csv')

df_emissions_init = pd.read_csv("./data/baseline_emissions_CMIP5_1850.csv",index_col=0,)
df_concentration_init = pd.read_csv("./data/baseline_concentration_CMIP5_1850.csv",index_col=0,)

for specie in species:
    if properties[specie]['input_mode'] == 'concentration':
        fill(f.species_configs['baseline_concentration'], df_concentration_init.loc[specie, str(start_date)], specie=specie)
    elif properties[specie]['input_mode'] == 'emissions':
        fill(f.species_configs['baseline_emissions'], df_emissions_init.loc[specie, str(start_date)], specie=specie)


initialise(f.concentration, f.species_configs['baseline_concentration'])
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)



for config in configs:
    condition = (df_configs.index == float(config))
    fill(f.climate_configs['ocean_heat_capacity'], df_configs.loc[condition, 'clim_c1':'clim_c3'].values.squeeze(), config=config)
    fill(f.climate_configs['ocean_heat_transfer'], df_configs.loc[condition, 'clim_kappa1':'clim_kappa3'].values.squeeze(), config=config)
    fill(f.climate_configs['deep_ocean_efficacy'], df_configs.loc[condition, 'clim_epsilon'].values[0], config=config)
    fill(f.climate_configs['gamma_autocorrelation'], df_configs.loc[condition, 'clim_gamma'].values[0], config=config)
    fill(f.climate_configs['sigma_eta'], df_configs.loc[condition, 'clim_sigma_eta'].values[0], config=config)
    fill(f.climate_configs['sigma_xi'], df_configs.loc[condition, 'clim_sigma_xi'].values[0], config=config)
    fill(f.climate_configs['stochastic_run'], True, config=config)
    fill(f.climate_configs['use_seed'], True, config=config)
    fill(f.climate_configs['seed'], df_configs.loc[condition, 'seed'].values[0], config=config)

f.fill_from_rcmip()


df_volcanic = pd.read_csv('./data/cmip5_forcings/volcanic_forcing_monthly_cmip5.csv', index_col=0)
# overwrite volcanic
# yearly values of volcanic forcing = mean of monthly values
volcanic_forcing = np.zeros((end_date-start_date)+1)

volcanic_forcing[:int(np.shape(df_volcanic)[0]/12+1)] = df_volcanic[start_date-1:].groupby(np.ceil(df_volcanic[start_date-1:].index) // 1).mean().squeeze().values

volcanic_forcing -= np.mean(volcanic_forcing)
fill(f.forcing, volcanic_forcing[:, None, None], specie="Volcanic")  # sometimes need to expand the array


df_solar = pd.read_csv('./data/cmip5_forcings/solar_forcing_cmip5.csv', index_col=0)
fill(f.forcing, df_solar['cmip5'].to_numpy()[:, None, None], specie='Solar')

df_landuse = pd.read_csv('./data/cmip5_forcings/land_use_forcing_LUH1.csv', index_col=0)
fill(f.forcing, df_landuse.loc[start_date:end_date,'LUH1-CMIP5'].to_numpy()[:, None, None], specie='Land use')

initialise(f.forcing, 0) 

f.run()




plt.figure()
plt.plot(f.timebounds, f.temperature.loc[dict(scenario='rcp60', layer=0)], color='0.5')
plt.plot(f.timebounds, np.mean(np.array(f.temperature.loc[dict(scenario='rcp60', layer=0)]), axis=1), color='k', lw=2.5)
plt.xlabel('time, years')
plt.ylabel('temperature')
plt.show()


ghg_forcing = np.zeros((155, 31))
aerosol_forcing = np.column_stack((np.array(f.forcing.loc[dict(scenario='rcp60', config='4520', specie='Aerosol-radiation interactions')]), 
                                   np.array(f.forcing.loc[dict(scenario='rcp60', config='4520', specie='Aerosol-cloud interactions')])))

other_anthro_forcing = np.column_stack((np.array(f.forcing.loc[dict(scenario='rcp60', config='4520', specie='Ozone')]), 
                                        np.array(f.forcing.loc[dict(scenario='rcp60', config='4520', specie='Land use')])))

natural_forcing = np.column_stack((np.array(f.forcing.loc[dict(scenario='rcp60', config='4520', specie='Solar')]), 
                                   np.array(f.forcing.loc[dict(scenario='rcp60', config='4520', specie='Volcanic')])))


plt.figure(layout='constrained')
label_col_ghg =[]
k=0
for specie in species:
    if properties[specie]['input_mode'] == 'concentration':
        plt.plot(f.timebounds, f.forcing.loc[dict(scenario='rcp60', config='4520', specie=specie)], color='navy')
        ghg_forcing[:,k] = f.forcing.loc[dict(scenario='rcp60', config='4520', specie=specie)]
        label_col_ghg.append(specie)
        k+=1
    elif properties[specie]['input_mode'] == 'forcing':
        plt.plot(f.timebounds, f.forcing.loc[dict(scenario='rcp60', config='4520', specie=specie)], color='mediumorchid')
    else:
        plt.plot(f.timebounds, f.forcing.loc[dict(scenario='rcp60', config='4520', specie=specie)], color='goldenrod')

plt.xlabel('time, years')
plt.ylabel('radiative forcing, W/m2')
plt.show()



save_ghg = pd.DataFrame(ghg_forcing, columns=label_col_ghg).to_csv('./fair_outputs/CMIP5_forcing_ghgs.csv', index=None)
save_aerosol = pd.DataFrame(aerosol_forcing, columns=['Aerosol-radiation interactions', 'Aerosol-cloud interactions']).to_csv('./fair_outputs/CMIP5_forcing_aerosols.csv', index=None)

save_other_anthro = pd.DataFrame(other_anthro_forcing, columns=['Ozone', 'Land Use']).to_csv('./fair_outputs/CMIP5_forcing_other_anthropogenic.csv', index=None)
save_natural = pd.DataFrame(natural_forcing, columns=['Solar', 'Volcanic']).to_csv('./fair_outputs/CMIP5_forcing_natural.csv', index=None)

plt.figure()
plt.plot(f.timebounds, f.forcing_sum.loc[dict(scenario='rcp60', config='4520')], color='k')
plt.show()




cmip5_time = f.timebounds
cmip5_temperature = np.mean(np.array(f.temperature.loc[dict(scenario='rcp60', layer=0)]), axis=1)
cmip5_forcings = f.forcing
cmip5_forcing_sum = f.forcing_sum.loc[dict(scenario='rcp60', config='4520')]




