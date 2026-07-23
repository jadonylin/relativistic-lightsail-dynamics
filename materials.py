"""
A module to store mechanical and thermal properties of preset materials.
All quantities are in SI units.

Young = Young's modulus, Pa
density = density, kg/m^3
Poisson = Poisson's ratio, dimensionless
thermal_expansion = thermal expansion coefficient, /K
thermal_conductivity = thermal conductivity, W/(m*K)
specific_heat = specific heat capacity per unit mass, J/(kg*K)
thermorefract = change in real refractive index with temperature, /K
absorption = absorption coefficient at specified wavelength, /m
thermoextinct = change in imaginary refractive index with temperature 
                at specified wavelength and temperature, /K
"""

Si3N4 = {"Young": 270e9,  # [Gad-el-Hak 2006, MEMS: Design and Fabrication, pg 3-172]
         "density": 3.1e3,  # [Steinlechner 2017, DOI: 10.1103/PhysRevD.96.022007]
         "Poisson": 0.27,  # [Gad-el-Hak 2006, MEMS: Design and Fabrication, pg 3-172]
         "thermal_expansion": 2.6e-6,  # [Steinlechner 2017, DOI: 10.1103/PhysRevD.96.022007]
         "thermal_conductivity": 4.9,  # [Steinlechner 2017, DOI: 10.1103/PhysRevD.96.022007]
         "specific_heat": 523,  # [Steinlechner 2017, DOI: 10.1103/PhysRevD.96.022007]
         "absorption": 3e-2,  # at 1550 nm, [Ji 2017, DOI: https://doi.org/10.1364/OPTICA.4.000619]
         "thermorefract": 4e-5,  # [Steinlechner 2017, DOI: 10.1103/PhysRevD.96.022007]
         "thermoextinct": 1.e-6  # calculated from data in [Fletcher 2018, DOI: 10.3389/fmats.2018.00001]
         }

SiO2_fused = {"Young": 73e9,  # [Corning 2015 Information Sheet, https://www.corning.com/media/worldwide/csm/documents/HPFS_Product_Brochure_All_Grades_2015_07_21.pdf]
              "density": 2.2e3,  # [Corning 2015 Information Sheet, https://www.corning.com/media/worldwide/csm/documents/HPFS_Product_Brochure_All_Grades_2015_07_21.pdf]
              "Poisson": 0.16,  # [Corning 2015 Information Sheet, https://www.corning.com/media/worldwide/csm/documents/HPFS_Product_Brochure_All_Grades_2015_07_21.pdf]
              "thermal_expansion": 0.57e-6,  # [Corning 2015 Information Sheet, https://www.corning.com/media/worldwide/csm/documents/HPFS_Product_Brochure_All_Grades_2015_07_21.pdf]
              "thermal_conductivity": 1.38,  # [Corning 2015 Information Sheet, https://www.corning.com/media/worldwide/csm/documents/HPFS_Product_Brochure_All_Grades_2015_07_21.pdf]
              "specific_heat": 770,  # [Corning 2015 Information Sheet, https://www.corning.com/media/worldwide/csm/documents/HPFS_Product_Brochure_All_Grades_2015_07_21.pdf]
              "absorption": 0.25e-4,  # at 1064 nm, [Hild 2006, DOI: https://doi.org/10.1364/AO.45.007269]
              "thermorefract": 8.6e-6,  # [Leviton 2008, DOI: https://doi.org/10.48550/arXiv.0805.0091]
              "thermoextinct": 0e-6  # calculated from data in []
              }