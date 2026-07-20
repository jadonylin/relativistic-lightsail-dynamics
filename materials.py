"""
A module to store mechanical and thermal properties of preset materials.
All quantities are in SI units.
"""

Si3N4 = {"Young": 270e9,  # Pa
         "density": 3.17e3,  # kg/m^3
         "Poisson": 0.27,  # dimensionless
         "thermal_expansion": 1.6e-6,  # /K
         "thermal_conductivity": 30,  # W/(m*K)
         "specific_heat": 900,  # J/(kg*K)
         "thermooptic": 2.45e-5  # /K
         }

SiO2_fused = {"Young": 73e9,  # Pa
              "density": 2.2e3,  # kg/m^3
              "Poisson": 0.17,  # dimensionless
              "thermal_expansion": 0.55e-6,  # /K
              "thermal_conductivity": 1.38,  # W/(m*K)
              "specific_heat": 740,  # J/(kg*K)
              "thermooptic": 8e-6  # /K
              }