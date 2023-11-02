from .algorithm import rk4_system_algorithm
from .rkacem import rkacem_system_method
# from .rkhem import rkhem_system_method
from .rk4 import rk4_system_method
from .rkhm import rkhm_system_method
from .rkhem import rkhem_system_method
from .rkcom import rkcom_system_method
from .rk55 import rk55_system_method

from .test_helper import test_ode_method, plot_approximate_vs_exact

ode_methods = {
    "rk4_system_method": rk4_system_method,
    "rkacem_system_method": rkacem_system_method,
    "rkhm_system_method": rkhm_system_method,
    "rkcom_system_method": rkcom_system_method,
    "rkhem_system_method": rkhem_system_method,
    "rk55_system_method": rk55_system_method
}
