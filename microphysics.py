import logging
import numpy as np
import numba


m_to_um = 1.0e6
um_to_m = 1.0e-6


def log_normal(mu_g, sigma_g, x_min, x_max, N, normalize=False):
    """
    f(x) = 1 / (x log(sigma_g) sqrt(2 pi))
           exp( - (log(x) - log(mu_g))^2 / (2 log^2(sigma_g)) )
    """
    x = np.linspace(x_min, x_max, N)
    # logging.debug(x)
    f = np.exp( - (np.log(x) - np.log(mu_g))**2 / (2 * (np.log(sigma_g))**2) ) \
      / (x * np.log(sigma_g) * np.sqrt(2 * np.pi))
    if normalize:
        f /= f.max()
    # logging.debug(f)
    return x, f


def saturation_vapor_pressure(T):
    """
    Feynman Volume I equation 45.15, Wallace and Hobbs problem 2.25

    e = const exp(- L / (R T))
    e_0 = const exp(- L / (R T_0))
    e = e_0 exp(- (L / R) (1 / T - 1 / T_0))
    """
    e_0 = 6.112  # mb
    T_0 = 273.15 # K
    L   = 2.5e6  # J kg-1
    R_v = 461    # J K-1 kg-1
    return e_0 * np.exp(- (L / R_v) * (1 / T - 1 / T_0))


def saturation_vapor_pressure_empirical(T):
    return 6.112 * np.exp(17.67 * (T - 273.15) / (T - 29.65))


def kohler(T, B, r_d, r_w):
    """
    A = 2 M_w sigma / (R T rho_w)
    M_w molecular weight water
    sigma surface tension water
    R ideal gas constant
    log(RH) = A / r_w - B r_d^3 / (r_w^3 - r_d^3)
    """
    M_w = 18.016  # kg kmol-1
    rho_w = 1.0e3 # kg m-3
    sigma = 0.076 # J m-2
    R = 8.3143e3  # J K-1 kmol-1

    # kg kmol-1 J m-2 / (J kmol-1 kg m-3)
    A = 2 * M_w * sigma / (R * rho_w * T) 
    print('A', A)
    logging.debug('%.2e m' % A)

    log_RH = A / r_w - B * r_d**3 / (r_w**3 - r_d**3) 
    return log_RH


def mix_hygroscopicity(species_info, q, RH):
    logging.info('mix_hygroscopicity')
    logging.info(len(RH))
    B_mixed = np.zeros(len(RH), dtype=np.float32)
    N_mixed = np.zeros(len(RH), dtype=np.float32)

    for species in q:
        logging.info('mix_hygroscopicity:' + species)
        rho = species_info[species]['density']
        B_coeffs = species_info[species]['hygroscopicity']
        logging.debug(B_coeffs)
        B = B_coeffs[0] + B_coeffs[1] * RH + B_coeffs[2] * RH**2
        B_mixed += (q[species] / rho) * B
        N_mixed += q[species] / rho

    B_mixed /= N_mixed

    return B_mixed


def wet_radius(r_d, B, RH):
    """
    A ~ 0
    x = r_w / r_d
    log(RH) / B = - 1 / (x^3 - 1) 
    x^3 - 1 = - B / log(RH)
    x = [1 - B / log(RH)]^(1/3)
    """
    logging.info('dry_radius:%.2f um' % (m_to_um * r_d))
    log_RH = np.log(np.clip(RH, 0, 0.99))
    r_w = np.zeros(len(RH), dtype=np.float32)
    # r_w = r_d * (1 - B / log_RH)**(1.0/3.0)
    r_w = r_d * (1 - B / log_RH)**(1.0/3.0)
    return r_w


def water_fraction(species_info, q, r_d, r_w):
    """
    volume mixing ratio q_j / rho_j for species j
    vmr_dry = (dry volume mixing ratio)
        = sum_j (all species except water) q_j / rho_j
    vmr_wet = vmr_dry * (r_w / r_d)^3
    vmr_wat = q_w / rho_w = vmr_wet - vmr_dry
    [q] = kg_p / kg_dry
    [rho] = g_p / cm_p^3
    [q / rho] = 1000 cm_p^3 / kg_dry = cm_p^3 / g_dry
    """
    vmr_dry = np.zeros(len(r_w), dtype=np.float32)
    for species in q:
        logging.info('water_fraction:' + species)
        rho = species_info[species]['density']
        vmr_dry += q[species] / rho
    vmr_wet = vmr_dry * (r_w / r_d)**3
    vmr_wat = vmr_wet - vmr_dry
    return vmr_dry, vmr_wat


def mix_indices(nx, species_info, refractive_re, refractive_im, q, vmr_wat):
    n_re = np.zeros(nx, dtype=np.float32)
    n_im = np.zeros(nx, dtype=np.float32)
    vmr_dry = np.zeros(nx, dtype=np.float32)
    for species in q:
        logging.info('mix_indices:' + species)
        rho = species_info[species]['density']
        vmr_dry += q[species] / rho
        logging.info('indices:%.2f,%.2f' % (refractive_re[species], refractive_im[species]))
        n_re += float(refractive_re[species]) * q[species] / rho
        n_im += float(refractive_im[species]) * q[species] / rho
    logging.info('mix_indices:WAT')
    n_re += float(refractive_re['WAT']) * vmr_wat
    n_im += float(refractive_im['WAT']) * vmr_wat
    vmr_wet = vmr_dry + vmr_wat
    n_re /= vmr_wet
    n_im /= vmr_wet
    return n_re, n_im


def ext_lookup(nx, n_re_field, n_im_field, r_w_field, ds_table):
    n_re_table = ds_table['n_real'].values
    n_im_table = ds_table['n_imag'].values
    r_w_table = ds_table['radius'].values
    ext_table = ds_table['ext'].values
    abs_table = ds_table['abs'].values
    asm_table = ds_table['asm'].values
    del_n_re = n_re_table[1] - n_re_table[0]
    del_n_im = n_im_table[1] - n_im_table[0]
    del_r_w = r_w_table[1] - r_w_table[0]
    n_re_field = np.clip(n_re_field, n_re_table.min(), n_re_table.max())
    n_im_field *= -1
    n_im_field = np.clip(n_im_field, n_im_table.min(), n_im_table.max())
    # print('n_re_table', n_re_table.min(), n_re_table.max())
    # print('n_re', n_re_field.min(), n_re_field.max())
    # print('n_im_table', n_im_table.min(), n_im_table.max()) 
    # print('n_im', n_im_field.min(), n_im_field.max()) 
    r_w_field = np.clip(r_w_field, r_w_table.min(), r_w_table.max())
    i_re_field = np.array(np.floor((n_re_field - n_re_table[0]) / del_n_re), dtype=np.int32)
    i_im_field = np.array(np.floor((n_im_field - n_im_table[0]) / del_n_im), dtype=np.int32)
    i_r_w_field = np.array(np.floor((r_w_field - r_w_table[0]) / del_r_w), dtype=np.int32)
    # print('n_re', len(n_re_table))
    # print('i_re_field', i_re_field.mean())
    # print('r_w', r_w_field[::1000])
    # print('i_r_w', i_r_w_field[::1000])
    print('n_r_w', len(r_w_table))
    print('r_w_table', r_w_table.min(), r_w_table.max())
    print('r_w', r_w_field.min(), r_w_field.max(), r_w_field.mean())
    print('i_r_w_field', i_r_w_field.mean())
    ext_field = ext_table[i_re_field, i_im_field, i_r_w_field]
    abs_field = abs_table[i_re_field, i_im_field, i_r_w_field]
    asm_field = asm_table[i_re_field, i_im_field, i_r_w_field]
    return ext_field, abs_field, asm_field


def layer_optical_depth(nx, pdel, num_mix_ratio, ext_field):
    """
    d(tau) = k rho_w  V_w      dp / g
             L-1      L^3 M-1  M L-2
    1 g cm-3 = 10^6 g m-3 = 1000 kg m-3
    ---
    d(tau) = sigma  n    dz
             L^2    L-3  L
             sigma  n_m  dp / g
             L^2    M-1  M L-2
    """
    tau = ext_field * num_mix_ratio * pdel * (um_to_m**2 / 9.8)
    return tau

