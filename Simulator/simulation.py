import enum
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter
import progressbar

def normalise(img, maximum=None, minimum=None):
    img = np.asarray(img)
    if maximum == None:
        maximum = np.max(img)
    if minimum == None:
        minimum = np.min(img)
    return (img - minimum)/(maximum - minimum)

def get_current(tc=None, gamma_s=None, gamma_d=None, epsilon=None):
    gamma_s[gamma_s == 0] = 10**(-100)
    denominator = (tc**2*(2 + gamma_d / gamma_s) + 0.25*gamma_d**2 + epsilon**2)
    denominator[denominator == 0] = 10**(-100)
    current_ss = ( tc**2 * gamma_d) / denominator
    current_ss = np.nan_to_num(current_ss)
    return current_ss

def fermi(x=None, mu=None, temp=None):
    k = 8.617E-5
    beta = (1E-3) / (k * temp) #meV -> eV
    return 1 / (1 + np.exp( (x - mu)*beta) )

def get_charge_jump_noise():
    n_rolls=np.random.choice((0, 1, 2))
    rollaxis=np.random.choice((0, 1))
    rolllengths=np.random.uniform(-0.05, 0.05, n_rolls)
    rollpoints=np.random.randint(0, 99, n_rolls)
    return n_rolls, rollaxis, rolllengths, rollpoints

def standard_sample_factors():
    mu_s = np.random.uniform(-0.5, 0.5)
    bias = np.random.uniform(-2,-0.1)
    mu_d = mu_s + bias
    temp = np.random.uniform(0.1, 1)
    shift = np.random.uniform(0.01, 2)
    
    lever_arms = np.random.uniform(0.5, 1.5, 2)
    cross_talk = np.random.uniform(0.0, 0.5, 2)
    
    n_pots_1 = np.random.randint(2, 4)
    n_pots_2 = np.random.randint(2, 7)
    
    charging1 = np.random.uniform(bias - 0.2, bias + 0.2)
    epsilon = charging1 / 4
    charging11_deltas = np.array([0] + [*np.random.uniform(-epsilon, epsilon, n_pots_1 - 1)])
    
    charging2 = np.random.uniform(0.2, 0.5)
    epsilon = charging2 / 4
    charging21_deltas = np.array([0] + [*np.random.uniform(-epsilon, epsilon, n_pots_2 - 1)])
    
    dot1_pot_arrangement = charging11_deltas+[i*charging1 for i in range(n_pots_1)]
    
    dot2_pot_arrangement = charging21_deltas+[i*charging2 for i in range(n_pots_2)]
    
    gamma_s = np.random.uniform(0.01, 0.5, n_pots_1)
    gamma_d = np.random.uniform(0.01, 0.5, n_pots_2)
    
    jitter_var = np.random.uniform(0,0.05)
    
    in_psb = np.random.choice((True, False))
    
    tc = np.random.uniform(0.01, 0.4, [n_pots_1, n_pots_2]) 
    
    gaussian_blur = np.random.uniform(0.8, 1.2)
    white_noise_level = np.random.uniform(3E-2, 7E-2)
    
        
    return {'tc':tc,'gamma_s':gamma_s,'gamma_d':gamma_d,
            'mu_s':mu_s,'mu_d':mu_d,'temp':temp,'lever_arms':lever_arms, 
             'cross_talk':cross_talk, 'shift':shift,
            'dot1_pot_arrangement':dot1_pot_arrangement,'dot2_pot_arrangement':dot2_pot_arrangement,
            'in_psb':in_psb,'jitter_var':jitter_var,
           'gaussian_blur':gaussian_blur,'white_noise_level':white_noise_level}


def get_triangle(dot1_base=None, dot2_base=None,dot1_pot_arrangement=None, dot2_pot_arrangement=None,
                mu_s=None, mu_d=None, temp=None, gamma_s=None, gamma_d=None, tc=None,
                jitter_var=None, kT=None, temp_broadening=True, in_psb=None, **kwargs):
    dot1_pots = (dot1_base[:,:,np.newaxis]+dot1_pot_arrangement[np.newaxis,:])

    dot2_pots = (dot2_base[:,:,np.newaxis] + dot2_pot_arrangement[np.newaxis,:])

    #to jitter each potential level randomly
    jitter = 1+jitter_var*np.random.randn(*dot1_pots.shape)
    jitter2 = 1+jitter_var*np.random.randn(*dot2_pots.shape)

    #source/drain tunneling rates
    rate_s = fermi(x=dot1_pots, mu=mu_s, temp=temp) * gamma_s
    rate_d = (1 - fermi(x=dot2_pots, mu=mu_d, temp=temp)) * gamma_d



    
    rate_s=rate_s[...,np.newaxis] # shape: [first_voltage,second_voltage, firstdot, seconddot]
    rate_d=rate_d[:,:,np.newaxis] # shape: [first_voltage,second_voltage, firstdot, seconddot]
    
    #difference between levels
    epsilon = (jitter * dot1_pots)[..., np.newaxis] - (jitter2 * dot2_pots)[:, :, np.newaxis]
    if in_psb:
        epsilon[:, :, 0, 0] = np.zeros(epsilon[:, :, 0, 0].shape)
        
    current = get_current(tc=tc, gamma_s=rate_s, gamma_d=rate_d, epsilon=epsilon)
    
    #tunneling only allowed in downward direction
    current = np.heaviside(epsilon, 0)*current
    #summing up over all possible channels
    current = np.sum(current, axis=(2, 3))
    
    #computing masks for bias window and PSB
    if temp_broadening:
        mu_s_window = mu_s + kT
        mu_d_window = mu_d - kT
    else:
        mu_s_window = mu_s
        mu_d_window = mu_d

    initial_setting = np.heaviside(mu_s_window-dot1_pots[:, :, 0], 0)
    interdot_matrix = np.heaviside(
                          ((jitter[:, :, 0] * dot1_pots[:, :, 0]) 
                          - jitter2[:, :, 0] * dot2_pots[:, :, 0]), 
                          0
    )
    dot2_setting = interdot_matrix * initial_setting
    final_matrix = np.heaviside(dot2_pots[:, :, 0] - mu_d_window, 0)
    
    bias_window_mask = final_matrix * dot2_setting

    psb_mask = np.heaviside(
                          ((jitter[:, :, 0] * dot1_pots[:, :, 0]) 
                           - jitter2[:, :, 1] * dot2_pots[:, :, 1]), 
                           0
    )

    current = bias_window_mask * current
    if in_psb:
        current = psb_mask * current
    return current

def get_voltage_extent(kT=None,lever_arms=None, cross_talk=None,
                       shift=None,
                       mu_s=None, mu_d=None, 
                        temp_broadening=None,
                       blank_space=0, **kwargs):
    """
    compute where the bias triangles are in the voltage space to correctly crop them
    """
    if temp_broadening:
        source_pot = mu_s + kT
        drain_pot = mu_d - kT
    else:
        source_pot = mu_s
        drain_pot = mu_d
    bias = source_pot - drain_pot
    #compute extent
    y_extent = []
    x_extent=[]
    a = lever_arms[0]
    b = cross_talk[0]
    c = cross_talk[1]
    d = lever_arms[1]

    y_extent.append((source_pot*(1 - c / a))
                    /(d - (c * b / a)))
    x_extent.append((source_pot - b * y_extent[-1]) / a)

    y_extent.append(((drain_pot - shift) * (1 - c / a))
                    /(d - (c * b / a)))
    x_extent.append((drain_pot - shift - b * y_extent[-1]) / a)

    y_extent.append((drain_pot - source_pot * c / a)
                    /(d - (c * b / a)))
    x_extent.append((source_pot - b * y_extent[-1])
                    /a)

    y_extent.append((drain_pot - shift - (source_pot - shift) * c / a)
                    /(d - (c * b / a)))
    x_extent.append((source_pot - shift - b * y_extent[-1])
                    /a)

    y_extent = [np.min(y_extent), np.max(y_extent)]
    x_extent = [np.min(x_extent), np.max(x_extent)]
    y_dis = y_extent[1] - y_extent[0]
    x_dis = x_extent[1] - x_extent[0]

    sidelength = (1 + blank_space) * np.max((x_dis, y_dis))

    dot1_voltage_bounds = [np.mean(x_extent) - sidelength / 2, 
                           np.mean(x_extent) + sidelength / 2]
    
    dot2_voltage_bounds = [np.mean(y_extent) - sidelength / 2,
                           np.mean(y_extent) + sidelength / 2]
    return [dot1_voltage_bounds, dot2_voltage_bounds]

def get_base_pots(params, adjust_voltage_window=True, blank_space=0.5):
    lever_arms = params['lever_arms']
    cross_talk = params['cross_talk']
    temp = params['temp']
    kT = 8.617E-5 * temp * 1E3
    
    try:
        params['blank_space']
    except:
        params['blank_space'] = blank_space
    if adjust_voltage_window:
        voltage_bounds = get_voltage_extent(kT=kT,**params)
    else:
        voltage_bounds = params['voltage_bounds']
        
    n_points = params['n_points']
    
    

    dot1_voltages = np.linspace(voltage_bounds[0][0], voltage_bounds[0][1], n_points[0])
    dot2_voltages = np.linspace(voltage_bounds[1][0], voltage_bounds[1][1], n_points[1])


    xv, yv = np.meshgrid(dot1_voltages, dot2_voltages)

    dot1_base = lever_arms[0] * xv + cross_talk[0] * yv
    dot2_base = lever_arms[1] * yv + cross_talk[1] * xv
    return dot1_base, dot2_base
    

def get_bias_triangles(params, gaussian_blur=None,white_noise_level=None, adjust_voltage_window=True, max_current=None, blank_space=None):
    
    try:
        blank_space = params['blank_space']
    except:
        pass
    if blank_space is None:
        blank_space = 0.5
        
    
    n_points=params['n_points']
    kT = 8.617E-5*params['temp']*1E3
    
    dot1_base,dot2_base=get_base_pots(params, adjust_voltage_window=adjust_voltage_window, blank_space=blank_space)

    first_triangle=get_triangle(**params, dot1_base=dot1_base,dot2_base=dot2_base,kT=kT)
    
    dot1_base +=params['shift']
    dot2_base +=params['shift']

    second_triangle=get_triangle(**params, dot1_base=dot1_base,dot2_base=dot2_base,kT=kT)

    first_triangle=first_triangle.reshape(-1)
    second_triangle=second_triangle.reshape(-1)
    current=np.max((first_triangle,second_triangle),axis=0).reshape(n_points)
    
    if gaussian_blur==None:
        gaussian_blur=params['gaussian_blur']
    if white_noise_level==None:
        white_noise_level=params['white_noise_level']
           
    # Blur current
    current = gaussian_filter(current, gaussian_blur)
    # Add noise
    if max_current==None:
        max_current=np.max(current)
    current = current + np.abs(np.random.normal(0, max_current*white_noise_level,current.shape))
    
    n_rolls, rollaxis, rolllengths, rollpoints=get_charge_jump_noise()
    rolllengths_px=np.array(rolllengths*current.shape[rollaxis],dtype=int)
    
    
    for i in range(n_rolls):
        if rollaxis==0:
            current[:,:rollpoints[i]]=np.roll(current[:,:rollpoints[i]],rolllengths_px[i],rollaxis)
        else:
            current[:rollpoints[i]]=np.roll(current[:rollpoints[i]],rolllengths_px[i],rollaxis)


    return current


def get_simulation(params, max_current=None):
    
    params['n_points']=[100,100]
    current=get_bias_triangles(params, max_current=max_current)
    current=np.rot90(current,2)
    
    return current



def simulate(n_imgs, return_sampling_factors=False , visualise=False, sample_factors=standard_sample_factors):
    """
    wrapper function to get n_imgs number of simulations
    """
    '''if sample_factors_func is not None:
        sample_factors=sample_factors_func'''

    sample_facs=[]
    imgs=[]
    psb_label=[]

    for i in tqdm(range(n_imgs)):

        params=sample_factors()
        sample_facs.append(params)
        psb=params['in_psb']

        
        
        
        params['in_psb']=False
        img_no_psb=get_simulation(params)
        if visualise:
            plt.imshow(img_no_psb)
            plt.show()
        
        
        ###########
        params['in_psb']=psb
        img=get_simulation(params, max_current=np.max(img_no_psb))
        if visualise:
            print('PSB:', in_psb)
            plt.imshow(img)
            plt.show()
        
        #sample_facs.append(sampled)
        this_imgs=normalise(np.array([img,img_no_psb]))
        #this_imgs=(np.array([img,img_no_psb]))
        imgs.append(this_imgs)
        psb_label.append(psb)
        if visualise:
            print('----')
    if return_sampling_factors:
        return imgs, psb_label, sample_facs
    else:
        return imgs, psb_label
   


def simulate_for_viola_jones(n_imgs, return_sampling_factors=False , visualise=False,  sample_factors=standard_sample_factors):
    """
    wrapper function to get n_imgs number of simulations
    """
    '''if sample_factors_func is not None:
        sample_factors=sample_factors_func'''
    sample_facs=[]
    imgs=[]

    for i in tqdm(range(n_imgs)):

        params=sample_factors()
        sample_facs.append(params)
        
        params['in_psb']=False
        
        img=get_simulation(params)
        
        if visualise:
            plt.imshow(img)
            plt.show()
        
        img=normalise(img)
        
        imgs.append(img)
        
        if visualise:
            print('----')
    if return_sampling_factors:
        return imgs, sample_facs
    else:
        return imgs
    
def sample_factors():
    
    mu_s = np.random.uniform(-0.5, 0.5)
    bias = np.random.uniform(-2,-0.1)
    mu_d = mu_s + bias
    temp = np.random.uniform(0.1, 1)
    shift = np.random.uniform(0.01, 2)
    
    lever_arms = np.random.uniform(0.5, 1.5, 2)
    cross_talk = np.random.uniform(0.0, 0.5, 2)
    
    n_pots_1 = np.random.randint(2, 4)
    n_pots_2 = np.random.randint(2, 7)
    
    charging1 = np.random.uniform(bias - 0.2, bias + 0.2)
    epsilon = charging1 / 4
    charging11_deltas = np.array([0] + [*np.random.uniform(-epsilon, epsilon, n_pots_1 - 1)])
    
    charging2 = np.random.uniform(0.2, 0.5)
    epsilon = charging2 / 4
    charging21_deltas = np.array([0] + [*np.random.uniform(-epsilon, epsilon, n_pots_2 - 1)])
    
    dot1_pot_arrangement = charging11_deltas+[i*charging1 for i in range(n_pots_1)]
    
    dot2_pot_arrangement = charging21_deltas+[i*charging2 for i in range(n_pots_2)]
    
    gamma_s = np.random.uniform(0.01, 0.5, n_pots_1)
    gamma_d = np.random.uniform(0.01, 0.5, n_pots_2)
    
    jitter_var = np.random.uniform(0,0.05)
    
    in_psb = np.random.choice((False, False))
    
    tc = np.random.uniform(0.01, 0.4, [n_pots_1, n_pots_2]) 
    
    gaussian_blur = np.random.uniform(0.8, 1.2)
    white_noise_level = np.random.uniform(3E-2, 7E-2)
    
    blank_space= 0.1
        
    return {'tc':tc,'gamma_s':gamma_s,'gamma_d':gamma_d,
            'mu_s':mu_s,'mu_d':mu_d,'temp':temp,'lever_arms':lever_arms, 
             'cross_talk':cross_talk, 'shift':shift,
            'dot1_pot_arrangement':dot1_pot_arrangement,'dot2_pot_arrangement':dot2_pot_arrangement,
            'in_psb':in_psb,'jitter_var':jitter_var,
           'gaussian_blur':gaussian_blur,'white_noise_level':white_noise_level,
           'blank_space':blank_space}
    
    
# sys.argv
# 1 num samples
# 2 output path for npy files
# (3 output path for png files)

n_samples = int(sys.argv[1])
output_npy_path = sys.argv[2]
samples=simulate_for_viola_jones(n_samples, sample_factors=sample_factors)#
bar = progressbar.ProgressBar()
for i, sample in bar(enumerate(samples)):
    np.save(os.path.join(output_npy_path, "simulation-{}".format(i)), sample, allow_pickle=True)