
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


# For cross section energy range

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)



# Strangeness enhancement study

def f_fs(f_s,parent):
    if parent=="Pions": 
        return (1 - f_s)
    elif parent=="Kaons":
        return (1 + 6.6*f_s)
    return 1

def mu_parent(G,
              dataset_name,
              parent,
              parent_lambdas,
              light_generator_base="SIBYLL",
              charm_generator_base="SIBYLL"):
    Np = len(G[dataset_name][parent].keys())
    lambda_bar = np.sum(parent_lambdas)
    generators = list(G[dataset_name][parent].keys())
    if parent in ["Pions","Kaons"]:
        generators.remove(light_generator_base)
        mu_p = G[dataset_name][parent][light_generator_base] * (1 - lambda_bar)
    else:
        generators.remove(charm_generator_base)
        mu_p = G[dataset_name][parent][charm_generator_base] * (1 - lambda_bar)
    for generator,lam in zip(generators,parent_lambdas):
        mu_p += G[dataset_name][parent][generator] * (1 + Np*lam - lambda_bar)
    mu_p /= Np
    return mu_p
    
def mu(G,
       dataset_name,
       lambdas,
       light_generator_base="SIBYLL",
       charm_generator_base="SIBYLL",
       f_s = 0):
    assert(lambdas.shape==(3,len(G[dataset_name]["Pions"])-1))
    mu = 0
    for ip,parent in enumerate(G[dataset_name].keys()):
        if parent=="All": continue
        mu_p = mu_parent(G,dataset_name,parent,lambdas[ip],
                         light_generator_base=light_generator_base,
                         charm_generator_base=charm_generator_base)
        mu_p *= f_fs(f_s,parent)
        mu += mu_p
    return mu

def nllh(lambdas,G,f_s,observation="null"):
    lambdas = lambdas.reshape((3,4))
    LLH = 0
    for dataset_name in G.keys():
        mu_d = mu(G,dataset_name,lambdas=lambdas,f_s=f_s)
        if observation=="null": k_d = mu(G,dataset_name,lambdas=np.zeros_like(lambdas),f_s=0)
        elif observation=="alt": k_d = mu(G,dataset_name,lambdas=np.zeros_like(lambdas),f_s=f_s)
        LLH += k_d*np.log(mu_d) - mu_d - gammaln(k_d)
    #print(lambdas,LLH)
    return -LLH

def delta_llh(G,lambdas,f_s,observation="null"):
    lambdas = lambdas.reshape((3,4))
    delta_LLH = 0
    #print(lambdas)
    for dataset_name in G.keys():
        mu_d_null = mu(G,dataset_name,lambdas=np.zeros_like(lambdas),f_s=0)
        mu_d_alt = mu(G,dataset_name,lambdas=lambdas,f_s=f_s)
        if observation=="null": k_d = mu_d_null
        elif observation=="alt": k_d = mu(G,dataset_name,lambdas=np.zeros_like(lambdas),f_s=f_s)
        #print(dataset_name,mu_d_alt,mu_d_null)
        delta_LLH += k_d * np.log(mu_d_alt/mu_d_null) - (mu_d_alt - mu_d_null)
    return -2 * delta_LLH

def profile_llh(G,
                f_s,
                observation="null",
                light_generator_base="SIBYLL",
                charm_generator_base="SIBYLL"):
    
    args = (G,f_s,observation)
    cons = ({'type': 'ineq', 'fun': lambda x : 1 - sum(x[0:4])},
            {'type': 'ineq', 'fun': lambda x : 1 - sum(x[4:8])},
            {'type': 'ineq', 'fun': lambda x : 1 - sum(x[8:])})
    for dataset_name in G.keys():
        for ip,parent in enumerate(G[dataset_name].keys()):
            i_lambda_start = 4*ip
            i_lambda_end = 4*(ip+1)
            generators = list(G[dataset_name][parent].keys())
            if parent in ["Pions","Kaons"]:
                generators.remove(light_generator_base)
                X = G[dataset_name][parent][light_generator_base]
            else:
                generators.remove(charm_generator_base)
                X = G[dataset_name][parent][charm_generator_base]
            Y = sum([G[dataset_name][parent][g] for g in generators])
            lambda_sum_min = -X/(Y - (X+Y)/len(generators))
            #print(dataset_name,parent,f_s,lambda_sum_min)
            if not np.isnan(lambda_sum_min):
                cons += ({'type': 'ineq', 'fun': lambda x : -sum(x[i_lambda_start:i_lambda_end]) + lambda_sum_min},)
            
    x0 = np.zeros(3*4)
    bounds = ((-1,1) for _ in range(len(x0)))
    res = minimize(nllh,x0,args,
                   bounds=bounds,
                   constraints=cons,
                   method="SLSQP",
                   options={"maxiter":1000})
    # while np.all(res.x==np.zeros_like(res.x)):
    #     x0 = np.array(2*np.random.rand(12)-1,dtype=float) # between -1 and 1
    #     print(x0)
    #     res = minimize(nllh,x0,args,
    #                    bounds=bounds,
    #                    constraints=cons,
    #                    #method="COBYLA",
    #                    options={"maxiter":1000})
        
    
    #print(f_s)
    #print(res.fun)
    #print(np.all(res.x==np.zeros_like(res.x)))
    return res.fun


# Fisher matrix element functions

def fisher_matrix_element(G,
                          lambdas,
                          f_s=0,
                          parent_1=None,generator_1=None,
                          parent_2=None,generator_2=None,
                          dataset_name_1=None,
                          dataset_name_2=None,
                          f_s_1=False,
                          f_s_2=False):
    assert(lambdas.shape==(3,len(G["SINE_rate_bin0"]["Pions"])-1))
    element = 0
    if parent_1 and generator_1 and parent_2 and generator_2:
        for dataset_name in G.keys():
            k = mu(G,dataset_name,np.zeros_like(lambdas))
            _mu = mu(G,dataset_name,lambdas)
            prefactor = k / (_mu**2)
            diff1 = G[dataset_name][parent_1][generator_1] - np.average(list(G[dataset_name][parent_1].values()))
            diff2 = G[dataset_name][parent_2][generator_2] - np.average(list(G[dataset_name][parent_2].values()))
            element += prefactor*diff1*diff2
        return element
    elif f_s_1 and parent_2 and generator_2:
        for dataset_name in G.keys():
            k = mu(G,dataset_name,np.zeros_like(lambdas))
            _mu = mu(G,dataset_name,lambdas)
            diff2 = G[dataset_name][parent_2][generator_2] - np.average(list(G[dataset_name][parent_2].values()))
            pion_idx = np.where(list(G[dataset_name].keys())=="Pions")
            kaon_idx = np.where(list(G[dataset_name].keys())=="Kaons")
            term_left = k/(_mu**2) * f_fs(f_s,parent_2)*(6.6*mu_parent(G,dataset_name,"Kaons",lambdas[kaon_idx]) - mu_parent(G,dataset_name,"Pions",lambdas[pion_idx]))
            term_right = (k/_mu - 1) * (6.6*(parent_2=="Kaons") - (parent_2=="Pions"))
            element += diff2 * (term_left - term_right)
        return element
    elif f_s_2 and parent_1 and generator_1:
        for dataset_name in G.keys():
            k = mu(G,dataset_name,np.zeros_like(lambdas))
            _mu = mu(G,dataset_name,lambdas)
            diff1 = G[dataset_name][parent_1][generator_1] - np.average(list(G[dataset_name][parent_1].values()))
            pion_idx = np.where(list(G[dataset_name].keys())=="Pions")
            kaon_idx = np.where(list(G[dataset_name].keys())=="Kaons")
            term_left = k/(_mu**2) * f_fs(f_s,parent_1)*(6.6*mu_parent(G,dataset_name,"Kaons",lambdas[kaon_idx]) - mu_parent(G,dataset_name,"Pions",lambdas[pion_idx]))
            term_right = (k/_mu - 1) * (6.6*(parent_1=="Kaons") - (parent_1=="Pions"))
            element += diff1 * (term_left - term_right)
        return element
    elif f_s_1 and f_s_2:
        for dataset_name in G.keys():
            k = mu(G,dataset_name,np.zeros_like(lambdas))
            prefactor = k / (mu(G,dataset_name,lambdas)**2)
            pion_idx = np.where(list(G[dataset_name].keys())=="Pions")
            kaon_idx = np.where(list(G[dataset_name].keys())=="Kaons")
            element += prefactor * (6.6*mu_parent(G,dataset_name,"Kaons",lambdas[kaon_idx]) - mu_parent(G,dataset_name,"Pions",lambdas[pion_idx]))**2
        return element
    elif parent_1 and generator_1 and dataset_name_2:
        k = mu(G,dataset_name_2,np.zeros_like(lambdas))
        prefactor = k / (mu(G,dataset_name_2,lambdas)**2)
        diff1 = G[dataset_name_2][parent_1][generator_1] - np.average(list(G[dataset_name_2][parent_1].values()))
        return prefactor*diff1
    elif parent_2 and generator_2 and dataset_name_1:
        k = mu(G,dataset_name_1,np.zeros_like(lambdas))
        prefactor = k / (mu(G,dataset_name_1,lambdas)**2)
        diff2 = G[dataset_name_1][parent_2][generator_2] - np.average(list(G[dataset_name_1][parent_2].values()))
        return prefactor*diff2
    elif dataset_name_1 and dataset_name_2:
        k = mu(G,dataset_name_1,np.zeros_like(lambdas))
        prefactor = k / (mu(G,dataset_name_1,lambdas)**2)
        return prefactor*(dataset_name_1==dataset_name_2)
    else:
        print("Invalide fisher matrix element request")
        return -np.inf

def condition_number_thresholding(I, threshold=1e-5):
    # Step 1: Eigenvalue Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(I)

    # Step 2: Adjust Small Eigenvalues
    # For each eigenvalue, if it's smaller than threshold, set it to threshold
    regularized_eigenvalues = np.maximum(eigenvalues, threshold)

    # Step 3: Reconstruct the Regularized Matrix
    # Rebuild the Fisher matrix using the regularized eigenvalues
    regularized_I = (eigenvectors @ np.diag(regularized_eigenvalues) @ eigenvectors.T)

    return regularized_I

def pca_fisher_reduction(I, variance_threshold=0.999999):
    # Step 1: Eigenvalue Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(I)

    # Step 2: Sort eigenvalues and eigenvectors in descending order of eigenvalue magnitude
    indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[indices]
    sorted_eigenvectors = eigenvectors[:, indices]

    # Step 3: Determine number of components to satisfy variance threshold
    total_variance = np.sum(sorted_eigenvalues)
    variance_explained = np.cumsum(sorted_eigenvalues) / total_variance
    num_components = np.searchsorted(variance_explained, variance_threshold) + 1

    # Step 4: Reconstruct Reduced Fisher Matrix
    V_selected = sorted_eigenvectors[:, :num_components]
    Lambda_selected = np.diag(sorted_eigenvalues[:num_components])
    
    I_reduced = V_selected @ Lambda_selected @ V_selected.T

    return I_reduced

def profile_information_matrix(I, param_index):
    """
    Profiles over a single parameter in the Fisher information matrix.

    Args:
    - I: 2D NumPy array representing the Fisher information matrix.
    - param_index: int representing the index of the parameter to profile out.

    Returns:
    - I_profiled: 2D NumPy array representing the profiled information matrix.
    """
    # Remove the nth row and column to form the reduced information matrix
    I_reduced = np.delete(np.delete(I, param_index, axis=0), param_index, axis=1)
    
    # Extract the nth column, with the nth entry removed, to form the vector m
    m = np.delete(I[:, param_index], param_index)

    # The diagonal term for normalization
    I_nn = I[param_index, param_index]
    
    # Compute the outer product of m with itself, and normalize by I_nn
    m_outer = np.outer(m, m) / I_nn if I_nn != 0 else np.zeros_like(I_reduced)

    # Profiled information matrix
    I_profiled = I_reduced - m_outer

    return I_profiled


# Cross section data

# flux ratios from arXiv:2403.12520
nu_nubar_nue = 1.03
nu_nubar_numu = 0.62
faser_nue_erange = np.linspace(560,1740,100)
faser_nue_alpha = [2.4,1.3,1.8] # cross section scaling
faser_nue_xs = [1.2,0.7,0.8]
faser_numu_erange = np.linspace(520,1760,100)
faser_numu_alpha = [0.9,0.3,0.5] # cross section scaling
faser_numu_xs = [0.5,0.2,0.2]

# digitized from https://arxiv.org/abs/2412.03186
FASER_2024_numu =  {
    "CV":np.array([(180.6761477105885, 0.8297531057324536),                
          (420.912354878894, 1.1354680853477306),                
          (758.965872460004, 1.0105083219867992)]),                
    "yhigh":[1.1756671470368967,1.472091188507286,1.3100856077821803],             
    "ylow":[0.4903000772230219, 0.8013817283675393, 0.7035090253379932],             
    "xhigh":[299.74266150619275, 595.8410016418873, 996.212544463017],             
    "xlow": [99.46530771157279, 299.59149201811414, 595.9612445064665]
                   }

FASER_2024_numubar =  {
    "CV":np.array([(179.5535207321342, 0.3705122119667118),
          (517.3298818349599, 0.2875208289608982)]),                
    "yhigh":[0.5506930064364908, 0.377886506179412],             
    "ylow":[0.19097140731772225, 0.19610766493967247],             
    "xhigh":[300.1427884861591, 998.2695261080769],             
    "xlow": [99.59696761088219, 300.27067260673084]
                   }

FASER_2024_numu_numubar =  {
    "CV":np.array([(1391.092445829504, 0.42123333544078356)]),                
    "yhigh":[0.5612443276750535],             
    "ylow":[0.27576951028587404],             
    "xhigh":[1867.7979610231228],             
    "xlow": [997.6430390038568]
                   }
