import nonuniform_embedding as nu
import mumpy as np

# idea: have a data matrix, use indices to access data, i.e., (a,b,c), where a is the time series, b is point in time, c is trial
# is passing indices and ALL data more expensive than cutting and passing data?


def multivariate_te(source_set, target, delta_min, delta_max):
    
    """ multivariate_te finds the effective network for the given source and 
    target processes. Uses non uniform embedding as proposed by Faes.
    """
    
    # find embedding, first for target then for sources
    idx_current_value = delta_max
    idx_candidate_set_target = np.array([0 idx_current_value-1])
    idx_candidate_set_source = np.array([0 idx_current_value-delta_min])
    embedding_target = nu.nonuniform_embedding(target, idx_current_value, idx_candidate_set_target)
    embedding_source = nu.nonuniform_embedding(source_set, idx_current_value, idx_candidate_set_source, embedding_target)
    
    # additiional pruning step 
    for candidate in conditional:
        realisations_current_candidate = get_realisations.single_process(data[candidate[0]], candidate[1])
        current_conditional = set_operations.substraction(conditional, candidate)
        realisations_current_conditional = get_realisations.set(data, current_conditional)
        temp_cmi = cmi_calculator_kraskov(realisations_current_value, realisations_current_candidate, realisations_current_conditional)
        significant = maximum_statistic(data, conditional, max_cmi)
        
        if !significant:
            conditional = current_conditional