def set(data, idx):
    
    np.empty(idx.size[1], data.size(1)-idx * data.size(2))
    realisations = np.empty(1, data.size(1)-idx * data.size(2))
    
    for i in idx:
        realisations[i][:] = get_realisations.single_process(data[idx[0]], idx[1])
        
    return realisations


def single_process(data, idx):
    
    realisations = np.empty(1, data.size(1)-idx * data.size(2))
    idx_realisations = 0
    
    for sample in range(idx, data.size(1)):
        for repetition in range(0, data.size(2)):
            realisations[idx_realisations] = data[sample][repetition]
    
    return realisations