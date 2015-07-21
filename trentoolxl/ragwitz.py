import embedding
import neighboursearch

def ragwitz(data, parameters):

	dim_candidates  = parameters.ragdim
	tau_candidates  = parameters.ragtau
	neighboursearch = getattr(neighboursearch, parameters.preprocessing_neighboursearch)
	
	for trial in data.trial.shape[1]
		for dim in dim_candidatesi+1:
			for tau in tau_candidates+1:
	
				pointset = embedding(data.trial[trial], dim, tau)
				neighbours = neighboursearch(pointset, parameters)

				for point in parameters.preprocessing_points+1:
					predicted_point[point] =  
					actual_point[point] =
	
	predicted_diff = predicted_point - actual_point
	mean_error = ((predicted_diff^2)/parameters.preprocessing_points)/data.trial.std(0, 1)

	mean_error.min
	# find dim and tau that minimize mean_error
	return embedding_parameters



			

