function generate_output_matlab

k = 5;

%%
for trial = 1:3
    load(sprintf('~/repos/trentoolxl/testing/data/ABA04_trial_%d_ps.mat', trial))
    
    neigh_source = knnsearch(single(ps_source),single(ps_source),'K',k, 'Distance', 'chebychev');
    neigh_source = neigh_source(:,2:k);
    save(sprintf('~/repos/trentoolxl/testing/output_matlab/ABA04_trial_%d_k_neigh_source.mat',trial), 'neigh_source')
    neigh_target = knnsearch(ps_target,ps_target,'K',k, 'Distance', 'chebychev');
    neigh_target = neigh_target(:,2:k);
    save(sprintf('~/repos/trentoolxl/testing/output_matlab/ABA04_trial_%d_k_neigh_target.mat',trial), 'neigh_target')
    neigh_predpoint = knnsearch(ps_predpoint,ps_predpoint,'K',k, 'Distance', 'chebychev');
    neigh_predpoint = neigh_predpoint(:,2:k);
    save(sprintf('~/repos/trentoolxl/testing/output_matlab/ABA04_trial_%d_k_neigh_predpoint.mat',trial), 'neigh_predpoint')
end

%% simple

trial = 1;
load(sprintf('~/repos/trentoolxl/testing/data/simple_trial_%d_ps.mat', trial))

neigh = knnsearch(single(ps),single(ps),'K',5);
neigh = neigh(:,2:k);
save(sprintf('~/repos/trentoolxl/testing/output_matlab/simple_trial_%d_k_neigh.mat',trial), 'neigh')