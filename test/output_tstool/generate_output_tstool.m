function generate_output_tstool

addpath ~/repos/TRENTOOL3/tstool_functions/mex_linux/mex64

%% ABA04

for trial = 1:3
    load(sprintf('~/repos/trentoolxl/testing/data/ABA04_trial_%d_ps.mat', trial))
    
    atria_source = nn_prepare(single(ps_source),'maximum');
    atria_target = nn_prepare(single(ps_target),'maximum');
    atria_predpoint = nn_prepare(single(ps_predpoint),'maximum');
    
    neigh_source = nn_search(single(ps_source),atria_source,1:size(ps_source,1),4,0);
    save(sprintf('~/repos/trentoolxl/testing/output_tstool/ABA04_trial_%d_k_neigh_source.mat',trial), 'neigh_source')
    neigh_target = nn_search(single(ps_target),atria_target,1:size(ps_source,1),4,0);
    save(sprintf('~/repos/trentoolxl/testing/output_tstool/ABA04_trial_%d_k_neigh_target.mat',trial), 'neigh_target')
    neigh_predpoint = nn_search(single(ps_predpoint),atria_predpoint,1:size(ps_source,1),4,0);
    save(sprintf('~/repos/trentoolxl/testing/output_tstool/ABA04_trial_%d_k_neigh_predpoint.mat',trial), 'neigh_predpoint')
end
clear 

%% simple

trial = 1;
load(sprintf('~/repos/trentoolxl/testing/data/simple_trial_%d_ps.mat', trial))

atria = nn_prepare(single(ps),'maximum');
neigh = nn_search(single(ps),atria,1:size(ps,1),4,0);
save(sprintf('~/repos/trentoolxl/testing/output_tstool/simple_trial_%d_k_neigh.mat',trial), 'neigh')