function generate_output_trentool

addpath ~/TRENTOOL_gpu_functions/

%%
for trial = 1:3
    load(sprintf('~/repos/trentoolxl/testing/data/ABA04_trial_%d_ps.mat', trial))
    
    neigh_source = fnearneigh_gpu(single(ps_source), single(ps_source), 4, 0, 1);
    save(sprintf('~/repos/trentoolxl/testing/output_trentool/ABA04_trial_%d_k_neigh_source.mat',trial), 'neigh_source')
    neigh_target = fnearneigh_gpu(single(ps_target), single(ps_target), 4, 0, 1);
    save(sprintf('~/repos/trentoolxl/testing/output_trentool/ABA04_trial_%d_k_neigh_target.mat',trial), 'neigh_target')
    neigh_predpoint = fnearneigh_gpu(single(ps_predpoint), single(ps_predpoint), 4, 0, 1);
    save(sprintf('~/repos/trentoolxl/testing/output_trentool/ABA04_trial_%d_k_neigh_predpoint.mat',trial), 'neigh_predpoint')
end

%%
ps_source_all = [];
ps_target_all = [];
ps_predpoint_all = [];

for trial = 1:3
    load(sprintf('~/repos/trentoolxl/testing/data/ABA04_trial_%d_ps.mat', trial))
    
    ps_source_all = cat(1, ps_source_all, ps_source);
    ps_target_all = cat(1, ps_target_all, ps_target);
    ps_predpoint_all = cat(1, ps_predpoint_all, ps_predpoint_all);
end

neigh_source = fnearneigh_gpu(single(ps_source_all), single(ps_source_all), 4, 0, 3);
save(sprintf('~/repos/trentoolxl/testing/output_trentool/ABA04_trial_%d_k_neigh_source_chunk.mat',trial), 'neigh_source')
neigh_target = fnearneigh_gpu(single(ps_target_all), single(ps_target_all), 4, 0, 3);
save(sprintf('~/repos/trentoolxl/testing/output_trentool/ABA04_trial_%d_k_neigh_target_chunk.mat',trial), 'neigh_target')
neigh_predpoint = fnearneigh_gpu(single(ps_predpoint_all), single(ps_predpoint_all), 4, 0, 3);
save(sprintf('~/repos/trentoolxl/testing/output_trentool/ABA04_trial_%d_k_neigh_predpoint_chunk.mat',trial), 'neigh_predpoint')

%% simple

trial = 1;
load(sprintf('~/repos/trentoolxl/testing/data/simple_trial_%d_ps.mat', trial))

neigh = fnearneigh_gpu(single(ps), single(ps), 4, 0, 1);
save(sprintf('~/repos/trentoolxl/testing/output_trentool/simple_trial_%d_k_neigh.mat',trial), 'neigh')
