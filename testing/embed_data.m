
cd ~/repos/TRENTOOL3/private/

%%

for trial = 1:3;
    
    ts_1 = data.trial{trial}(1,:);
    ts_2 = data.trial{trial}(2,:);
    dim  = 5;
    tau = 3;
    delay = 3;
    extracond = 'none';
    
    pointset = TEembedding(ts_1, ts_2, dim, tau, delay, extracond);
    
    ps_predpoint = pointset.pointset_p2(:,1);
    ps_source = pointset.pointset_1;
    ps_target = pointset.pointset_2;
    
    csvwrite(...
        sprintf('~/repos/trentoolxl/testing/ABA04_trial_%d_ps_predpoint.csv', trial), ...
        'ps_predpoint')
    csvwrite(...
        sprintf('~/repos/trentoolxl/testing/ABA04_trial_%d_ps_source.csv', trial), ...
        'ps_source')
    csvwrite(...
        sprintf('~/repos/trentoolxl/testing/ABA04_trial_%d_ps_target.csv', trial), ...
        'ps_target')
    save(sprintf('~/repos/trentoolxl/testing/ABA04_trial_%d_ps.mat', trial), ...
        'ps_predpoint', 'ps_source', 'ps_target')
end