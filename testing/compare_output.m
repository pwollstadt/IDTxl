function compare_output(output, prog1, prog2, dataset)

% prog1, prog2:
%   'trentool', 'matlab', 'tstool', 'jidt'
%
% output:
%   'k_neigh_source', 'k_neigh_target', 'k_neigh_predpoint'
%
% dataset:
%   'ABA04'
%   'simple'

datapath = '/data/home1/pwollsta/repos/trentoolxl/testing/';

prog1_output = load(...
    sprintf('%soutput_%s/%s_trial_1_%s.mat', datapath, prog1, dataset, output));
prog2_output = load(...
    sprintf('%soutput_%s/%s_trial_1_%s.mat', datapath, prog2, dataset, output));

f1 = fieldnames(prog1_output);
f2 = fieldnames(prog2_output);

n_eqal = sum(sum(prog1_output.(f1{1}) == prog2_output.(f2{1})));

if n_eqal ~= numel(prog1_output.(f1{1}))
    warning('Results do not match!')
end

%%
load ~/repos/trentoolxl/testing/data/simple_trial_1_ps.mat
figure
scatter(ps(:,1),ps(:,2));
hold on
a = [1:10]'; b = num2str(a); c = cellstr(b);
dx = 0.1; dy = 0.1; 
text(ps(:,1)+dx, ps(:,2)+dy, c);

scatter(ps(1,1), ps(1,2),'r','MarkerFaceColor','r')
scatter(ps(2:5,1), ps(2:5,2),'r','MarkerFaceColor','g')
xlim([0 11]); ylim([0 11])
%%

fprintf('\n')