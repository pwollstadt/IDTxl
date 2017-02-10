addpath('~/repos/wmtsa-matlab-0.2.6/wmtsa/')
run ~/repos/wmtsa-matlab-0.2.6/wmtsa/startup.m     % compile c files if necessary
addpath ~/repos/wmtsa-matlab-0.2.6/munit/munit     % Add MUnit toolbox to path.

N = 20;
Vin = 0:N-1;
j = 3;
coeff_length = 16;
ht = 0:coeff_length-1;
gt = 0:coeff_length-1;
[Wt_j, Vout] = modwtj(Vin, ht, gt, j);
disp(Vout)

% Python output:
% output modwtj - Wout: [ 0.04012019  0.55435283  0.59574893  0.39338066  0.71219834  0.87718698
%   0.13572487  0.9638539   0.17673328  0.24642782  0.50325363  0.87908735
%   0.9666808   0.67575844  0.29310915  0.12573063  0.10030681  0.31808279
%   0.62969807  0.44807319]