% Startup the project
%
% (c) MJMJ/2017

% Do: cd <root_folder_cnngait>

CNNGAIT_PATH = pwd; % *** Or set me manually ***

%% Set VLFeat library, if needed
PATH_VLFEAT = '/home/mjmarin/libs/vlfeat-0.9.19/'; % *** SET ME ***
run(fullfile(PATH_VLFEAT, 'toolbox/vl_setup.m'));

%% Set MatConvNet library
PATH_MATCONVNET = '/home/mjmarin/libs/matconvnet-1.0-beta24/'; % *** SET ME ***

run(fullfile(PATH_MATCONVNET,'matlab/vl_setupnn.m'));

PATH_MATCONVNETREG = fullfile(CNNGAIT_PATH, '3rdparts/matconvnet-deepReg');
run(fullfile(PATH_MATCONVNETREG,'matlab/vl_setupnn.m'));

%% Set CNNGait utils
addpath(genpath(pwd))

%% Additional paths?