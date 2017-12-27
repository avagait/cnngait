% File: cg_demo_test.m
%
% CNN Gait Demo test
%
% (c) MJMJ/2017

%% Load model
dirdata = fullfile(CNNGAIT_PATH, 'data');
dirmodels = fullfile(dirdata, 'models');
modelname = 'dagV00_of25_id+gender0100+age0100_est_adp_60x60_N150_nf064/net-final.mat'; % *** SET ME ***
matmodel = fullfile(dirmodels, modelname);

netstr = load(matmodel); net = netstr.net; clear netstr
disp('Model loaded.')

% Prepare network for testing mode
net = fc_cnn_deploy(net) ;
net.mode = 'test';

%% Load test data
dirsamples = fullfile(dirdata, 'testsamples');
matdata = fullfile(dirsamples, 'matimdbtum_gaid_N155-n-05_06-of25_60x60.mat');
disp('Loading test data...')
samplestr = load(matdata); samples = samplestr.imdbtest.images; clear samplestr
disp('Test samples loaded.')

if mj_isCompressedData(samples.data)
   samples.data = mj_decompressOF(samples.data, 1.0/samples.compressFactor);
   dfactor = 1.0/samples.compressFactor;
   meanval = mean(samples.data(:));     % WARNING: this should be computed on training samples
   
   samples.data = single(samples.data) - meanval;
end

disp('Data ready.')

% Show one sample
figure(1), imagesc(samples.data(:,:,13,1)), colormap jet, axis image, title('Single OF channel'), colorbar

%% Run network
nTestSamples = 15; % Choose what you need
fprintf('Testing network on %d samples. \n', nTestSamples);
softmaxLayerName = 'probs'; % Options: 'probs' for id; 'probsG' for gender; 'full02a' for age
scoresSMtest = mj_classifyWithDAG(net, samples.data(:,:,:,1:nTestSamples), softmaxLayerName, 1);
[bestScore, bestId] = max(scoresSMtest);

%% See results
figure(2), clf, bar(bestId), title('Label per sample'), xlabel 'Samples', ylabel 'Id', grid on
