function [] = gabor_features(sfmin,sfmax)
addpath('/auto/k1/pulkit/Codes/utils/matlab');
DataDir = '/auto/k6/pulkit/data/scene/';

%Gabor Parameters
%ADD PATHS
addpath(genpath('/auto/k2/share/strflabGOLD/'))

% GET THE DEFAULT PARAMETER SET
params = preprocWavelets3d;

% MAKE IT A STATIC MODEL
params.veldivisions=1;
params.tfmax=1;
params.tsize=1;

% ADJUST THE FREQUENCY RANGE
params.sfmax=sfmax;
params.sfmin=sfmin;
params.f_step_log=1;

% NO NORMALIZATION
params.zeromean=0;
params.normalize=0;


%% initialize experiment parameters
prms.experiment.name = sprintf('gabor_sfmn%d_sfmx%d',sfmin,sfmax); 
prms.imdb = load(strcat(DataDir,'/imdb/imdb_scene.mat')); % IMDB file
prms.paths.dataset = '/auto/k6/pulkit/data/scene/gallantLabData/image/';
prms.prmsFileName = fullfile(DataDir,sprintf('prms/%s_prms.mat',prms.experiment.name));
prms.paths.codes = fullfile(DataDir,sprintf('codes/%s/',prms.experiment.name));
prms.splits.train = {'train', 'val'}; % cell array of IMDB splits to use when training
prms.splits.test = {'test'}; % cell array of IMDB splits to use when testing

prms.paths.trainvalCodes = fullfile(prms.paths.codes,'trainval');
prms.paths.testCodes = fullfile(prms.paths.codes,'test');

if ~exist(prms.paths.codes,'dir')
	system(['mkdir ' prms.paths.codes]);
	system(['mkdir ' prms.paths.trainvalCodes]);
	system(['mkdir ' prms.paths.testCodes]);
end

imdb = prms.imdb;
trainvalIdx = imdb.images.id(imdb.images.set==1 | imdb.images.set==2);
testIdx = imdb.images.id(imdb.images.set==3);

for i=1:1:length(trainvalIdx)
	imName = fullfile(prms.paths.dataset,imdb.images.name{trainvalIdx(i)});
	image = rgb2gray(imread(imName));
	% REDUCE IMAGE SIZE
	image = imresize(image,[128 128]);
	% PREPROCESS...
	[code,paramsOut] = preprocWavelets3d(image,params);
	code = code';
	outStr = imName(end-9:end-4);
	outFileName = fullfile(prms.paths.trainvalCodes,strcat(outStr,'.mat'));
	save(outFileName,'code');
end
	
for i=1:1:length(testIdx)
	imName = fullfile(prms.paths.dataset,imdb.images.name{testIdx(i)});
	image = rgb2gray(imread(imName));
	% REDUCE IMAGE SIZE
	image = imresize(image,[128 128]);
	% PREPROCESS...
	[code,paramsOut] = preprocWavelets3d(image,params);
	code = code';
	outStr = imName(end-9:end-4);
	outFileName = fullfile(prms.paths.testCodes,strcat(outStr,'.mat'));
	save(outFileName,'code');
end
prms.fisherVecDim = length(code); 
save(prms.prmsFileName,'prms');
	

end
