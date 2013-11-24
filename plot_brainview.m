addpath(genpath('~dustin/code/matlab/mriTools/'));
addpath(genpath('~dustin/code/matlab/utils/'));

% CREATE SOME METADATA USED BY THE PLOTTER TO LOOK UP ANATOMICAL INFO
md.subj = 'ds';
md.study = 'colornatims';

% CREATE A VOLUME OF DATA (WHAT YOU WANT TO PLOT)
% MUST BE [104 X 104 X 25] TO USE THE roi INDICES
vol = zeros(104,104,25); 

% GET ROIS
dataHome = '/auto/k7/dustin/data/MRI/DS/colorNatims/';
tmp = load(fullfile(dataHome,'rois.mat'));
roi = tmp.roiVox; clear tmp

% ISOLATE THE VOXELS IN PPA
ppaIdx = [roi.lh.ppa;roi.rh.ppa];
vol(ppaIdx) = 1;

expName = get_expName('fisher',8);
modelStr =  fullfile(modelDataPath,'%s',strcat(expName,'_ignore_tp1.00'));
modelFileName = strcat(sprintf(modelStr,'final_results'),'.mat');
disp(modelFileName);
model = load(modelFileName);
regionNames = fields(model);

corr = model.(regionNames{r})  
