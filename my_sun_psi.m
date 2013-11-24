addpath('/home/eecs/pulkitag/Research/codes/codes/myutils/Matlab');
addpath('/home/eecs/pulkitag/Research/codes/codes/pkgs/fisher');
featpipem_addpaths;
add_toolbox('vlfeat');
add_toolbox('libsvm');

clear all;
voc_size = 256;
fileNum = 4;
prms.voc_size = voc_size; % vocabulary size
desc_dim = 80; % descriptor dimensionality after PCA
DataDir = '/work4/pulkitag/projFisher/sun';

%% initialize experiment parameters
prms.experiment.name = sprintf('FK_%d_%d',fileNum,voc_size); % experiment name - prefixed to all output files other than codes
%prms.experiment.name = sprintf('FK_%d',voc_size); % experiment name - prefixed to all output files other than codes
prms.experiment.codes_suffix = prms.experiment.name;%'LFK'; % string prefixed to codefiles (to allow sharing of codes between multiple experiments)
prms.experiment.dict_suffix = 'FK'; % string prefixed to share the learnt dictionary.
%prms.experiment.dict_suffix = sprintf('FK_%d',fileNum); % string prefixed to share the learnt dictionary.
prms.experiment.classif_tag = ''; % additional string added at end of classifier and results files (useful for runs with different classifier parameters)
prms.imdb = load(strcat(DataDir,sprintf('/imdb/imdb_sun_%d.mat',fileNum))); % IMDB file
prms.codebook = fullfile(DataDir, sprintf('codebooks/%s_gmm_%d_%d.mat',prms.experiment.dict_suffix, voc_size, desc_dim)); % desired location of codebook
prms.dimred = fullfile(DataDir, sprintf('dimred/%s_pca_%d.mat',prms.experiment.dict_suffix, desc_dim)); % desired location of low-dim projection matrix
prms.experiment.dataset = 'sun'; % dataset name - currently only VOC2007 supported
prms.experiment.evalusing = 'accuracy'; % evaluation method - currently only precision recall supported

prms.paths.dataset = '/work4/pulkitag/data_sets/sun/SUN397/';
prms.paths.codes = fullfile(DataDir,sprintf('codes/%s/',prms.experiment.codes_suffix)); % path where codefiles should be stored
prms.paths.compdata = fullfile(DataDir,sprintf('compdata/%s/',prms.experiment.codes_suffix)); % path where all other compdata (kernel matrices, SVM models etc.) should be stored
prms.paths.results = fullfile(DataDir,sprintf('results/%s/',prms.experiment.codes_suffix)); % path where results should be stored

prms.tmpModelFile = strcat(DataDir,'/models/metadata/%s/tmp_w.mat'); 
prms.paths.model_metadata = strcat(DataDir,'/models/metadata/%s');
prms.classifier_posList = strcat(DataDir,'/models/metadata/%s/posList.mat');
prms.classifier_negList = strcat(DataDir,'/models/metadata/%s/negList.mat');

prms.chunkio.chunk_size = 100; % number of encodings to store in single chunk
prms.chunkio.num_workers = max(matlabpool('size'), 1);%1; % number of workers to use when generating chunks
disp(sprintf('Num Workers found are %d',prms.chunkio.num_workers));

% initialize split parameters
prms.splits.train = {'train', 'val'}; % cell array of IMDB splits to use when training
prms.splits.test = {'test'}; % cell array of IMDB splits to use when testing

%Feature Parameters
prms.isMyFeat = false;%false; %true: use my matlab implementation of fv.
prms.sparsityThresh = 0.5; %Sparsity used while computing fisher vector.
prms.is_fv_order2 = true;%false;
prms.fisherVecDim = (1+prms.is_fv_order2)*desc_dim*voc_size;
if(prms.isMyFeat)
    prms.myFeat_l2 = false;
    prms.myFeat_he = true;
end

%other parameters
prms.isVal = false;%true;
prms.codeLogPath = '/work4/pulkitag/projFisher/logs/';
prms.numGauss = voc_size;
prms.codebookFile = prms.codebook;
prms.numfvPerChunk = 250;
prms.wInitFile = strcat(DataDir,'/compdata/',prms.experiment.name,'_%s_classifier.mat');
prms.gaussDim = desc_dim;

prms.prmsFileName = strcat(DataDir,'/prms/',prms.experiment.name,'_prms.mat');

if(~(exist(prms.paths.codes,'dir')==7))
    system(['mkdir ' prms.paths.codes]);
end
if(~(exist(prms.paths.compdata,'dir')==7))
    system(['mkdir ' prms.paths.compdata]);
end
if(~(exist(prms.paths.results,'dir')==7))
    system(['mkdir ' prms.paths.results]);
end

%%%%% Learning Codebook %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% initialize experiment classes
featextr = featpipem.features.PhowExtractor();
featextr.step = 3;
%featextr.sizes = [16];
%featextr.remove_zero=true;

if desc_dim ~= 128
    dimred = featpipem.dim_red.PCADimRed(featextr, desc_dim);
    featextr.low_proj = featpipem.wrapper.loaddimred(dimred, prms);
else
   
   featextr.low_proj = [];
end

%return;
% Store dimensionality reduced SIFT features for each image
%featpipem.chunkio.compRawFeatIMDB(prms, featextr);

% train/load codebook
prms.myCodebookTrain = false;%true;
codebkgen = featpipem.codebkgen.GMMCodebkGen(featextr, voc_size);
codebkgen.GMM_init = 'kmeans';%'rand';
codebook = featpipem.wrapper.loadcodebook(codebkgen, prms);

% save_prms file
save(prms.prmsFileName,'prms','featextr');

%disp('Waiting for user..');
%keyboard;

%save features
%SubmitJobs_SaveFeatures(prms,prms.prmsFileName);

% %Get w_init
%   c = 1.6;
%   ap = get_initW(c,prms,codebook,featextr);
% %TrainClassifiers(prms,{'train'});

%% Cross Validate
bCrossValSVM = false;%true;
%If cross Valdidate.
%initialize encoder + pooler
encoder = featpipem.encoding.FKEncoder(codebook);
encoder.pnorm = single(0.0);
encoder.alpha = single(1.0);
encoder.grad_weights = false;
encoder.grad_means = true;
if(prms.is_fv_order2)
    encoder.grad_variances = true;
else
    encoder.grad_variances = false;
end


pooler = featpipem.pooling.SPMPooler(encoder);
pooler.subbin_norm_type = 'l2';
pooler.norm_type = 'none';
pooler.pool_type = 'sum';
pooler.kermap = 'hellinger';
pooler.post_norm_type = 'l2';

% pooler = featpipem.pooling.MyPooler(encoder);
% pooler.norm_type = 'none';%s'l2';%
% pooler.kermap =  'hellinger';%'none';%
% pooler.post_norm_type = 'l2'; %'none';%Post Normalization only works if kermap is
% %not none, other wise change norm_type.

%assert(prms.fisherVecDim==pooler.get_output_dim);
prms.fisherVecDim=pooler.get_output_dim;
%keyboard;

compute_all_chunks(prms,featextr, pooler);

classifier = featpipem.classification.svm.LibSvmDual();
if bCrossValSVM
    prms.splits.train = {'train'};
    prms.splits.test = {'val'};
    c = [0.001 0.005 0.01 0.05 0.1 0.5 1 1.6 3.2 6 10];
    %c = [1.6];
    maxmap = 0;
    maxci = 1;
    for ci = 1:length(c)
        prms.experiment.classif_tag = sprintf('c%g', c(ci));
        classifier.c = c(ci);
        acc = featpipem.wrapper.dstest(prms, codebook, featextr, encoder, pooler, classifier);
        disp(sprintf('Accuracy with c %f is %f',c(ci),mean(acc{1})));
        if (mean(acc{1}) > maxmap)
            maxci = ci;
            maxmap = mean(acc{1});
        end
    end
    
    prms.splits.train = {'train','val'};
    prms.splits.test = {'test'};
    prms.experiment.classif_tag = sprintf('TESTc%f', c(maxci));
    classifier.c = c(maxci);
    acc = featpipem.wrapper.dstest(prms, codebook, featextr, encoder, pooler, classifier);
else
    classifier.c = 3.2;
    acc = featpipem.wrapper.dstest(prms, codebook, featextr, encoder, pooler, classifier);
end



