function [] = my_sun(voc_size,varargin)
addpath('/auto/k1/pulkit/Codes/utils/matlab');
add_toolbox('vlfeat');
add_toolbox('fisher');
%add_toolbox('libsvm');
if(matlabpool('size')==0)
    matlabpool(4);
end
%clear all;

%voc_size = 256;
prms.voc_size = voc_size; % vocabulary size
desc_dim = 80; % descriptor dimensionality after PCA
pca_desc_dim = 80;
DataDir = '/auto/k6/pulkit/data/scene/sun/experiment';
DataDirModel = '/auto/k6/pulkit/data/scene/';

isColor = false;

%Run
runNum = 0;
prms.runNum = runNum;

%im params
imSz = 480;
prms.imSz = imSz;

%Kind of sift
stepSize = 3;
is_rootSift = true;
spatialEmbedding = 0;
grid_sz = 1;

%Kind of pooling
poolerType = 'spm';%'hor';
numHorDivs = 3;%0;

%Kind of normalization
fkerMap = 'he025';%'hellinger';

%type of encoding
encoderType = 'fisher'; %'fisher'
if (~isempty(varargin))
	encoderType = varargin{1};
end
disp(encoderType);

%SVW Details
nswWords = 4;
swThresh = 0.5;

if strcmp(poolerType,'hor')
	poolStr = sprintf('hor%d',numHorDivs);
else
	poolStr = poolerType;
end

if(~isColor)
    if(is_rootSift)
        expPreStr = sprintf('LFK_root_imsz%d_enc%s',imSz,encoderType);
        %expPreStr = sprintf('LFK_root');
    else
        expPreStr = 'LFK';
    end
else
    prms.isColor = true;
    expPreStr = 'LFK_color_pca_v1';
end


if(prms.runNum>0)
    expPreStr = strcat(expPreStr,sprintf('_run%d',runNum));
end


switch spatialEmbedding
    case 0
        detailStr = sprintf('voc%d_pool%s_%s_step%d',voc_size,poolStr,fkerMap,stepSize);
    case 1
		% add (x,y) location to descriptor, thus +2
        detailStr = sprintf('voc_%d_pool_%s_%s_step_%d%s_spatial',voc_size,poolStr,fkerMap,stepSize);
        desc_dim = desc_dim + 2;
end


%% initialize experiment parameters
prms.experiment.name = sprintf('%s_%s_o2',expPreStr,detailStr); % experiment name - prefixed to all output files other than codes
prms.experiment.codes_suffix = prms.experiment.name;%'LFK'; % string prefixed to codefiles (to allow sharing of codes between multiple experiments)
prms.experiment.dict_suffix = expPreStr;%'LFK'; % string prefixed to share the learnt dictionary.
prms.experiment.classif_tag = ''; % additional string added at end of classifier and results files (useful for runs with different classifier parameters)
prms.imdb = load(strcat(DataDir,'/imdb/imdb_1.mat')); % IMDB file

switch spatialEmbedding
    case 0
        prms.codebook = fullfile(DataDirModel, sprintf('codebooks/%s_%d_%d.mat',prms.experiment.dict_suffix, voc_size, desc_dim)); % desired location of codebook
    otherwise
        prms.codebook = fullfile(DataDirModel, sprintf('codebooks/%s_%d_%d_spatial.mat',prms.experiment.dict_suffix, voc_size, desc_dim)); % desired location of codebook
end
prms.dimred = fullfile(DataDirModel, sprintf('dimred/%s_pca_%d.mat',prms.experiment.dict_suffix, pca_desc_dim)); % desired location of low-dim projection matrix
prms.experiment.dataset = 'scene';
prms.experiment.evalusing = 'class_accuracy';

prms.paths.dataset = '/auto/k6/pulkit/data/scene/sun/SUN397/';
prms.prmsFileName = fullfile(DataDir,sprintf('prms/%s_prms.mat',prms.experiment.codes_suffix));
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

%prms.prmsFileName = strcat(DataDir,'/',prms.experiment.name,'_prms.mat');

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

if(~isColor)
    switch spatialEmbedding
        case 0
            featextr = featpipem.features.PhowExtractor();
        case 1
            featextr = featpipem.features.SpatialPhowExtractor();
        case 2
            featextr = featpipem.features.SpatialPhowExtractorV2();
            featextr.grid_sz = grid_sz;
    end
    featextr.step = stepSize;
    featextr.is_rootSift = is_rootSift;
    featextr.imSz = imSz;
    %featextr.sizes = [4];
    if(featextr.is_rootSift)
        featextr.remove_zero=true;
    end

    if desc_dim ~= 128
        dimred = featpipem.dim_red.PCADimRed(featextr, desc_dim);
        featextr.low_proj = featpipem.wrapper.loaddimred(dimred, prms);
    else

        featextr.low_proj = [];
    end
else

    featextr = featpipem.features.ColorExtractor();
    %featextr.sizes = %4*[4 6 8 10];
    featextr.sizes = [4 6 8 10];
    featextr.imSz = imSz;
    dimred = featpipem.dim_red.PCADimRed(featextr, desc_dim);
    featextr.low_proj = featpipem.wrapper.loaddimred(dimred, prms);
    
end

prms.myCodebookTrain = false;%true;

%initialize encoder + pooler
disp('Initializing Encoder');
disp(encoderType);
switch encoderType
	case 'fisher'
		% train/load codebook
		disp('Encoding Fisher');
		codebkgen = featpipem.codebkgen.GMMCodebkGen(featextr, voc_size);
		codebkgen.GMM_init = 'kmeans';%'rand';
		codebook = featpipem.wrapper.loadcodebook(codebkgen, prms);
		
		encoder = featpipem.encoding.FKEncoder(codebook);
		encoder.pnorm = single(0.0);
		encoder.alpha = single(1.0);
		encoder.grad_weights = false;%true;%false;
		encoder.grad_means = true;
		if(prms.is_fv_order2)
			encoder.grad_variances = true;
		else
			encoder.grad_variances = false;
		end


	case 'vq'
		%% train/load codebook
		disp('Encoding: VQ');
		codebkgen = featpipem.codebkgen.KmeansCodebkGen(featextr, voc_size);
		codebkgen.descount_limit = 10e5;
		codebook = featpipem.wrapper.loadcodebook(codebkgen, prms);

		encoder = featpipem.encoding.VQEncoder(codebook);
		encoder.max_comps = 25; % max comparisons used when finding NN using kdtrees
		encoder.norm_type = 'none'; % normalization to be applied to encoding (either 'l1' or 'l2' or 'none')
end



%Note: Post Normalization only works if kermap is not none, other wise change norm_type.
disp('Intializing Pooler');
switch poolerType
    case 'spm'
        pooler = featpipem.pooling.SPMPooler(encoder);
        pooler.subbin_norm_type = 'l2';
        pooler.norm_type = 'none';
        pooler.pool_type = 'sum';
        pooler.kermap = fkerMap;%'he025';%'hellinger';
        pooler.post_norm_type = 'l2';
        pooler.horiz_divs = numHorDivs;

    case 'hor'

        pooler = featpipem.pooling.MyPooler(encoder);
        pooler.subbin_norm_type = 'l2';
        pooler.norm_type = 'none';
        pooler.pool_type = 'sum';
        pooler.kermap = fkerMap;%'he025';
        pooler.post_norm_type = 'l2';
        pooler.horiz_divs = numHorDivs;

    case 'simpleLoc'
        pooler = featpipem.pooling.MySimpleLocPooler(encoder);
        pooler.subbin_norm_type = 'l2';
        pooler.norm_type = 'none';
        pooler.pool_type = 'sum';
        pooler.kermap = fkerMap;%'he025';
        pooler.post_norm_type = 'l2';

    case 'gridLoc'
        pooler = featpipem.pooling.MySimpleLocPooler(encoder);
        pooler.subbin_norm_type = 'l2';
        pooler.norm_type = 'none';
        pooler.pool_type = 'sum';
        pooler.kermap = fkerMap;%'he025';
        pooler.post_norm_type = 'l2';

	otherwise

        error(my:argChk,'Invalid Pooler Type');

end

pooler.subbin_norm_type = 'l2';
pooler.norm_type = 'none';
pooler.pool_type = 'sum';
pooler.kermap = fkerMap;%'he025';
pooler.post_norm_type = 'l2';

prms.fisherVecDim=pooler.get_output_dim;

% save_prms file
save(prms.prmsFileName,'prms','featextr');

%Compute the features.
disp('Computing Features'); 
compute_all_chunks(prms,featextr, pooler);


