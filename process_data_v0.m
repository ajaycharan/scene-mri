function [varargout] = process_data_v0(encType,vocSz,trainPercent,varargin) 

assert(trainPercent>0 && trainPercent<=1, 'Invalid trainPercent');
order=2;
paths.dataPath = '/auto/k6/pulkit/data/scene/gallantLabData/';
paths.featDataPath = '/auto/k6/pulkit/data/scene/';
paths.resultPath = fullfile(paths.featDataPath, 'final_results/');

%RUn Params
isProfile = false;
allVoxData = false;
isUnitTest = true;

if (isProfile)
	profile on;
end

switch encType
	case 'fisher'
		poolType = 'hor0'
		if isProfile
			expName = sprintf('LFK_root_imsz480_enc%s_debug_voc%d_pool%s_he025_step3_o%d',encType,vocSz,poolType,order);
		else
			expName = sprintf('LFK_root_imsz480_enc%s_voc%d_pool%s_he025_step3_o%d',encType,vocSz,poolType,order);
		end
	case 'vq'
		if vocSz<=256
			poolType='spm';
		else
			poolType='hor';
		end
		poolType = 'spm'
		expName = sprintf('LFK_root_imsz480_enc%s_voc%d_pool%s_he025_step3_o%d',encType,vocSz,poolType,order);
	case 'gabor'
		assert(~isempty(varargin),'3 inputs required for gabor');
	    sfMin = vocSz;
		sfMax = varargin{1};
		expName = sprintf('gabor_sfmn%d_sfmx%d',sfMin,sfMax);
		clear vocSz;	
	end

%Load the voxel data
dataHome = '/auto/k7/dustin/data/MRI/DS/colorNatims/';
load(fullfile(dataHome,'corticalVox.mat'),'cortVox');
load(fullfile(dataHome,'responses.mat'),'r');
trainvalVoxel = single(r.trn);
testVoxel = single(r.val);
clear r;

%Load the ROI Information
roiTmp = load(fullfile(dataHome,'rois.mat'));
roi =  roiTmp.roiVox;
clear roiTmp;

imdbFile = fullfile(paths.featDataPath,'imdb','imdb_scene.mat');
prmsFile = fullfile(paths.featDataPath,'prms',strcat(expName,'_prms.mat'))

%Imdb File
imdb = load(imdbFile);

%exp prms file
prms = load(prmsFile);
prms = prms.prms;

if isProfile
	runNum = varargin{1};
	prms.outFileName = fullfile(paths.resultPath,strcat(expName,sprintf('run%d.mat',runNum)));
	s = RandStream('mcg16807','Seed',runNum);
	RandStream.setGlobalStream(s);
else
	if allVoxData
		prms.outFileName = fullfile(paths.resultPath,strcat(expName,sprintf('_allvox_tp%.02f.mat',trainPercent)));
	else
		prms.outFileName = fullfile(paths.resultPath,strcat(expName,sprintf('_ignore_tp%.02f.mat',trainPercent)));
	end
end

lhFieldNames = fieldnames(roi.lh);
rhFieldNames = fieldnames(roi.rh);
fieldNames = union(lhFieldNames, rhFieldNames);
if exist(prms.outFileName,'file')
	eData = load(prms.outFileName);
	existingFields = fieldnames(eData);
	clear eData;
	fieldNames = setdiff(fieldNames,existingFields); 
	%system(['rm ' prms.outFileName]);
end

if isProfile
	%numFields = 1;
	numFields = length(fieldNames)
else
	numFields = length(fieldNames)
end


permutation = randperm(1260);
numTrainVal = ceil(trainPercent*1260);
permutation = permutation(1:numTrainVal);
%permutation = 1:1:numTrainVal;

if isUnitTest
	permutation = 1:1:1260;
	nDims = 100;
	nVox = 10;
	nTest = 150;

	fakeData = randn(numTrainVal,nDims);
	normData = sum(fakeData.*fakeData,2);
	fakeData = bsxfun(@rdivide,fakeData,normData);
	
	fakeTest = randn(nTest,nDims);
	normTest = sum(fakeTest.*fakeTest,2);
	fakeTest = bsxfun(@rdivide,fakeTest,normTest);	

	fakeW = randn(nDims,nVox);
	fakeVoxel = (fakeData*fakeW)';
	fakeTestVoxel = (fakeTest*fakeW)';
	resStruct = learn_params('fake',imdb,prms,fakeVoxel,fakeTestVoxel,numTrainVal,permutation,true,fakeData,fakeTest);
	corr = get_corr(resStruct.voxValue);
	disp(sprintf('NUmber of voxels are %d',length(corr)));
	disp(sprintf('Mean Correlation: %.2f, Median Corr: %.2f, Min Corr: %.2f, Max: %.2f',mean(corr),median(corr),max(corr),min(corr)));
	%err = norm(fakeW-resStruct.wMat);
	%disp(sprintf('Error in estimation is %f',err));
	varargout{1} = resStruct;
	
else
	if allVoxData
		numPerRound = 1000;
		numVoxels = size(trainvalVoxel,1);
		for i=1:1:ceil(numVoxels/numPerRound)
			st = (i-1)*numPerRound + 1;
			en = min(numVoxels,st + numPerRound - 1)
			
			trainvalResponse = trainvalVoxel(st:en,:);
			testResponse = testVoxel(st:en,:);
			assert(size(trainvalResponse,1)==size(testResponse,1),'Size Mismatch');
			name = sprintf('all%d',st);
			disp(name);
			learn_params(name,imdb,prms,trainvalResponse,testResponse,numTrainVal,permutation,false);

		end	
	else
		for i = 1:1:numFields
			name = fieldNames{i};
			isLh = ismember(name,lhFieldNames);
			isRh = ismember(name,rhFieldNames);
			idxLh = [];
			idxRh = [];
			if (isLh)
				[~,idxLh] = intersect(cortVox,roi.lh.(name));
			end
			if (isRh)
				[~,idxRh] = intersect(cortVox,roi.rh.(name));
			end
			idxAll = [idxLh;idxRh];

			trainvalResponse = trainvalVoxel(idxAll,:);
			testResponse = testVoxel(idxAll,:);

			assert(size(trainvalResponse,1)==size(testResponse,1),'Size Mismatch');
			disp(name);
			learn_params(name,imdb,prms,trainvalResponse,testResponse,numTrainVal,permutationi,false);
		end
	end
end

if isProfile
	p = profile('info');
	save('profileData','p');
end

end

function [varargout] = learn_params(regionName,imdb,prms,trainvalVox,testVox,numTrainVal,permutation,isDebug,varargin)

delay = 1;
lamda = [1e-8,1e-7,1e-6,1e-5,1e-4, 0.001,0.005, 0.01,0.05, 0.1,0.5,1,10,100,500,1000,5000];
numCross = 2;

%NaN Voxels to 0
trainvalVox(isnan(trainvalVox)) = 0;
testVox(isnan(testVox))=0;

%NUm Voxels
numVoxels = size(trainvalVox,1);

if ~isDebug
	numTest = 126; 
	assert(size(testVox,2)==numTest,'testVoxel Number mismatch');
	trainvalFeat = ones(numTrainVal,prms.fisherVecDim);
	testFeat = ones(numTest,prms.fisherVecDim);
	fileStr = '000000';
	%Get train-val features.
	for i=1:1:numTrainVal 
		fName = num2str(permutation(i));
		l = length(fName);
		fName = strcat(fileStr(1:end-l),fName);
		featFileName = fullfile(prms.paths.codes,'trainval',strcat(fName,'.mat'));
		feat = load(featFileName);
		feat = feat.code;
		trainvalFeat(i,:) = feat;
	end
	trainvalFeat(isnan(trainvalFeat)) = 0;

	%Get Test Features
	for i=1:1:numTest 
		fName = num2str(i);
		l = length(fName);
		fName = strcat(fileStr(1:end-l),fName);
		featFileName = fullfile(prms.paths.codes,'test',strcat(fName,'.mat'));
		feat = load(featFileName);
		feat = feat.code;
		testFeat(i,:) = feat;
	end
	[testFeat,blah] = build_features(testFeat,0,0,delay);  
	testFeat(isnan(testFeat)) = 0;
else
	numTest = size(testVox,2);
	trainvalFeat = varargin{1};
	testFeat = varargin{2};
	[testFeat,blah] = build_features(testFeat,0,0,delay);  
	featDim = size(trainvalFeat,2);
end


crossLength = ceil(numTrainVal/numCross);
performance = zeros(numVoxels,length(lamda),numCross);
for c=1:1:numCross
	disp(sprintf('CrossValidation round %d',c));
	valSt = (c-1)*crossLength + 1;
	valEn = min(numTrainVal,valSt + crossLength);

    disp(sprintf('%d,%d',valSt,valEn));
	[trainFeat,valFeat] = build_features(trainvalFeat,valSt,valEn,delay);

	trainImgNum = permutation([1:valSt-1,valEn+1:numTrainVal]);
	valImgNum = permutation(valSt:valEn);
	assert(isempty(intersect(trainImgNum,valImgNum)),'Common Train and val elements');

	for i=1:1:length(lamda)
		disp(sprintf('Value of Lamda: %f', lamda(i)));
	    mat = trainFeat'*inv(single(trainFeat*trainFeat' + lamda(i)*eye(numTrainVal-(valEn-valSt+1))));
	    %mat = inv(single(trainFeat'*trainFeat + lamda(i)*eye(featDim)))*trainFeat';
		y = trainvalVox(:,trainImgNum)';
		gtVox = trainvalVox(:,valImgNum)';
		est = mat*y;
		predVox = valFeat*est;
		%keyboard;
		for v=1:1:numVoxels
			corr = corrcoef(predVox(:,v),gtVox(:,v));
			performance(v,i,c) = corr(1,2);
			%disp(corr(1,2));			
		end
		
		clear mat;

	end	

	clear trainFeat,valFeat;

end
disp('Cross Validation Finished..');

performance = median(performance,3);
[bestVal,bestLamda] = max(performance,[],2);
voxValue = cell(numVoxels,2);

%Use Full Train Val Set to calculate
[featMat,blah] = build_features(trainvalFeat,0,0,delay); 
clear trainvalFeat;

disp('Pre-Computing Matrices for final predictions');
mat = cell(length(lamda),1);
for i=1:1:length(lamda)
	%mat{i} = inv(single(featMat*featMat' + lamda(i)*eye(numTrainVal)));
	%mat{i} = featMat'*inv(single(featMat*featMat' + lamda(i)*eye(numTrainVal)));
	mat{i} = inv(single(featMat'*featMat + lamda(i)*eye(featDim)))*featMat';
end

wMat = cell(numVoxels,1);
for v=1:1:numVoxels
	y = trainvalVox(v,permutation)';
	gtTestVox = testVox(v,:);
	%est = mldivide(mat{bestLamda(v)},y);
	est = mat{bestLamda(v)}*y;
	%est = featMat'*est;

	%disp(size(testFeat));
	%disp(size(est));	
	predVox = testFeat*est;
	voxValue{v,1} = gtTestVox;
	voxValue{v,2} = predVox;
	corr = corrcoef(predVox,gtTestVox);
	corr = corr(1,2);
	wMat{v} = est;
	%disp(corr);
	
end
clear mat;

resStruct = struct();
resStruct.voxValue = voxValue;
resStruct.lamda = bestLamda;
resStruct.trainCorr = bestVal;
resStruct.wMat = wMat;
if isDebug
	varargout{1} = resStruct;
else
	disp('Saving..');
	eval([regionName '=resStruct']);
	if exist(prms.outFileName,'file')
		save(prms.outFileName,regionName,'-append','-v7.3');
	else
		save(prms.outFileName,regionName,'-v7.3');
	end
end
end

function [trainFeat,valFeat] = build_features(featMat,valSt,valEn,delay)

%This is probably not a very good thing to do.
featMat(isnan(featMat)) = 0;
[numEx,featDim] = size(featMat);
assert(delay<=1,'Permutations will not work for delay > 1');

if delay>1
	dFeatMat = ones(numEx,delay*featDim + 1);
	for i =1:1:length(numEx)
		for d=1:1:delay
			st = (d-1)*featDim + 1;
			en = st + featDim - 1;
			if i-d>0
				dFeatMat(i,st:en) = featMat(i-d,:);
			else
				dFeatMat(i,st:en) = 0;	
			end
		end
	end
else
	dFeatMat = featMat;
end

clear featMat;

trainFeat = [dFeatMat(1:valSt-1,:);dFeatMat(valEn+1:end,:)];
if valSt==valEn
	valFeat = [];
else
	valFeat = dFeatMat(valSt:valEn,:);
end

end
 
