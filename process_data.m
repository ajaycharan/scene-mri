function [varargout] = process_data(encType,layerName,trainPercent,varargin) 

assert(trainPercent>0 && trainPercent<=1, 'Invalid trainPercent');
pathPrefix = '/work4/pulkitag/projMri/';
paths.dataPath  = fullfile(pathPrefix,'gallantLabData/');
paths.decafFeat = '/work4/pulkitag/decafFeat/mri/';
paths.resultPath = fullfile(pathPrefix, 'exp/results/');

if ~exist(paths.resultPath)
	system(['mkdir -p ' paths.resultPath]);
end

%RUn Params
isProfile = false;
allVoxData = true;
isUnitTest = false;

if (isProfile)
	profile on;
end

switch encType
	case 'decaf'
		expName = layerName; 
end
%Load the voxel data
dataHome = pathPrefix;
load(fullfile(dataHome,'corticalVox.mat'),'cortVox');
load(fullfile(dataHome,'responses.mat'),'r');
trainvalVoxel = single(r.trn);
testVoxel = single(r.val);
clear r;

%Load the ROI Information
roiTmp = load(fullfile(dataHome,'rois.mat'));
roi =  roiTmp.roiVox;
clear roiTmp;


%Imdb File
imdbFile = fullfile(paths.decafFeat,'imdb.mat');
imdb = load(imdbFile);

%Parameters
prms = struct();
prms.layerName = layerName;
switch layerName
	case 'l1'
		prms.featVecDim = 27*27*96;
	case 'l2'
		prms.featVecDim = 27*27*256;
	case 'l3'
		prms.featVecDim = 13*13*384;
	case 'l4'
		prms.featVecDim = 13*13*384;
	case 'l5'
		prms.featVecDim = 6*6*256;
	case 'l6'
		prms.featVecDim = 4096;
	case 'l7'
		prms.featVecDim = 4096;
	case 'l8'
		prms.featVecDim = 4096;
end
prms.paths = paths;

if isProfile
	runNum = varargin{1};
	prms.outFileName = fullfile(paths.resultPath,strcat(expName,sprintf('run%d.mat',runNum)));
	s = RandStream('mcg16807','Seed',runNum);
	RandStream.setGlobalStream(s);
else
	if allVoxData
		prms.outFileName = fullfile(paths.resultPath,strcat(expName,sprintf('_allvox_tp%.02f.mat',trainPercent)));
	else
		prms.outFileName = fullfile(paths.resultPath,strcat(expName,sprintf('_roi_tp%.02f.mat',trainPercent)));
	end
end

disp(prms.outFileName);

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
	nDims = 100;
	nVox = 300;
	nTest = 150;

	
	fakeData = randn(numTrainVal,nDims);
	normData = sum(fakeData.*fakeData,2);
	fakeData = bsxfun(@rdivide,fakeData,normData);
	
	fakeTest = randn(nTest,nDims);
	normTest = sum(fakeTest.*fakeTest,2);
	fakeTest = bsxfun(@rdivide,fakeTest,normTest);	

	fakeW = randn(nDims,nVox);
	fakeVoxel = (fakeData*fakeW)' + 0.2*randn(nVox,numTrainVal);
	fakeTestVoxel = (fakeTest*fakeW)';
	resStruct = learn_params('fake',imdb,prms,fakeVoxel,fakeTestVoxel,numTrainVal,permutation,true,fakeData,fakeTest);
	corr = get_corr(resStruct.voxValue);
	disp(sprintf('NUmber of voxels are %d',length(corr)));
	disp(sprintf('Mean Correlation: %.2f, Median Corr: %.2f, Min Corr: %.2f, Max: %.2f',mean(corr),median(corr),max(corr),min(corr)));
	%err = norm(fakeW-resStruct.wMat);
	%disp(sprintf('Error in estimation is %f',err));
	varargout{1} = resStruct;
	varargout{2} = fakeW;	
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
			learn_params(name,imdb,prms,trainvalResponse,testResponse,numTrainVal,permutation,false);
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
lamda = [ 0.001,0.005, 0.01,0.05, 0.1,0.5,1,10,100,500,1000,5000];
numCross = 5;

%NaN Voxels to 0
trainvalVox(isnan(trainvalVox)) = 0;
testVox(isnan(testVox))=0;

%NUm Voxels
numVoxels = size(trainvalVox,1);

if ~isDebug
	numTest = 126; 
	assert(size(testVox,2)==numTest,'testVoxel Number mismatch');
	trainvalFeat = ones(numTrainVal,prms.featVecDim);
	testFeat = ones(numTest,prms.featVecDim);
	fileStr = '000000';
	%Get train-val features.
	for i=1:1:numTrainVal 
		fName = num2str(permutation(i));
		l = length(fName);
		fName = strcat(fileStr(1:end-l),fName);
		featFileName = fullfile(prms.paths.decafFeat,prms.layerName,'train',strcat(fName,'.mat'));
		feat = load(featFileName);
		feat = squeeze(feat.feat);
		l1Norm = sum(abs(feat(:)));
		if l1Norm > 0
			feat = feat/l1Norm;
		end	
		trainvalFeat(i,:) = feat(:);
	end
	trainvalFeat(isnan(trainvalFeat)) = 0;

	%Get Test Features
	for i=1:1:numTest 
		fName = num2str(i);
		l = length(fName);
		fName = strcat(fileStr(1:end-l),fName);
		featFileName = fullfile(prms.paths.decafFeat,prms.layerName,'test',strcat(fName,'.mat'));
		feat = load(featFileName);
		feat = squeeze(feat.feat);
		l1Norm = sum(abs(feat(:)));
		if l1Norm > 0
			feat = feat/l1Norm;
		end	
		testFeat(i,:) = feat(:);
	end
	[testFeat,blah] = build_features(testFeat,0,0,delay);  
	testFeat(isnan(testFeat)) = 0;
else
	numTest = size(testVox,2);
	someFeat = varargin{1};
	trainvalFeat = someFeat(permutation,:);
	testFeat = varargin{2};
	[testFeat,blah] = build_features(testFeat,0,0,delay);  
	featDim = size(trainvalFeat,1);
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
	mat{i} = featMat'*inv(single(featMat*featMat' + lamda(i)*eye(numTrainVal)));
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
 
