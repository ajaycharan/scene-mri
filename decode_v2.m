function [rankAcc] = decode_v2(encType,vocSz,varargin);

order=2;
paths.dataPath = '/auto/k6/pulkit/data/scene/gallantLabData/';
paths.featDataPath = '/auto/k6/pulkit/data/scene/';
paths.resultPath = fullfile(paths.featDataPath, 'final_results/');

isProfile = false;
switch encType
	case 'fisher'
		if vocSz<=128
			poolType='spm';
		else
			poolType='hor';
		end
		poolType = 'spm'
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

imdbFile = fullfile(paths.featDataPath,'imdb','imdb_scene.mat');
prmsFile = fullfile(paths.featDataPath,'prms',strcat(expName,'_prms.mat'))

%Imdb File
imdb = load(imdbFile);

%exp prms file
prms = load(prmsFile);
prms = prms.prms;

%Use all voxel Data File to get correlations of voxels in the training set.
trainPercent = 1;
allVoxel = load(fullfile(paths.resultPath,strcat(expName,sprintf('_allvox_tp%.02f.mat',trainPercent))));
str = 'all%d';
corr = zeros(19796,1,'single');
disp(allVoxel.all1);
for i=1:1:length(fields(allVoxel))
	st = (i-1)*1000 + 1;
	en = min(19796,st + 1000 - 1);
	corr(st:en) = single(allVoxel.(sprintf(str,st)).trainCorr);
end
clear allVoxel;


%Set NaN valued voxels to 0
trainvalVoxel(isnan(trainvalVoxel))=0;
testVoxel(isnan(testVoxel))=0;
nDims = -1;
rankAcc =learn_params(imdb,prms,trainvalVoxel,testVoxel,nDims,corr);
end

function [rankAcc] = learn_params(imdb,prms,alltrainvalVox,testVox,nDims,trainCorr)
delay = 1;
lamda = [ 0.001,0.005, 0.01,0.05, 0.1,0.5,1,10,100,500,1000,5000];
corrCross = 0:0.05:0.35;
numCross = 5;

numTrainVal = 1260;
permutation = 1:1:numTrainVal;

numTest = 126; 
assert(size(testVox,2)==numTest,'testVoxel Number mismatch');
alltrainvalVox = alltrainvalVox';
testVox = testVox';

alltrainvalVox(isnan(alltrainvalVox))=0;
testVox(isnan(testVox))=0;

fileStr = '000000';
%Get train-val features.
trainvalFeat = ones(numTrainVal,prms.fisherVecDim);
for i=1:1:numTrainVal 
	fName = num2str(permutation(i));
	l = length(fName);
	fName = strcat(fileStr(1:end-l),fName);
	featFileName = fullfile(prms.paths.codes,'trainval',strcat(fName,'.mat'));
	feat = load(featFileName);
	feat = feat.code;
	trainvalFeat(i,:) = feat;
end

%Get Test Features
testFeat = ones(numTest,prms.fisherVecDim);
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


%Modification which splits the testSet into 2 and uses that for decoding - ignores trainset.
%%%%%%
numTest = 63;
numTrainVal = 63;
alltrainvalVox = testVox(1:63,:);
testVox = testVox(64:end,:);
trainvalFeat = testFeat(1:63,:);
testFeat = testFeat(64:end,:);
permutation = 1:1:63;
%%%%%%


if nDims==-1
	nDims = size(trainvalFeat,2);
end

crossLength = ceil(numTrainVal/numCross);

allBestVal = cell(length(corrCross),1);
allBestLamda = cell(length(corrCross),1);
identifyAcc = zeros(length(corrCross),numCross);
for j=1:1:length(corrCross)
	mask = trainCorr >=corrCross(j);
	trainvalVox = alltrainvalVox(:,mask);
	performance = zeros(nDims,length(lamda),numCross);
	for c=1:1:numCross
		disp(sprintf('CrossValidation round %d',c));
		valSt = (c-1)*crossLength + 1;
		valEn = min(numTrainVal,valSt + crossLength);

		%Nt: trainvalFeat is already properly permuted.	
		[trainFeat,valFeat] = build_features(trainvalFeat,valSt,valEn,delay);

		trainImgNum = permutation([1:valSt-1,valEn+1:numTrainVal]);
		valImgNum = permutation(valSt:valEn);
		assert(isempty(intersect(trainImgNum,valImgNum)),'Common Train and val elements');
		for i=1:1:length(lamda)
			trainVoxFeat = trainvalVox(trainImgNum,:);
			valVoxFeat = trainvalVox(valImgNum,:);
			disp(sprintf('Value of Lamda: %f', lamda(i)));
			mat = trainVoxFeat'*inv(single(trainVoxFeat*trainVoxFeat' + lamda(i)*eye(numTrainVal-(valEn-valSt+1))));
			est = mat*trainFeat;
			predFeat = valVoxFeat*est;

			%We'll have a different lambda for different dimensions.
			for d=1:1:nDims
				corr = corrcoef(predFeat(:,d),valFeat(:,d));
				performance(d,i,c) = corr(1,2);
			end
			clear mat;
		end	
	end
	performance = median(performance,3);
	[bestVal,bestLamda] = max(performance,[],2);


	%Now compute decoding accuracy using cross-validation
	for c=1:1:numCross
		valSt = (c-1)*crossLength + 1;
		valEn = min(numTrainVal,valSt + crossLength);
		[trainFeat,valFeat] = build_features(trainvalFeat,valSt,valEn,delay);
		
		trainImgNum = permutation([1:valSt-1,valEn+1:numTrainVal]);
		valImgNum = permutation(valSt:valEn);
		assert(isempty(intersect(trainImgNum,valImgNum)),'Common Train and val');

		[acc,~] = identification_performance(trainvalVox(trainImgNum,:),trainFeat,...
							  trainvalVox(valImgNum,:),valFeat,lamda,bestLamda,nDims)
		
		identifyAcc(j,c) = acc;
	end
	disp(sprintf('Acccuracy for corr: %f is %f',corrCross(j),mean(identifyAcc(j,:))));
	allBestVal{j} = bestVal;
	allBestLamda{j} = bestLamda;

end

identifyAcc = mean(identifyAcc,2);
[acc,bestIdx] = max(identifyAcc,[],1);
disp(sprintf('Best Acc on val: %f, for corr: %f',acc,corrCross(bestIdx)));
bestLamda = allBestLamda{bestIdx};
mask = trainCorr >=corrCross(bestIdx);
testVox = testVox(:,mask);

mat = cell(length(lamda),1);
for i=1:1:length(lamda)
	mat{i} = single(trainvalVox'*inv(single(trainvalVox*trainvalVox' + lamda(i)*eye(numTrainVal))));
end

disp('Predicting Feature Vector');
predFeat = zeros(size(testFeat),'single');
for d=1:1:nDims
	est = mat{bestLamda(d)}*trainvalFeat(:,d);
	predFeat(:,d) = testVox*est;
end


disp('Decoding');

testLabels = zeros(numTest,1);
testScores = zeros(numTest,numTest);
for u=1:1:numTest
	corrMax = -inf;
	maxIdx = 0;
	for v=1:1:numTest
		c = corrcoef(predFeat(u,:),testFeat(v,:));
		testScores(u,v) = c(1,2);
		if c(1,2)>=corrMax
			corrMax = c(1,2);
			maxIdx = v;
		end
	end 
	testLabels(u) = maxIdx;
	%disp(maxIdx);	
end
assert(sum(testLabels==0)==0,'A test label cannot be equal to 0');
acc = sum(testLabels==(1:1:numTest)')/numTest
disp(acc);
disp('Indices by rank....');
[~,sortIdx] = sort(-testScores,2);
position = zeros(numTest,1);
rankAcc = zeros(numTest,1);
for i=1:1:numTest
	position(i) = find(sortIdx(i,:)==i);
end
for i=1:1:numTest
	rankAcc(i) = sum(position<=i)/numTest;
	disp(rankAcc(i));
end
end

function [trainFeat,valFeat] = build_features(featMat,valSt,valEn,delay)

%This is probably not a very good thing to do.
featMat(isnan(featMat)) = 0;
[numEx,featDim] = size(featMat);
assert(delay<=1,'Permutations will not work for delay > 1');
dFeatMat = featMat;
clear featMat;

trainFeat = [dFeatMat(1:valSt-1,:);dFeatMat(valEn+1:end,:)];
if valSt==valEn
	valFeat = [];
else
	valFeat = dFeatMat(valSt:valEn,:);
end

end



function [acc,testLabels] = identification_performance(trainVox,trainFeat,testVox,testFeat,lamda,bestLamda,nDims)

mat = cell(length(lamda),1);
numTrain = size(trainVox,1);
numTest = size(testVox,1);

for i=1:1:length(lamda)
	mat{i} = single(trainVox'*inv(single(trainVox*trainVox' + ...
										 lamda(i)*eye(numTrain))));
end

predFeat = zeros(size(testFeat),'single');
for d=1:1:nDims
	est = mat{bestLamda(d)}*trainFeat(:,d);
	predFeat(:,d) = testVox*est;
end

testLabels = zeros(numTest,1);
testScores = zeros(numTest,numTest);
for u=1:1:numTest
	corrMax = -inf;
	maxIdx = 0;
	for v=1:1:numTest
		c = corrcoef(predFeat(u,:),testFeat(v,:));
		testScores(u,v) = c(1,2);
		if c(1,2)>=corrMax
			corrMax = c(1,2);
			maxIdx = v;
		end
	end 
	testLabels(u) = maxIdx;
	%disp(maxIdx);	
end
assert(sum(testLabels==0)==0,'A test label cannot be equal to 0');
acc = sum(testLabels==(1:1:numTest)')/numTest


end 
