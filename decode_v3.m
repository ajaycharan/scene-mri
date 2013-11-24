function [rankAcc] = decode_v3(encType,vocSz,varargin);

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
wts = cell(length(fields(allVoxel)),1);
disp(allVoxel.all1);
for i=1:1:length(fields(allVoxel))
	st = (i-1)*1000 + 1;
	en = min(19796,st + 1000 - 1);
	corr(st:en) = single(get_corr(allVoxel.(sprintf(str,st)).voxValue));
	wts{i} = single(cat(2,allVoxel.(sprintf(str,st)).wMat{:}));
end
wts = cat(2,wts{:});
clear allVoxel;


testVoxel(isnan(testVoxel))=0;
nDims = -1;
numRepeats = 100;
muAcc = zeros(numRepeats,1);
testNull = true;
for i=1:1:numRepeats
	muAcc(i) =learn_params(imdb,prms,testVoxel,corr,nDims,wts,testNull);
end
mu = mean(muAcc);
sd = std(muAcc);
md = median(muAcc);
rankAcc = mu;
if testNull
	disp('Results for testing Null hypothesis');
end
disp(sprintf('Accuracy: Mean: %f, median: %f, std: %f',mu,md,sd));

end

function [acc] = learn_params(imdb,prms,alltestVox,testCorr,nDims,wts,testNull)
delay = 1;
corrCross = 0:0.05:0.35;

numTest = 126; 
assert(size(alltestVox,2)==numTest,'testVoxel Number mismatch');
alltestVox = alltestVox';  %numEx*numVox
alltestVox(isnan(alltestVox))=0;

fileStr = '000000';
%Get Test Features
alltestFeat = ones(numTest,prms.fisherVecDim);
for i=1:1:numTest 
	fName = num2str(i);
	l = length(fName);
	fName = strcat(fileStr(1:end-l),fName);
	featFileName = fullfile(prms.paths.codes,'test',strcat(fName,'.mat'));
	feat = load(featFileName);
	feat = feat.code;
	alltestFeat(i,:) = feat;
end
[alltestFeat,blah] = build_features(alltestFeat,0,0,delay);  

if nDims==-1
	nDims = size(alltestFeat,2);
end


%Randomly select 63 test images for finding the mask
numVal = 63;
perm = randperm(numTest);
valIdx = perm(1:numVal);
testIdx = perm(numVal+1:end);

valFeat = alltestFeat(valIdx,:);

disp('Decoding');
accArray = zeros(length(corrCross),1);
for i=1:1:length(corrCross)
	mask = testCorr>=corrCross(i);
	valVox = alltestVox(valIdx,mask);

	predVox = valFeat*wts(:,mask);  %wts should by featDin*voxDim
	[acc,~] = get_labels(valVox,predVox);
	accArray(i) = acc;
	disp(sprintf('Acc for corr: %f is %f \n',corrCross(i),acc));
end

[maxAcc,maxIdx] = max(accArray);
mask = testCorr>=corrCross(maxIdx);
disp(sprintf('Max accuracy of %f found for corr: %f \n',maxAcc,corrCross(maxIdx)));

testVox = alltestVox(testIdx,mask);
testFeat = alltestFeat(testIdx,:);
predVox = testFeat*wts(:,mask);
if testNull
	valVox = alltestVox(valIdx,mask);
	[acc,~] = get_labels(valVox,predVox);
else
	[acc,~] = get_labels(testVox,predVox);
end

disp(sprintf('Final Accuracy is %f \n',acc));

%{
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
%}
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



function [acc,testLabels] = get_labels(gtVox,predVox)

numTest = size(gtVox,1);
testLabels = zeros(numTest,1);
testScores = zeros(numTest,numTest);
for u=1:1:numTest
	corrMax = -inf;
	maxIdx = 0;
	for v=1:1:numTest
		c = corrcoef(predVox(u,:),gtVox(v,:));
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
acc = sum(testLabels==(1:1:numTest)')/numTest;
end 
