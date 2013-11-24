function [] = vis_result_vocSz(encType)
if strcmp(encType,'vq')
	vocSz = [8,16,32,64,128,256,512,1000,2000,3000];
else
	vocSz = [8,16,32,64,128];
end
%vocSz = [0.25,0.50,0.75,1];
%vocSz = [0,1,2,3,4];
paths.dataPath = '/auto/k6/pulkit/data/scene/gallantLabData/';
paths.featDataPath = '/auto/k6/pulkit/data/scene/';
paths.resultPath = fullfile(paths.featDataPath, 'final_results/');

%ROI Information
dataHome = '/auto/k7/dustin/data/MRI/DS/colorNatims/';
load(fullfile(dataHome,'corticalVox.mat'),'cortVox');
roiTmp = load(fullfile(dataHome,'rois.mat'));
roi =  roiTmp.roiVox;
clear roiTmp;

%Get CC from LDA Model
load('/auto/k1/dustin/data7/Analyses/lda/lda3/performanceInfo.mat','performVal');
ccLDA = performVal.DS.ccMean;

lhFieldNames = fieldnames(roi.lh);
rhFieldNames = fieldnames(roi.rh);

ccResAll = cell(length(vocSz),1);
ccResLdaAll = cell(length(vocSz),1);
allNames = {};
for v =1:1:length(vocSz)
	expName = get_expName(encType,vocSz(v),1);
	dataFileName = fullfile(paths.resultPath,strcat(expName,'.mat'));
	results = load(dataFileName);
	roiName = fields(results);
	if v==1
		allNames = roiName;
	else
		allNames = intersect(allNames,roiName);
	end
	ccRes = struct();
	ccResLda = struct();
	for i=1:1:length(roiName)
		name = roiName{i};
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
		
		%My CC Calc
		voxValue = results.(name).voxValue;
		numVox = size(voxValue,1);
		cc = zeros(numVox,1);
		for j=1:1:numVox
			c = corrcoef(voxValue{j,1},voxValue{j,2});
			cc(j) = c(1,2);
		end
		ccRes.(name) = cc;
		ccResLda.(name) = ccLDA(idxAll);
	end
	ccResAll{v} = ccRes;
	ccResLdaAll{v} = ccResLda;	
end

numPlots = ceil(sqrt(length(allNames)));
figure();
title('Yes');
for i=1:1:length(allNames)
	name = allNames{i};
	meanVoc = zeros(length(vocSz),1);
	medVoc = zeros(length(vocSz),1);
	meanVocLda = zeros(length(vocSz),1);
	medVocLda = zeros(length(vocSz),1);
	for v=1:1:length(vocSz)
		ccRes = ccResAll{v};
		ccResLda = ccResLdaAll{v}
		%ccRes
		meanVoc(v) = mean(ccRes.(name));
		medVoc(v) = median(ccRes.(name));
		meanVocLda(v) = mean(ccResLda.(name));
		medVocLda(v) = median(ccResLda.(name));
	end
	subplot(numPlots,numPlots,i);
	hold on;
	
	semilogx(log2(vocSz),meanVoc,'b');
	semilogx(log2(vocSz),meanVocLda,'--b');
	semilogx(log2(vocSz),medVoc,'r');
	semilogx(log2(vocSz),medVocLda,'--r');
	
	%plot(vocSz,meanVocLda,'--b');
	%plot(vocSz,medVoc,'r');
	%plot(vocSz,medVocLda,'--r');
	title(name);	
end

end 
