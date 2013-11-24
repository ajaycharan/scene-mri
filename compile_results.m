function [] = compile_results(encType,vocSz,varargin)
order=2;
paths.dataPath = '/auto/k6/pulkit/data/scene/gallantLabData/';
paths.featDataPath = '/auto/k6/pulkit/data/scene/';
paths.resultPath = fullfile(paths.featDataPath, 'final_results/');

switch encType
	case 'fisher'
		if vocSz<=256
			poolType='spm';
		else
			poolType='hor';
		end
		poolType = 'spm';
		expName = sprintf('LFK_root_imsz480_enc%s_voc%d_pool%s_he025_step3_o%d_ignore_tp1.00',encType,vocSz,poolType,order);
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

dataFileName = fullfile(paths.resultPath,strcat(expName,'.mat'));
results = load(dataFileName);

%Load Dusting results
dataHome = '/auto/k7/dustin/data/MRI/DS/colorNatims/';
load(fullfile(dataHome,'corticalVox.mat'),'cortVox');
roiTmp = load(fullfile(dataHome,'rois.mat'));
roi =  roiTmp.roiVox;
clear roiTmp;

%Get CC
load('/auto/k1/dustin/data7/Analyses/lda/lda3/performanceInfo.mat','performVal');
ccLDA = performVal.DS.ccMean;


lhFieldNames = fieldnames(roi.lh);
rhFieldNames = fieldnames(roi.rh);
roiName = fields(results);

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
	
	ccLDARoi = ccLDA(idxAll);

	%My CC Calc
	voxValue = results.(name).voxValue;
	numVox = size(voxValue,1);
	cc = zeros(numVox,1);
	for j=1:1:numVox
		c = corrcoef(voxValue{j,1},voxValue{j,2});
		cc(j) = c(1,2);
	end

	%disp(size(cc));	
	%disp(size(ccLDARoi));
	disp(sprintf('Region: %s, (My,DS): Mean: (%0.3f,%0.3f), Med: (%0.3f,%0.3f), Max: (%0.3f,%0.3f)', name,mean(cc),mean(ccLDARoi),median(cc),median(ccLDARoi),max(cc),max(ccLDARoi)));

end 



end
