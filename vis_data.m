function [] = vis_data(encType,vocSz,varargin)
order=2;
get_paths();

pngFileName = 'results/%s.png';
switch encType
	case 'fisher'
		if vocSz<=256
			poolType='hor0';
		else
			poolType='hor';
		end
		poolType = 'spm';
		%expName = sprintf('LFK_root_imsz480_enc%s_voc%d_pool%s_he025_step3_o%d_ignore_tp1.00',encType,vocSz,poolType,order);
		expName = sprintf('LFK_root_imsz480_enc%s_voc%d_pool%s_he025_step3_o%d_allvox_tp1.00',encType,vocSz,poolType,order);
	case 'vq'
		if vocSz<=256
			poolType='spm';
		else
			poolType='hor';
		end
		poolType = 'spm';
		%expName = sprintf('LFK_root_imsz480_enc%s_voc%d_pool%s_he025_step3_o%d_ignore_tp1.00',encType,vocSz,poolType,order);
		expName = sprintf('LFK_root_imsz480_enc%s_voc%d_pool%s_he025_step3_o%d_allvox_tp1.00',encType,vocSz,poolType,order);
	case 'gabor'
		assert(~isempty(varargin),'3 inputs required for gabor');
	    sfMin = vocSz;
		sfMax = varargin{1};
		expName = sprintf('gabor_sfmn%d_sfmx%d_ignore_tp1.00',sfMin,sfMax);
		clear vocSz;
	case 'decaf'
		layerName = vocSz;
		%expName = sprintf('%s_allvox_tp1.00',layerName);
		expName = sprintf('%s_roi_tp1.00',layerName);
		
end

outFileName = fullfile(paths.resultPath,strcat(expName,'.mat'));
results = load(outFileName);

areaNames = fields(results);
numPlots = ceil(sqrt(length(areaNames)));
fig = figure();
disp(areaNames);
threshSum = 0;
allCorr = cell(length(areaNames),1);
for i=1:1:length(areaNames)
	%disp(i);
	areaData = results.(areaNames{i});
	voxValues = areaData.voxValue;
	numVox = size(voxValues,1);
	%disp(sprintf('numVox: %d',numVox));
	corrVals = zeros(numVox,1);
	for v=1:1:numVox
		corr = corrcoef(voxValues{v,1},voxValues{v,2});
		corrVals(v) = corr(1,2);
	end
	threshSum = threshSum + sum(corrVals>=0.34);
	%disp(size(corrVals));
	corrVals(isnan(corrVals))=0;
	med = median(corrVals);
	mn = mean(corrVals);
	allCorr{i} = corrVals;
	
	disp(sprintf('%s_%0.2f_%0.2f',areaNames{i},med,mn));
	subplot(numPlots,numPlots,i);
	xCenters = linspace(-1,1,50);
	hist(corrVals,xCenters);
	title(sprintf('%s_%0.2f_%0.2f',areaNames{i},med,mn));
	%}
end
allCorr = cat(1,allCorr{:});
mn = mean(allCorr);
md = median(allCorr);
bestVox = allCorr >= 0.34;
allCorr = allCorr(bestVox);
bMn = mean(allCorr);
bMd = median(allCorr);
disp(sprintf('Mean: %f, Median: %f, threshSum: %d',mn,md,threshSum));		
disp(sprintf('Abov thresh vox: Mean: %f, Median: %f ',bMn,bMd));		

outFile = sprintf(pngFileName,expName);
disp(outFile);
print('-dpng','-r400',outFile);
end
