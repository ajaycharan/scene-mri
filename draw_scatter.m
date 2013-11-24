function [] = draw_scatter(encType,vocSz,varargin)
order=2;
paths.dataPath = '/auto/k6/pulkit/data/scene/gallantLabData/';
paths.featDataPath = '/auto/k6/pulkit/data/scene/';
paths.resultPath = fullfile(paths.featDataPath, 'final_results/');

expName = get_expName(encType,vocSz,varargin{:});

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
	%disp(sprintf('%s_%0.2f_%0.2f',areaNames{i},med,mn));
	%subplot(numPlots,numPlots,i);
	%xCenters = linspace(-1,1,50);
	%hist(corrVals,xCenters);
	%title(sprintf('%s_%0.2f_%0.2f',areaNames{i},med,mn));
	
end

