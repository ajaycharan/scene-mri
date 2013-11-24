%function [distMat] = vis_wcorr(encType,vocSz)

trainPercent = 1;
expName = get_expName(encType,vocSz,trainPercent);
resultPath = fullfile('/auto/k6/pulkit/data/scene/final_results/');

results = load(fullfile(resultPath,strcat(expName,'_ignore_tp1.00.mat')));
regionNames = fields(results);
numCompare = 25;
numRegions = length(regionNames);

distMat = zeros(numCompare*numRegions,numCompare*numRegions);
w = cell(numCompare*numRegions,1);
regionIdx = struct();
for r1=1:1:numRegions
	corr = get_corr(results.(regionNames{r1}).voxValue);
	[~,idx] = sort(-corr);
	idx = idx(1:numCompare);
	regionIdx.(regionNames{r1}) = idx;
	regionOrder{r1} = regionNames{r1}; 
	disp(sprintf('Region: %s, %f, %f',regionNames{r1},corr(idx(1)),corr(idx(end))));
	st = (r1-1)*numCompare + 1;
	en = st + numCompare -1;
	w(st:en) = results.(regionNames{r1}).wMat(idx);
end
for i=1:1:numCompare*numRegions
	for j=1:1:numCompare*numRegions
		c = corrcoef(w{i},w{j});
		distMat(i,j) = c(1,2);
	end
end
%}
[idx,C] = kmeans(distMat,3);
clusterIds = cell(3,1);
for i=1:1:3
	for r=1:1:numRegions
		clusterIds{i}.(regionNames{r}) = [];
	end
end

for i=1:1:3
	cIds = find(idx==i);
	for j=1:1:length(cIds);
		r = ceil((cIds(j)/25));
		v = mod(cIds(j),25);
		if v==0; v = 25;end;
		clusterIds{i}.(regionNames{r}) = [clusterIds{i}.(regionNames{r}) regionIdx.(regionNames{r})(v)];
	end
end

%{
regionMain = 'eba';
corr = get_corr(results.(regionMain).voxValue);
[~,idx] = sort(-corr);
idx = idx(1:numCompare);
wMat = results.(regionMain).wMat(idx);
disp(size(wMat));
figure();
numPlots = ceil(sqrt(length(regionNames)));
for r=1:1:length(regionNames)
	voxData = get_corr(results.(regionNames{r}).voxValue);
	[~,idx] = sort(-corr);
	idx = idx(1:numCompare);
	wRegionMat = results.(regionNames{r}).wMat(idx);
	disp(size(wRegionMat));
	
	corrMat = zeros(numCompare,numCompare);
	for i=1:1:numCompare
		for j=1:1:numCompare
			c = corrcoef(wMat{i},wRegionMat{j});
			corrMat(i,j) = c(1,2);
		end
	end
	%COnvert to 0-1 scale
	corrMat = (corrMat + 1)/2.0;
	subplot(numPlots,numPlots,r);
	imshow(corrMat);
	title(regionNames{r});
	

end
%}
%end
