assert(trainPercent>0 && trainPercent<=1, 'Invalid trainPercent');
order=2;
paths.dataPath = '/auto/k6/pulkit/data/scene/gallantLabData/';
paths.featDataPath = '/auto/k6/pulkit/data/scene/';
paths.resultPath = fullfile(paths.featDataPath, 'final_results/');

%RUn Params
isProfile = true;
allVoxData = false;

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

for r = 1:1:5
	prms.outFileName = fullfile(paths.resultPath,strcat(expName,sprintf('run%d.mat',r-1)));
	resp{r} = load(prms.outFileName);
	if r==1
		regions = fields(resp{r});
	else
		regions = intersect(fields(resp{r}),regions);
	end
end

regions = {regions{1}};
numPlots = ceil(sqrt(length(regions)));
figure();	
for i=1:1:length(regions)
	for r =1:1:5
		regionData = resp{r}.(regions{i});
		voxData = regionData.voxValue;
		numVox = size(voxData,1);
		if r==1
			corrVals = zeros(numVox,5);
		end
		for v=1:1:numVox
			corr = corrcoef(voxData{v,1},voxData{v,2});
			corrVals(v,r) = corr(1,2);
		end
	end	
	subplot(numPlots,numPlots,i);
	hold on;
	color = {'r','g','b','black','y'};
	for r=1:1:5
		plot(1:1:numVox,corrVals(:,r),color{r});
	end
	title(regions{i});
	
end



