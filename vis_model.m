encType = 'fisher';
vocSz = 8;
isAvgPlot = false;
plotSingle = true; %dave the single best and worst image

expName = get_expName(encType,vocSz);
modelDataPath = '/auto/k6/pulkit/data/scene/';
sunPath = '/auto/k6/pulkit/data/scene/sun/experiment/';

%Model DATA
modelStr =  fullfile(modelDataPath,'%s',strcat(expName,'_ignore_tp1.00'));
modelFileName = strcat(sprintf(modelStr,'final_results'),'.mat');
disp(modelFileName);
model = load(modelFileName);
regionNames = fields(model);

scorePath = sprintf(modelStr,'region_scores');
singleImPath = sprintf(modelStr,'region_scores/single_best');
if (~exist(scorePath,'dir'))
	system(['mkdir ' scorePath]);
end

imdb = load(fullfile(sunPath,'imdb/imdb_1.mat'));
trainIds = find(imdb.images.set==1);
valIds = find(imdb.images.set==2);
testIds = find(imdb.images.set==3);

numTrain = length(trainIds);
numVal = length(valIds);
numTest = length(testIds);

chunkindex = load(fullfile(sunPath,'codes',expName,'chunkindex.mat'));
%numTrainFiles = 120;
%numValFiles = 8;
numTrainFiles = length(chunkindex.chunk_files('train'));
numValFiles = length(chunkindex.chunk_files('val'));
numTestFiles = length(chunkindex.chunk_files('test'));

regionNames = {'eba'};
for r=1:1:length(regionNames)
	regionW = model.(regionNames{r}).wMat;
	regionFile = fullfile(scorePath,strcat(regionNames{r},'.mat'));
	singleImFolder = fullfile(singleImPath,regionNames{r});
	if ~exist(singleImFolder)
		system(['mkdir -p ' singleImFolder]);
	end

	regionFigFile = fullfile(scorePath,strcat(regionNames{r},'.fig'));
	disp(regionNames{r});
	numVoxel = length(regionW);
	scores = zeros(numTrain+numVal+numTest,numVoxel);
	ids = zeros(numTrain+numVal+numTest,1);
	st = 1;
	for i=1:1:numTrainFiles
		fileStr = (i-1)*100 + 1;
		fileName = fullfile(sunPath,'codes',expName,strcat(sprintf('train_chunk%d',fileStr),'.mat'));
		featMat = load(fileName);
		featMat = (featMat.chunk)';
	    en = min(st+100-1,numTrain);
	    assert(en-st+1==size(featMat,1));
		for v=1:1:numVoxel
			scores(st:en,v) = featMat*regionW{v}; 
		end
		ids(st:en) = trainIds(st:en);
		st = st + (en - st) + 1;
	end

	for i=1:1:numValFiles
		fileStr = (i-1)*100 + 1;
		fileName = fullfile(sunPath,'codes',expName,strcat(sprintf('val_chunk%d',fileStr),'.mat'));
		featMat = load(fileName);
		featMat = (featMat.chunk)';
	    en = min(st+100-1,numTrain+numVal);
	    assert(en-st+1==size(featMat,1));
		for v=1:1:numVoxel
			scores(st:en,v) = featMat*regionW{v}; 
		end
		ids(st:en) = valIds(st-numTrain:en-numTrain);
		st = st + (en - st) + 1;
	end

	for i=1:1:numTestFiles
		fileStr = (i-1)*100 + 1;
		fileName = fullfile(sunPath,'codes',expName,strcat(sprintf('test_chunk%d',fileStr),'.mat'));
		featMat = load(fileName);
		featMat = (featMat.chunk)';
	    en = min(st+100-1,numTrain+numVal+numTest);
	    assert(en-st+1==size(featMat,1));
		for v=1:1:numVoxel
			scores(st:en,v) = featMat*regionW{v}; 
		end
		ids(st:en) = testIds(st-(numTrain+numVal):en-(numTrain+numVal));
		st = st + (en - st) + 1;
	end

	assert(all(ids>0),'Ids cannot be zero');
	assert(length(unique(ids))==length(ids),'Ids have to be unique');

	disp('Plotting');
	topNum = 50;
	bottomNum = 50;
	numBestVoxels = 25;
	fig = figure();
	imCount = 1;

	corr = get_corr(model.(regionNames{r}).voxValue);
	[~,corrIdx] = sort(-corr);
	corrIdx = corrIdx(1:numBestVoxels);

	if isAvgPlot	
		%Average scores across the top 25 best predicted voxels.
		topRegionScores = sum(scores(:,corrIdx),2);
		disp(size(topRegionScores));	
		[~,sortIdx] = sort(-topRegionScores);
		sortIds = ids(sortIdx);

		imCount = 1;
		for n=1:1:topNum
			im = imread(fullfile(imdb.dir,imdb.images.name{sortIds(n)}));
			subplot(10,10,imCount);
			imshow(im);	
			title(sprintf('%0.4f',topRegionScores(sortIdx(n))));	
			imCount = imCount + 1;
		end
		for n=length(sortIds)-bottomNum+1:length(sortIds)
			im = imread(fullfile(imdb.dir,imdb.images.name{sortIds(n)}));
			subplot(10,10,imCount);
			imshow(im);	
			title(sprintf('%0.4f',topRegionScores(sortIdx(n))));	
			imCount = imCount + 1;
		end
		hgsave(fig,regionFigFile);	
		close all;
		save(regionFile,'scores','ids','-v7.3')
	else
		if plotSingle
			numBestVoxels = 6;
			posImFile = fullfile(singleImFolder,'pos%d_%d.png');
			negImFile = fullfile(singleImFolder,'neg%d_%d.png');
			for v=1:1:numBestVoxels
				fig = figure();
				imCount = 1;
				voxelFigFile = fullfile(scorePath,sprintf('%s_%d%s',regionNames{r},v,'.fig'));
				voxelImFile = fullfile(scorePath,sprintf('%s_%d%s',regionNames{r},v,'.png'));
				voxScore = scores(:,corrIdx(v));
				[~,sortIdx] = sort(-voxScore);
				sortIds = ids(sortIdx);
				
				%Read Best Image
				for n=1:1:10
					im = imread(fullfile(imdb.dir,imdb.images.name{sortIds(n)}));
					imwrite(im,sprintf(posImFile,v,n));
				end

				for n=1:1:10
					%Read Worst Image
					im = imread(fullfile(imdb.dir,imdb.images.name{sortIds(end-n+1)}));
					imwrite(im,sprintf(negImFile,v,n));
				end				

			end

		else
			for v=1:1:numBestVoxels
				fig = figure();
				imCount = 1;
				voxelFigFile = fullfile(scorePath,sprintf('%s_%d%s',regionNames{r},v,'.fig'));
				voxelImFile = fullfile(scorePath,sprintf('%s_%d%s',regionNames{r},v,'.png'));
				voxScore = scores(:,corrIdx(v));
				[~,sortIdx] = sort(-voxScore);
				sortIds = ids(sortIdx);
				for n=1:1:topNum
					im = imread(fullfile(imdb.dir,imdb.images.name{sortIds(n)}));
					subplot(10,10,imCount);
					imshow(im);	
					title(sprintf('%0.4f',voxScore(sortIdx(n))));	
					imCount = imCount + 1;
				end
				for n=length(sortIds)-bottomNum+1:length(sortIds)
					im = imread(fullfile(imdb.dir,imdb.images.name{sortIds(n)}));
					subplot(10,10,imCount);
					imshow(im);	
					title(sprintf('%0.4f',voxScore(sortIdx(n))));	
					imCount = imCount + 1;
				end	
				%hgsave(fig,voxelFigFile);	
				print('-dpng','-r400',voxelImFile);
				close all;
			end
			save(regionFile,'scores','ids','-v7.3')

		end
	end
end
