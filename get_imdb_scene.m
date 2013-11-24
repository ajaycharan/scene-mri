function [out] = get_imdb_scene()

dataPath = '/auto/k6/pulkit/data/scene/gallantLabData/image/'; 

images = struct();
classes = struct();
sets=struct();
sets.TRAIN=1;
sets.VAL=2;
sets.TEST=3;



count = 1;
for cl=1:1:1
  
	clCount = 1
	trainFiles = dir(strcat(dataPath,'train/*.png')); 
    testFiles = dir(strcat(dataPath,'test/*.png'));
    numTrain = ceil(0.6*(length(trainFiles)));
    permutation = (1:1:length(trainFiles));
   
    %Train Set
    for j=1:1:length(trainFiles)
		i = permutation(j);
        images.id(count) = count;
        images.class(count) = cl;
        images.name{count} = strcat('train/',trainFiles(i).name);
        img = imread(fullfile(dataPath,'train/',trainFiles(i).name));
        images.size(1,count) = size(img,1);
        images.size(2,count) = size(img,2);
        
		if(count<=numTrain)
            images.set(count) = sets.TRAIN;
        else
            images.set(count) = sets.VAL;
        end
        count = count + 1;
		clCount = clCount + 1;
    end
    
    %Test Set
    for i=1:1:length(testFiles)
        images.id(count) = count;
        images.class(count) = cl;
        images.name{count} = strcat('test/',testFiles(i).name);
        img = imread(fullfile(dataPath,'test/',testFiles(i).name));
        images.size(1,count) = size(img,1);
        images.size(2,count) = size(img,2);
        
        %{
		imName = clTestFiles{i};
        imName = imName(1:end-3);
        imName = strcat(imName,'mat');
        images.feat_file_name{count} = fullfile(siftFeaturePath,imName);
        %}
       
        images.set(count)=3;
        count = count + 1;
        clCount = clCount + 1;
   
    end

    classes.imageIds{cl} = (count-clCount +1):1:(count-1);
    

end

eval('dir= dataPath;');
outFileName = '/auto/k6/pulkit/data/scene/imdb/imdb_scene.mat';
save(outFileName,'sets','dir','images','classes');
out=1;

end
