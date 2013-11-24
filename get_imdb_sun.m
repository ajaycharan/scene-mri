function [out] = get_imdb_sun()

fileNum = 1;
dataPath = '/auto/k6/pulkit/data/scene/sun/SUN397/'; 
dataInfoPath =  '/auto/k6/pulkit/data/scene/sun/';  


images = struct();
classes = struct();
sets=struct();
sets.TRAIN=1;
sets.VAL=2;
sets.TEST=3;

classNameFile = fullfile(dataInfoPath,'ClassName.txt');
classFolders = textread(classNameFile,'/%s','delimiter','\n');

classPath = strcat(dataPath,'%s/');

%Get the train and test names.
trainFileName = fullfile(dataInfoPath,sprintf('Training_%02d.txt',fileNum));
disp(trainFileName);
testFileName = fullfile(dataInfoPath,sprintf('Testing_%02d.txt',fileNum));

trainNames = textread(trainFileName,'/%s','delimiter','\n');
testNames = textread(testFileName,'/%s','delimiter','\n');


assert(length(classFolders)==397,'There should be 397 class folders'); 

count = 1;
for cl=1:1:length(classFolders)
    disp(sprintf('Processing Class %d',cl));
    imgFiles = dir(sprintf(classPath,classFolders{cl}));
    imgFiles = imgFiles(3:end);
    classes.name{cl} =  classFolders{cl};
%     if(exist(createPath,'dir')~=7)
%         system(['mkdir ' createPath]);
%     end

    clCount = 1;
    clFileNames = cell(length(imgFiles),1);
    for i=1:1:length(imgFiles)
        clFileNames{i} = fullfile(classes.name{cl},imgFiles(i).name);
    end
    [fil1,ia,ib] = intersect(trainNames,clFileNames);
    clTrainFiles = trainNames(sort(ia));
    [clTestFiles,ia,ib] =  intersect(testNames,clFileNames);
    assert(isempty(intersect(clTrainFiles,clTestFiles)),'Train and Test Files should not intersect');
    
    numTrain = ceil(0.6*(length(clTrainFiles)));
         
   
    %Train Set
    for i=1:1:length(clTrainFiles)
        img = imread(fullfile(dataPath,clTrainFiles{i}));
		if (ndims(img)>3)
			continue;
		end
        images.id(count) = count;
        images.class(count) = cl;
        images.name{count} = clTrainFiles{i};
        images.size(1,count) = size(img,1);
        images.size(2,count) = size(img,2);
        
		if(clCount<=numTrain)
            images.set(count)=1;
        else
            images.set(count)=2;
        end
        count = count + 1;
        clCount = clCount + 1;
   
    end
    
    %Test Set
    for i=1:1:length(clTestFiles)
        img = imread(fullfile(dataPath,clTestFiles{i}));
		if (ndims(img)>3)
			continue;
		end
		images.id(count) = count;
        images.class(count) = cl;
        images.name{count} = clTestFiles{i};
        images.size(1,count) = size(img,1);
        images.size(2,count) = size(img,2);
        
        images.set(count)=3;
        count = count + 1;
        clCount = clCount + 1;
   
    end

    classes.imageIds{cl} = (count-clCount +1):1:(count-1);
    

end

eval('dir= dataPath;');
%save('imdb_caltech_ver1.mat','sets','dir','images','classes');
outFileName = sprintf( '/auto/k6/pulkit/data/scene/sun/experiment/imdb/imdb_%d.mat',fileNum);
save(outFileName,'sets','dir','images','classes');
out=1;

end
