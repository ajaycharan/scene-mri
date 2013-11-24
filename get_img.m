dataPath = '/auto/k6/pulkit/data/scene/gallantLabData/'
imDataPath = strcat(dataPath,'image/')

%Train File Names
trainDataPath = strcat(dataPath,'training/');
testDataPath = strcat(dataPath,'validation/');

%{
fileNames = dir(trainDataPath);
for i =3:1:length(fileNames)
	name = fileNames(i).name;
	imName = strcat(imDataPath, 'train/',name(1:end-4), '.png');
	disp(imName);
	data = load(strcat(trainDataPath,name));
	imwrite(data.image,imName);
end
%}

fileNames = dir(strcat(testDataPath,'*.mat'));
for i =1:1:length(fileNames)
	name = fileNames(i).name;
	imName = strcat(imDataPath, 'test/',name(1:end-4), '.png');
	disp(imName);
	data = load(strcat(testDataPath,name));
	imwrite(data.image,imName);
end
