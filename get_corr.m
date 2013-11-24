function [corr] = get_corr(voxData)
	numVox = length(voxData);
	corr = zeros(numVox,1);
	for i=1:1:numVox
		c = corrcoef(voxData{i,1},voxData{i,2});
		corr(i) = c(1,2);
	end
end
