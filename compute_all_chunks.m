function [] = compute_all_chunks(prms,featextr, pooler)

% --------------------------------
% Prepare output filenames
% --------------------------------

trainSetStr = [];
for si = 1:length(prms.splits.train)
    trainSetStr = [trainSetStr prms.splits.train{si}]; %#ok<AGROW>
end

testSetStr = [];
for si = 1:length(prms.splits.test)
    testSetStr = [testSetStr prms.splits.test{si}]; %#ok<AGROW>
end

if ~isfield(prms.experiment, 'classif_tag')
    prms.experiment.classif_tag = '';
end

%kChunkIndexFile = fullfile(prms.paths.codes, sprintf('%s_chunkindex.mat', prms.experiment.codes_suffix));
%MyChunkFile
kChunkIndexFile = fullfile(prms.paths.codes, 'chunkindex.mat');

% --------------------------------
% Compute Chunks (for all splits)
% --------------------------------
if exist(kChunkIndexFile,'file')
    load(kChunkIndexFile)
else
    if(prms.isMyFeat)
        chunk_files = my_compChunksIMDB(prms, featextr, pooler);
    else
        chunk_files = featpipem.chunkio.compChunksIMDB(prms, featextr, pooler);
    end
    % save chunk_files to file
    save(kChunkIndexFile, 'chunk_files');
end
end