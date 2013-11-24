import sys
from subprocess import call
import process_data_jobprms as const #Specify all the constants
#import features_jobprms as const #Specify all the constants
import pickle

sourceFileName = '/auto/k1/pulkit/.bashrc'

#Store the generated scripts in script path
scriptPath = const.scriptPath
logPath = scriptPath

#The location where function to be called is present
codePath = const.codePath

#Name of the job
jobname = const.jobName 

if const.codeLanguage=='python':
	runcmd = 'python -c "%s"\n'
	codeStr = 'import sys; import ' + moduleName + ';'+ 'sys.path.append(\'%s\');'
	codeStr = codeStr % codePath
	codeStr = codeStr +moduleName+'.'+functionName
elif const.codeLanguage=='matlab':
	matlabStr = '/auto/k2/share/matlab/matlab80/bin/matlab -nodisplay'
	if const.singleCompThread:
		runcmd = '%s -singleCompThread -r "addpath(\'%s\'); %s; quit;"' % (matlabStr,const.codePath,'%s')
	else:
		runcmd = '%s -r "addpath(\'%s\'); %s; quit;"' % (matlabStr,const.codePath,'%s')

else:
	print "Code Language Not understood"
	exit()



#main file issuing jobs
mainFile = scriptPath + 'run_' + jobname + '.sh'
mf = open(mainFile,'w')
mf.write('#!/bin/bash\n')

#job command
#sbatchCmd = 'sbatch -o "%s" -J "%s" --sockets-per-node=1 ' + ('--mem=%d --time=%d:0:0') % (const.mem,const.hours) + ' -e "%s" %s \n'
sbatchCmd = 'sbatch -o "%s" -J "%s" --nodes=1 --ntasks-per-node=%d ' % ('%s','%s',const.cores) + ('--mem=%d --time=%d:0:0') % (const.mem,const.hours) + ' -e "%s" %s \n'

oFileList = []
for i,jobStr in enumerate(const.jobNames):
	filename = scriptPath + jobname + str(i) + '%s'
	oFileName = jobname + str(i)	
	oFileList.append(oFileName)

	with open(filename % '.sh','w') as f:
		f.write('#!/bin/bash\n')
		f.write('source ' + sourceFileName + '\n')
		f.write(runcmd % jobStr)
		f.close()	
	
	#Write script files to disk
	perm = 'chmod u+x '+ (filename % '.sh')
	call(perm,shell=True)
	jobstr = jobname + '-' + str(i)
	mf.write(sbatchCmd % ((filename % '.log'),jobstr,(filename % '.err'),(filename % '.sh')))

mf.close()
perm = 'chmod u+x ' + mainFile
call(perm,shell=True)

#Submit jobs to slurm
print 'Submitting Jobs'
submitJobs = 'bash ' + mainFile
call(submitJobs,shell=True)

	
