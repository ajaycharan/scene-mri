#Language
codeLanguage = 'matlab' #'python'

#jobs prms
mem=8*1024 #in MB
hours = 96 #time 
cores = 1

#jobName
jobName = 'vq'

if codeLanguage=='matlab':
	if cores==1:
		singleCompThread = True
	
elif codeLanguage=='python':
	moduleName = ''


#Paths
codePath = '/auto/k1/pulkit/Codes/scene/'
resultPath = '/auto/k6/pulkit/data/scene/final_results/'
scriptPath = codePath + 'tmp/'

#Constants
encType = ['vq']
vocSz = [5000]
#trainPercent = [0.25,0.50,0.75,1]
#trainPercent = [1]
#runNum = [0,1,2,3,4]
#encType = ['gabor']
#minSz = [4]
#maxSz = [8,16,24,32]

#Specify the jobs
jobNames = []
for e in encType:
	if e=='gabor':
		for mn in minSz:
			for mx in maxSz:
				callName = 'process_data(\'%s\',%d,%d,%d)' % (e,mn,1,mx)
				jobNames.append(callName)
	else:
		for v in vocSz:
			#for r in runNum:
			#callName = 'process_data(\'%s\',%d,%f,%d)' % (e,v,1,r)
			callName = 'process_data(\'%s\',%d,%f)' % (e,v,1)
			jobNames.append(callName)






