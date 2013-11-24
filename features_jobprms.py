#Language
codeLanguage = 'matlab' #'python'

#jobs prms
mem=4*1024 #in MB
hours = 24 #time 

#jobName
jobName = 'feat'

if codeLanguage=='matlab':
	singleCompThread = False
	sockets = 1
	cores = 1
elif codeLanguage=='python':
	moduleName = ''


#Paths
codePath = '/auto/k1/pulkit/Codes/scene/'
scriptPath = codePath + 'tmp/'

#Constants
vocSize = [8,16,32,64,128]
encType = ['fisher']
#encType = ['gabor']
#sfMin = [4]
#sfMax = [8,16,24,32,40]

#Specify the jobs
jobNames = []
for e in encType:
	if e=='gabor':
		for mn in sfMin:
			for mx in sfMax:
				callName = 'gabor_features(%d,%d)' % (mn,mx)
				jobNames.append(callName)
	else:
		for v in vocSize:
			callName = 'my_scene(%d,\'%s\')' % (v,e)
			jobNames.append(callName)






