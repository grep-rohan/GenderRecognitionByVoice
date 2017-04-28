library(warbleR)

#expects 2 arguments: location of the folder which contains all the scripts and sounds folder
# and name of the sound file with extension

args <- commandArgs(trailingOnly = TRUE)
if(length(args)!=2){

	fileConn<-file("failed")
	writeLines(c("Failed Execution"), fileConn)
	close(fileConn)

	print ("USAGE ERROR!")
	print ("Expected 2 Argument: Supplied"+length(args))
	
	stop()
}

location=args[1]
soundLocation = paste(location,"/sounds",sep="")
textLocation= paste(location,"/output",sep="")

filename=args[2]

fileInfo<-data.frame(sound.files=filename,selec=1,start=0,end=3)
result=specan(fileInfo,path=soundLocation)
setwd(textLocation)
write.table(result,"voiceDetails.txt",sep=",",row.names=FALSE)

