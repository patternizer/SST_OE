#!/bin/sh

FIDHOME=/gws/nopw/j04/fiduceo/Users/mtaylor/sst_oe
INMYVENV=${FIDHOME}/inmyvenv.sh
GENERATE=${FIDHOME}/run-oe-prepare.py
MEM_REQ=60000 # in MB
MEM_MAX=60000 # in MB 

submit(){
    SAT=$1     
    START=$2
    END=$3 
    for year in $(seq $START $END)  
    do
        JOBNAME="${SAT}${year}"
	LOGFILE=$(echo $SAT $year | awk '{ printf "run.%s_%4.4d.log", $1, $2}')
	LOGDIR="$FIDHOME"	       
	if [ -d "$LOGDIR" ]; then
	    echo ${JOBNAME}
            LOGFILE=${LOGDIR}/${LOGFILE}		    		   
 	    bsub -q short-serial -W24:00 -R "rusage[mem=$MEM_REQ]" -M $MEM_MAX -oo ${LOGFILE} -J "$JOBNAME" $INMYVENV $GENERATE "${SAT}" "${year}"
	fi
    done    
}

submit MTA 2006 2017
wait


