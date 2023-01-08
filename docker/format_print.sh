#!/bin/bash

REST='\e[0m';
GREEN='\e[0;32m';
BGREEN='\e[7;32m';
BRED='\e[7;31m';
Cyan='\033[0;36m';
BCyan='\033[7;36m'

printd(){            
    
    if [ -z $2 ];then COLOR=$REST
    elif [ $2 = "G" ];then COLOR=$GREEN
    elif [ $2 = "R" ];then COLOR=$BRED
    elif [ $2 = "Cy" ];then COLOR=$Cyan
    elif [ $2 = "BCy" ];then COLOR=$BCyan
    else COLOR=$REST
    fi

    echo -e "${COLOR}$1${REST}"
}
