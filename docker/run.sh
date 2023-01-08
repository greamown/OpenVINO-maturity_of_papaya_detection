#!/bin/bash
# ---------------------------------------------------------
# Set the default value of the getopts variable 
server=false
# ---------------------------------------------------------
# color ANIS
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'
# ---------------------------------------------------------
function papaya(){
echo -e "${GREEN}"
echo "
==========================================
   ____   __    ____   __   _  _   __   
  (  _ \ /__\  (  _ \ /__\ ( \/ ) /__\  
   )___//(__)\  )___//(__)\ \  / /(__)\ 
  (__) (__)(__)(__) (__)(__)(__)(__)(__)
 
==========================================
"
echo -e "${NC}"
}

papaya
# ---------------------------------------------------------
# help
function help(){
	echo "-----------------------------------------------------------------------"
	echo "Run the OpenVINO environment."
	echo
	echo "Syntax: scriptTemplate [-sh]"
	echo "options:"
	echo "s		Server mode for non vision user"
	echo "h		help."
	echo "-----------------------------------------------------------------------"
}
while getopts "sh" option; do
	case $option in
		s )
			server=true
			;;
		h )
			help
			exit
			;;
		\? )
			help
			exit
			;;
		* )
			help
			exit
			;;
	esac
done

# ---------------------------------------------------------
sudo apt-get install -qy boxes > /dev/null 2>&1
# ---------------------------------------------------------
# Setup variable
docker_image="openvino_cv:latest"
mount_camera=""
set_vision=""
command="bash"
workspace="/workspace"
docker_name="openvino_cv"
# ---------------------------------------------------------
# SERVER or DESKTOP MODE
if [[ ${server} = false ]];then
	mode="DESKTOP"
	set_vision="-v /etc/localtime:/etc/localtime:ro -v /tmp/.x11-unix:/tmp/.x11-unix:rw -e DISPLAY=unix${DISPLAY}"
	# let display could connect by every device
	xhost + > /dev/null 2>&1
else
	mode="SERVER"
fi

# ---------------------------------------------------------
# Combine Camera option
all_cam=$(ls /dev/video*)
cam_arr=(${all_cam})

for cam_node in "${cam_arr[@]}"
do
    mount_camera="${mount_camera} --device ${cam_node}:${cam_node}"
done

# ---------------------------------------------------------
title="\n\
MODE:  ${mode}\n\
DOCKER: ${docker_name} \n\
CAMERA:  $((${#cam_arr[@]}/2))\n\
COMMAND: ${command}"

# ---------------------------------------------------------
# Run container
docker_cmd="docker run \
--user root \
--rm -it \
--name ${docker_name} \
--net=host --ipc=host \
-w ${workspace} \
-v `pwd`:${workspace} \
--privileged \
-v /dev:/dev \
${mount_camera} \
${set_vision} \
${docker_image} \"${command}\""

echo ""
echo -e "Command: ${docker_cmd}"
echo ""
bash -c "${docker_cmd}"
