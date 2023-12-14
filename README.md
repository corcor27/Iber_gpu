# How to create an enviorment for using tensorflow with gpu support on the ibers cluster 

If you need anaconda start at 1, if annaconda installed go to 3.  

1) first install a version of anaconda to contain your enviroment. I personally used Miniconda as it solves dependencys faster than the base system. https://docs.conda.io/projects/miniconda/en/latest/  

2) intiate your conda enviroment with "conda init" and then reset your connection for the update to take place.  

3) Now to create your anaconda enviroment please use the .yml file constained within this repositry. to this enter "conda env create --name envname --file=ibers_env_gpu.yml". This will create you an enviroment called IBERS_GPU  

4) activate your enviroment with "conda activate IBERS_GPU"  

5) Now you need to map the cudatookit and cudnn files, so the gpu can actually see them. To do this enter all the follwoing commands inorder:
## Create the directories to place our activation and deacivation scripts in  
mkdir -p $CONDA_PREFIX/etc/conda/activate.d  
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d  

## Add commands to the scripts there are two lines to map the files sperated by printf
printf 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}\nexport LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib/\n' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh  

printf 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}\nunset OLD_LD_LIBRARY_PATH\n' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh  

source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh  

You only have to entre these 5 commands once

6) To test run "run-ABER.sh" and in your out put file you should find that it displays the gpu

7) The "run-ABER.sh" is your batch file to submit the job, to run your file simply change the python file named to your script that you want to run.

you should now be good to go.   
