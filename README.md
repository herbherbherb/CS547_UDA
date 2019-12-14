This is for CS547 final project.
You should install:

pytorch >= 1.0.1

tensorflow >= 1.4.1

tensorboardX

fire

To run this code, you should do download first:

bash download.sh

After downloading data, run code like that:(Make sure I set debug mode and mdbp dataset as default.)

python main.py --debug False

If you want to run code in other dataset, for example, DBP dataset, you should run code as below:

python main.py --debug False --task dbp
