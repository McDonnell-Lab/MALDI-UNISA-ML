call conda create -n MALDI_ML python=3.7 -y
call conda activate MALDI_ML
call conda install -y -c anaconda tensorflow-gpu==2.1
call conda install -y -c anaconda git
call conda install -y -c anaconda scikit-learn
call conda install -y -c anaconda matplotlib
call conda install -y -c anaconda pandas
call conda install -y -c anaconda jupyter
pip install pyimzml
pip install tables
pip install xlrd==1.2.0