echo "=============Start dataprocessing !================="

python dataprocess.py

echo "=============Start PCA reconstrruct !==============="

for n in 64 128 256 512 1024
do 
    python pca_reconstruct.py $n
done

echo "==============Start evaluate !======================"

for n in 64 128 256 512 1024
do 
    python evaluate.py $n
done