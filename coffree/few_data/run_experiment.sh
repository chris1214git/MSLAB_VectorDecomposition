mkdir -p record
if [ -z $1 ]
then
    echo "Experiment on MLP, BERT_GAN, and GAN..."
    bash "MLP.sh" &&
    bash "BERT_GAN.sh" &&
    bash "GAN.sh"
elif [ $1 == "MLP" ]
then
    echo "Experiment on MLP only..."
    bash "MLP.sh"
elif [ $1 == "BERT_GAN" ]
then
    echo "Experiment on BERT_GAN only..."
    bash "BERT_GAN.sh"
elif [ $1 == "GAN" ]
then
    echo "Experiment on GAN only..."
    bash "GAN.sh"
fi