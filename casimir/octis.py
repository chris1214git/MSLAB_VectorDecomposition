import argparse
from octis.models.CTM import CTM
from octis.models.ETM import ETM
from octis.models.LDA import LDA
from octis.models.LSI import LSI
from octis.models.NMF import NMF
from octis.models.NeuralLDA import NeuralLDA
from octis.models.ProdLDA import ProdLDA
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='topic model baseline')
    parser.add_argument('--model', type=str, default="all")
    args = parser.parse_args()
    config = vars(args)

    # Define dataset
    dataset = Dataset()
    dataset.fetch_dataset("20NewsGroup")

    if config['model'] == 'all':
        for idx in range(7):
            for epochs in range(50):
                if idx == 0:
                    name = 'CTM'
                    model = CTM(num_topics=20)
                elif idx == 1:
                    name = 'ETM'
                    model = ETM(num_topics=20)
                elif idx == 2:
                    name = 'LDA'
                    model = LDA(num_topics=20, alpha=0.1)
                elif idx == 3:
                    name = 'LSI'
                    model == LSI(num_topics=20)
                elif idx == 4:
                    name = 'NMF'
                    model = NMF(num_topics=20)
                elif idx == 5:
                    name = 'NeuralLDA'
                    model = NeuralLDA(num_topics=20)
                elif idx == 6:
                    name = 'ProdLDA'
                    model = ProdLDA(num_topics=20)
            # Train the model using default partitioning choice 
                output = model.train_model(dataset)

                print(*list(output.keys()), sep="\n") # Print the output identifiers

                for t in output['topics'][:5]:
                    print(" ".join(t))

                # Initialize metric
                npmi = Coherence(texts=dataset.get_corpus(), topk=10, measure='c_npmi')

                # Initialize metric
                topic_diversity = TopicDiversity(topk=10)

                # Retrieve metrics score
                topic_diversity_score = topic_diversity.score(output)
                print('---- {} ----'.format(name))
                print("Topic diversity: "+str(topic_diversity_score))

                npmi_score = npmi.score(output)
                print("Coherence: "+str(npmi_score))
                print('------------')
                record = open('./'+name+'.txt', 'a')
                record.write(name)
                record.write('\n')
                record.write('Topic diversity: '+str(topic_diversity_score)+'\n')
                record.write('Coherence: '+str(npmi_score)+'\n')
                record.write('-------------------------------------------------\n')