python ide_gan.py --model=IDE_GAN --dataset=20news --ratio=0.1 | tee record/GAN_20news_0.1.txt &&
python ide_gan.py --model=IDE_GAN --dataset=20news --ratio=0.2 | tee record/GAN_20news_0.2.txt &&
python ide_gan.py --model=IDE_GAN --dataset=20news --ratio=0.4 | tee record/GAN_20news_0.4.txt &&
python ide_gan.py --model=IDE_GAN --dataset=20news --ratio=0.6 | tee record/GAN_20news_0.6.txt &&
python ide_gan.py --model=IDE_GAN --dataset=20news --ratio=0.8 | tee record/GAN_20news_0.8.txt