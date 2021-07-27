# GanLib
Um verschiedene Arten von GANs zu trainieren, folgenden Befehl nutzen:

```python3 main.py --epochs 500 --batch_size 32 --model gan --dataset svhn```

Werte für **model** könnten sein:
```--model = (gan, dcgan, wgan, dragan)```

Werte für **dataset** könnten sein:
```--model = (mnist, svhn)```


# Convolutional VAE
Um ein *CVAE* auf SVHN zu trainieren, folgenden Befehl nutzen:

```python3 cvae_svhn2.py```


# Speicherung
Ergebnisse werden im Ordner *results/* gespeichert.
Die .h5 Files werden im Ordner *save/* gespeichert.

# Info über Files
* **classifier.py** (Ist ein Classifier für den SVHN Datensatz.)
* **cvae_svhn.py** (War ein Test auf den Latent Space zuzugreifen, vom CVAE.)
* **sample_cvae.py** (Generiert weitere Samples und Latent Space)
* **vae.py** und **vae2.py** (Sind jeweils für den MNIST Datensatz. Hier auch vae2.py zum trainieren nehmen!)