# FermiLatExploration

Questo progetto python permette di navigare i file fits raccolti dal telescopio Fermi Lat. Al momento è disponibile
soltanto il pacchetto [_explore4FGL_](https://gitlab.com/LMRuggiero/fermilatexploration/-/tree/main/explore4FGL) che
permette di analizzare i dati del catalogo 4FGL

## Installazione

### Prerequisito utenti Windows

Per utenti Windows è necessario installare la [WSL 2](https://docs.microsoft.com/it-it/windows/wsl/install)

### Installazione progetto

Fare il clone del progetto sulla directory scelta:

```
git clone https://gitlab.com/LMRuggiero/fermilatexploration.git
```

### Prima esecuzione

E' necessario creare ed attivare un ambiente virtuale coi pacchetti presenti
nel [_requirement.txt_](https://gitlab.com/LMRuggiero/fermilatexploration/-/blob/main/requirements.txt).

Per farlo bisogna prima entrare nella directory del progetto clonato:

```
cd fermilatexploration 
```

e successivamente eseguire i seguenti comandi:

```
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

### Esecuzioni successive

Ogni volta che si riapre il progetto bisogna riattivare l'ambiente python:

```
source venv/bin/activate
```

### Fine esecuzione

Per uscire dall'ambiente virtuale

```
deactivate
```

## Utilizzo

Un esempio di utilizzo del pacchetto explore4FGL è presente
nel [_main_](https://gitlab.com/LMRuggiero/fermilatexploration/-/blob/main/main.py) del progetto. Per eseguire il main:

```
python3 main.py
```

Tutti i risultati di questa run sono visibili nella folder _output_ che verrà creata automaticamente nella radice del
progetto.

__N.B.:__ E' possibile ottenere altri tipi di rappresentazioni grafiche modificando il file main.py accingendo direttamente
ai metodi della classe [_Source4FGLData_](https://gitlab.com/LMRuggiero/fermilatexploration/-/blob/main/explore4FGL/explore4FGL.py) definita
all'interno di questo pacchetto.

## Documentazione

Sono consigliate le seguenti documentazioni:

* [Fermi Large Area Telescope Fourth Source Catalog](https://gitlab.com/LMRuggiero/fermilatexploration/-/blob/main/documents/1902.10045.pdf)
  per una maggior comprensione dei dati raccolti dal Fermi LAT
* [documentazione del progetto](https://fermilatexploration.readthedocs.io/en/latest/index.html)