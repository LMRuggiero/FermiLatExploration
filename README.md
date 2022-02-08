# FermiLatExploration
Questo progetto python permette di navigare i file fits raccolti dal telescopio Fermi Lat.
Al momento è disponibile soltanto il pacchetto [explore4FGL](https://gitlab.com/LMRuggiero/fermilatexploration/-/tree/main/explore4FGL) che permette di analizzare i dati del catalogo 4FGL

## Installazione
### Prerequisito utenti Windows
Per utenti Windows è necessario installare la [WSL 2](https://docs.microsoft.com/it-it/windows/wsl/install)

### Prima esecuzione
E' necessario creare ed attivare un ambiente virtuale coi pacchetti presenti nel [requirement.txt](https://gitlab.com/LMRuggiero/fermilatexploration/-/blob/main/requirements.txt)
Per farlo basta eseguire i seguenti comandi:
```
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```
### Esecuzioni successive
Ogni volta che si riapre il progetto bisogna riattivare l'ambiente python

## Utilizzo
Un esempio di utilizzo del pacchetto explore4FGL è presente nel [main](https://gitlab.com/LMRuggiero/fermilatexploration/-/blob/main/main.py) del progetto.
E' possibile ottenere altri tipi di rappresentazioni grafiche accingendo direttamente ai metodi della classe [_Source4FGLData_](https://gitlab.com/LMRuggiero/fermilatexploration/-/blob/main/explore4FGL/explore4FGL.py) definita all'interno del pacchetto