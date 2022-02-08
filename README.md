# FermiLatExploration
Questo progetto python permette di navigare i dati del catalogo 4FGL del telescopio Fermi Lat 


## Installazione
#### Prerequisito utenti Windows
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