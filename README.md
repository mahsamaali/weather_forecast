# Prédiction des Températures à Montréal : Analyse et Modélisation Basées sur les Données Climatiques

## Introduction et Problématique

Le réchauffement climatique a entraîné une augmentation de 1,7 °C des températures moyennes au Canada entre 1948 et 2016, avec un taux de réchauffement deux fois plus élevé que la moyenne mondiale. Les effets sont particulièrement marqués dans le nord du pays et durant l'hiver, rendant la prédiction des températures de plus en plus complexe. Ce projet vise à prédire les températures annuelles à Montréal en utilisant des données historiques et à identifier les facteurs clés pour améliorer la classification des températures.

## Source des Données

Les données utilisées proviennent de [montreal.weatherstats.ca](https://montreal.weatherstats.ca/), qui compile des informations issues de bases de données gouvernementales canadiennes sur l'environnement et le changement climatique. Ces données sont librement accessibles pour un usage non commercial, avec l'obligation de citer la source : « weatherstats.ca based on Environment and Climate Change Canada data ».

## Structure du Projet

- `data/` : Contient les jeux de données climatiques historiques de Montréal.
- `src/` : Inclut les scripts Python pour le traitement des données, l'analyse et la modélisation.
- `results/` : Regroupe les visualisations, les modèles entraînés et les prédictions générées.
- `requirements.txt` : Liste des dépendances Python nécessaires à l'exécution du projet.

## Prérequis

Assurez-vous d'avoir Python 3.7 installé sur votre machine. Installez les dépendances requises avec :

```bash
pip install -r requirements.txt

## Utilisation

1. **Cloner le dépôt** :

   ```bash
   git clone https://github.com/mahsamaali/weather_forecast.git
   cd weather_forecast

2.**Exécuter les scripts**:
Utilisez les scripts du répertoire src/ pour prétraiter les données, entraîner les modèles et générer des prédictions.



3.**Visualiser les résultats**:
Les résultats, y compris les visualisations et les prédictions, seront sauvegardés dans le répertoire results/. 


