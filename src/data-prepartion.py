import pandas as pd
import numpy as np
from numpy import savetxt

file = 'weatherstats_montreal_donnees_finales.xlsx'

dataset = pd.read_excel(file, sheet_name='in')

date=dataset.iloc[:, 0:1].values
avgWind=dataset.iloc[:, 4:5].values
temperature=dataset.iloc[:, 3:4].values
precipitation=dataset.iloc[:, 5:6].values
snow=dataset.iloc[:, 6:7].values
avg_cloud=dataset.iloc[:, 7:8].values
print(avg_cloud)


#modification de la date
# date_filter=np.where((date ==1 )  ,"JAN",date)
# date_filter=np.where((date ==2 )  ,"FEV",date_filter)
# date_filter=np.where((date ==3 )  ,"MAR",date_filter)
# date_filter=np.where((date ==4 )  ,"AVR",date_filter)
# date_filter=np.where((date ==5 )  ,"MAI",date_filter)
# date_filter=np.where((date ==6 )  ,"JUN",date_filter)
# date_filter=np.where((date ==7 )  ,"JUL",date_filter)
# date_filter=np.where((date ==8 )  ,"AOU",date_filter)
# date_filter=np.where((date ==9 )  ,"SEP",date_filter)
# date_filter=np.where((date ==10 )  ,"OCT",date_filter)
# date_filter=np.where((date ==11 )  ,"NOV",date_filter)
# date_filter=np.where((date ==12 )  ,"DEC",date_filter)
date_filter=np.where((date ==1 )  ,"1",date)
date_filter=np.where((date ==2 )  ,"2",date_filter)
date_filter=np.where((date ==3 )  ,"3",date_filter)
date_filter=np.where((date ==4 )  ,"4",date_filter)
date_filter=np.where((date ==5 )  ,"5",date_filter)
date_filter=np.where((date ==6 )  ,"6",date_filter)
date_filter=np.where((date ==7 )  ,"7",date_filter)
date_filter=np.where((date ==8 )  ,"8",date_filter)
date_filter=np.where((date ==9 )  ,"9",date_filter)
date_filter=np.where((date ==10 )  ,"10",date_filter)
date_filter=np.where((date ==11 )  ,"11",date_filter)
date_filter=np.where((date ==12 )  ,"12",date_filter)
dataset["date"]=date_filter

#modification des temperatures
temperature_filter=np.where((temperature >=-30 ) & (temperature <= -10) ,"0",temperature)
temperature_filter=np.where((temperature >-10 ) & (temperature <= 0) ,"1",temperature_filter)
temperature_filter=np.where((temperature >0 ) & (temperature <= 10) ,"2",temperature_filter)
temperature_filter=np.where((temperature >10 ) & (temperature <= 20) ,"3",temperature_filter)
temperature_filter=np.where((temperature >20 ),"4",temperature_filter)
# temperature_filter=np.where((temperature >=-30 ) & (temperature <= -10) ,"[-30,-10]",temperature)
# temperature_filter=np.where((temperature >-10 ) & (temperature <= 0) ,"]-10,0]",temperature_filter)
# temperature_filter=np.where((temperature >0 ) & (temperature <= 10) ,"]0,10]",temperature_filter)
# temperature_filter=np.where((temperature >10 ) & (temperature <= 20) ,"]10,20]",temperature_filter)
# temperature_filter=np.where((temperature >20 ),"]20,...]",temperature_filter)
dataset["avg_temp (celsius)"]=temperature_filter


#modification du vent
avgWind_filter = np.where((avgWind >=0 ) & (avgWind < 12.5) ,"W1",avgWind)
avgWind_filter = np.where((avgWind >=12.5 ) & (avgWind <= 20.5) ,"W2",avgWind_filter)
avgWind_filter = np.where((avgWind >20.5 )  ,"W3",avgWind_filter)
# avgWind_filter = np.where((avgWind >=0 ) & (avgWind < 12.5) ,"[0,12.5]",avgWind)
# avgWind_filter = np.where((avgWind >=12.5 ) & (avgWind <= 20.5) ,"]12.5,20.5]",avgWind_filter)
# avgWind_filter = np.where((avgWind >20.5 )  ,"]20.5,...]",avgWind_filter)
dataset['avg_wind_speed (km/h)']=avgWind_filter


#modification de la pluie

precipitation_filter=np.where((precipitation >=0 ) & (precipitation <= 2),"P1",precipitation)
precipitation_filter=np.where((precipitation >2 ) & (precipitation <= 5),"P2",precipitation_filter)
precipitation_filter=np.where((precipitation >5 ) & (precipitation <= 30),"P3",precipitation_filter)
precipitation_filter=np.where((precipitation >30 ) ,"P4",precipitation_filter)
# precipitation_filter=np.where((precipitation >=0 ) & (precipitation <= 2),"[0,2]",precipitation)
# precipitation_filter=np.where((precipitation >2 ) & (precipitation <= 5),"]2,5]",precipitation_filter)
# precipitation_filter=np.where((precipitation >5 ) & (precipitation <= 30),"]5,30]",precipitation_filter)
# precipitation_filter=np.where((precipitation >30 ) ,"]30,...]",precipitation_filter)
dataset['precipitation (mm)']=precipitation_filter
pd.DataFrame(precipitation_filter).to_csv("data.csv")
#modification de la neige

snow_filter=np.where((snow >=0 ) & (snow < 1),"S1",snow)
snow_filter=np.where((snow >=1 ) & (snow <= 9.99),"S2",snow_filter)
snow_filter=np.where((snow >=10 ),"S3",snow_filter)

# snow_filter=np.where((snow >=0 ) & (snow < 1),"[0,1[",snow)
# snow_filter=np.where((snow >=1 ) & (snow <= 9.99),"[1,9.99]",snow_filter)
# snow_filter=np.where((snow >=10 ),"[10,...]",snow_filter)
dataset['snow (cm)']=snow_filter


# modification de avg_cloud
avg_cloud_filter=np.where((avg_cloud >=0 ) & (avg_cloud <= 2),"C1",avg_cloud)
avg_cloud_filter=np.where((avg_cloud >2 ) & (avg_cloud <=5.99 ),"C2",avg_cloud_filter)
avg_cloud_filter=np.where((avg_cloud >=6 ) ,"C3",avg_cloud_filter)

# avg_cloud_filter=np.where((avg_cloud >=0 ) & (avg_cloud <= 2),"[0,2]",avg_cloud)
# avg_cloud_filter=np.where((avg_cloud >2 ) & (avg_cloud <=5.99 ),"]2,5.99]",avg_cloud_filter)
# avg_cloud_filter=np.where((avg_cloud >=6 ) ,"[6,...]",avg_cloud_filter)
dataset['avg_cloud (oktas)']=avg_cloud_filter



# enregistrer dans une fichier exel
dataset.to_excel("data.xlsx","data")
#save to csv file
# pd.DataFrame(avg_cloud_filter).to_csv("data.csv")



