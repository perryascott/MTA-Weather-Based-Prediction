# MTA Weather-Based Prediction

This repository contains code and data for predicting MTA ridership using weather data. The project aims to develop a model that can accurately predict ridership based on weather conditions.

Medium Article discussing some of the results and methodologies [here](https://medium.com/@perryascott2/cloudy-with-a-chance-of-commuters-weathers-impact-on-nyc-transit-91678e2a8198).

### TODO:
create predict_ridership.ipynb notebook
- input variable search. What combination of variables give the best validation performance and for what models?
- Use more event based model that only considers situations that meet a condition (i.e. when precipitation > 0)
- Use residuals for certain weather metrics? (Only when temp diverges from norm by a lot)
- Make a search for the best model type and the input parameters. The critieria should be lowest validation MAE (or RMSE) for certain conditions, select 12 conditions that show strong correlations based on the correlation data.
