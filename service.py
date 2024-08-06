import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner=bentoml.sklearn.get("iris_clf:latest").to_runner()
svc=bentoml.Service("iris_classifier",runners=[iris_clf_runner])# any no of models we can given in the list

@svc.api(input=NumpyNdarray(),output=NumpyNdarray()) #svc is the service model name
def classify(input_series:np.ndarray)->np.ndarray:
    result=iris_clf_runner.predict.run(input_series)
    return result

'''
#  to run this file in bentoml serve service.py:svc --reload
# go to goggle chrome write "localhost:3000"
# open request body clcik it on try it now give some[1,3.2,4.3,1.0 an click on execute]'''
 #bento ml build will take the total folder and deploy it anywhere.
