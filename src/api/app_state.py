from dataclasses import dataclass
from typing import Optional, Any
from mlflow.pyfunc import PyFuncModel


@dataclass
class AppState:
    '''
    AppState is a singleton class that holds the state of the application. It is 
    used to store the model and its version, as well as any other state that may 
    be needed in the future.

    Attributes:
    champ_model (Optional[PyFuncModel]):
        The current champion model used for recommendations
    champ_model_version (Optional[str]):
        The version of the champion model

    '''
    champ_model: Optional[PyFuncModel] = None
    champ_model_version: Optional[str] = None

