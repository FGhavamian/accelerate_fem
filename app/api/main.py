from typing import List, Optional
import argparse
import os

import uvicorn
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import matplotlib.pyplot as plt

from accelerate_simulations.predict import Predictor, max_per_case


print('[INFO] starting predictor ...')
make_prediction = Predictor(
    '../../examples/box_w_aggr/saved_model', 
    circle_density=0.9,  
    box_size=(200,200), 
    gap=10,
    resolution=(128,128),
    names_boundary = ['circles_boundaries']
)
print('[INFO] warming up the predictor ...')
make_prediction(1, (20,50))


app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Case(BaseModel):
    name: str
    n_samples: int
    radius_range: List[int]


class PredictResponse(BaseModel):
    status: int
    data: Optional[List[float]] = None 


@app.put("/predict/", response_model=List[PredictResponse])
async def predict(
    name: str,
    cases: List[Case]
):
    # prediction
    print('[INFO] making predictions ...')
    results = []
    for case in cases:
        print(f'[INFO] \tcase {case.name}')
        try:
            plastic_strains = make_prediction(
                case.n_samples, 
                case.radius_range
            )

            plastic_strains_max = max_per_case(plastic_strains)

            results.append(
                PredictResponse(status=200, data=plastic_strains_max.tolist()))

        except:
            results.append(
                PredictResponse(status=400, data=None))

    return results


if __name__ == "__main__":
    uvicorn.run('main:app', host='127.0.0.1', port=8000)