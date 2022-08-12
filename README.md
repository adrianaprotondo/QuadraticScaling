# Quadratic scaling 

In this project we investigate how adding parameters to a loss changes the geometry of the loss landscape in a way that can either be beneficial or detrimental to learning.  

We consider the case of a quadratic loss function as it is mathematically tractable. 

## Getting started

The whole project is implemented in [Julia](https://docs.Wjulialang.org/en/v1/).  

1. Download [Julia](https://julialang.org/downloads/#long_term_support_release).
2. cd to the folder where you want to clone the project.
3. clone the repo on your device
```
git init
git clone https://github.com/adrianaprotondo/QuadraticScaling.git
```
4. Install [Pluto.jl](https://www.juliapackages.com/p/pluto) to run the notebooks 
   1. Open a julia terminal
   2. add Pluto 
        ```
        ]
        add Pluto
        ```
    Using the package manager for the first time after installing Julia can take up to 15 minutes - hang in there!
5. Go back to the julia terminal (by pressing `ESC`) and use activate Pluto 
    ``` 
    using Pluto
    Pluto.run()
    ```
6. Pluto will launch in your browser. Navigate to open one of the notebooks `1Dto2D.jl` or `1DtoND.jl`
   

