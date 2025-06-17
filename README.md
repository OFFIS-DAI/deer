# DEER
Decentralizing Redispatch in Distributed Energy Systems

## 💾 Overview
<p align="justify">
The DEER project aims to optimize the integration of small-scale assets into the redispatch process, ensuring grid stability and avoiding network congestion.
This project leverages multi-agent systems (MAS) to manage the complexity and scalability of integrating numerous 
small-scale assets, such as battery storages and heat pumps, into the redispatch process.
The use of MAS provides modularity and scalability, where individual agent entities represent distinct assets or components of the system these agents work collaboratively to ensure an optimized and stable redispatch process within a simulation environment.
</p>

## 🗒️ Installation and Setup

1. Install Python >= 3.9.
2. Clone the project repository.
    ```console
    git clone <repository_url>
    ```
3. Navigate to the project folder and install dependencies using the following command:
   ```console
   pip install -r requirements.txt
   ```
4. In case the project reports an issue related to the CBC solver, install the required dependencies based on your operating system:
   - **For Linux**:  
     ```console
     sudo apt-get install coinor-cbc coinor-libcbc-dev
     ```
   - **For Windows**:  
     Download and set up the [CBC solver from Coin-OR](https://github.com/coin-or/Cbc/releases).

## 💻Running the comparison

To start the flexibility calculation and aggregation comparison, start the script in `simulation/simulation_studies/run_calculations.py`. 