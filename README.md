# GroceryAssistant

The goal of this project is to build a simple Intelligent Process Automation(IPA) robot that does the followings:
1. Provides recipe recommendations based on user inputs
2. Automates the buying process on online grocer platforms like FairPrice and Redmart. (based on Singapore context)

### Embedding projection
Files in `viz` folder can be used to visualize the embedding vector space using [Tensorflow's Embedding Projector](https://projector.tensorflow.org/).

## Setup
### 1. Create main directory
> mkdir {your-path}/GroceryAssistant \
> cd {your-path}/GroceryAssistant
### 2. Download repo files
> git clone https://github.com/syalabi-seet/GroceryAssistant.git
### 3. Download data files from [link](https://drive.google.com/file/d/1K2opCI32NBybaEwi-EjiY58gEwfmcx8h/view?usp=sharing)
### 4. Extract files into data/ folder
### 5. Create new conda environment
> Open anaconda prompt \
> conda create --name GroceryAssistant python==3.8.13 -y \
> conda activate GroceryAssistant \
> pip install -r requirements.txt
### Run app.py
> streamlit run d:/School-stuff/Sem-3/RecipeRecommender/src/app.py
## References
